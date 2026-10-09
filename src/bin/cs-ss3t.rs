// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-ss3t`: Single-Shell 3-Tissue CSD reconstruction.
//!
//! Loads a single-shell DWI bundle + three MRtrix-format tissue response
//! files (WM, GM, CSF) and writes three NIfTIs of SH coefficients
//! (`_wm`, `_gm`, `_csf`) plus one JSON sidecar capturing the algorithm
//! parameters and provenance.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

/// CLI flavour of [`LmaxWmStrategy`] — flat for clap.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum LmaxWmStrategyFlag {
    Fixed,
    PathBic,
}

use cs_dmri::io::aux::{sibling_path, write_3d_f32, write_3d_u8, write_3d_u32_as_f32, write_4d_f32};
use cs_dmri::io::dwi::load_dwi;
use cs_dmri::multitissue::sidecar::{
    NCoefficients, ResponseProvenance, Ss3tSidecar, Ss3tSolverParams,
};
use cs_dmri::multitissue::ss3t::{LmaxWmStrategy, Ss3tConfig, Ss3tResponses};
use cs_dmri::multitissue::volume::{Ss3tFitConfig, fit_volume_ss3t_reporting};
use cs_dmri::multitissue::TissueResponse;
use cs_dmri::qspace::{BvecFrame, TORTOISE_DEFAULT_GMAX};
use cs_dmri::solver::icls::IclsConfig;
use cs_dmri::{
    CsDmriError, Heartbeat, ProvenanceBuilder, ProvenanceMode, atomic_write,
    effective_thread_count,
};

#[derive(Parser, Debug)]
#[command(version, about = "Single-Shell 3-Tissue CSD (Dhollander 2016) — native Rust port")]
struct Cli {
    /// 4D DWI NIfTI input (one b=0 shell + one DWI shell).
    #[arg(long)]
    dwi: PathBuf,
    /// FSL bval file.
    #[arg(long)]
    bval: PathBuf,
    /// FSL bvec file.
    #[arg(long)]
    bvec: PathBuf,
    /// Optional brain mask NIfTI. Auto-generated from b0 mean if absent.
    #[arg(long)]
    mask: Option<PathBuf>,

    /// MRtrix-format WM single-fibre response (.txt). Two rows expected:
    /// b=0 (isotropic) and the DWI shell (anisotropic, lmax ≥ requested
    /// `--lmax-wm`).
    #[arg(long)]
    response_wm: PathBuf,
    /// MRtrix-format GM response (.txt). Two rows, one column each (lmax = 0).
    #[arg(long)]
    response_gm: PathBuf,
    /// MRtrix-format CSF response (.txt). Two rows, one column each (lmax = 0).
    #[arg(long)]
    response_csf: PathBuf,

    /// Output WM FOD NIfTI (4D, n_sh_wm volumes).
    #[arg(long)]
    output_wm: PathBuf,
    /// Output GM compartment NIfTI (4D, 1 volume).
    #[arg(long)]
    output_gm: PathBuf,
    /// Output CSF compartment NIfTI (4D, 1 volume).
    #[arg(long)]
    output_csf: PathBuf,

    /// SS3T outer iterations (default 3, must be ≥ 2).
    #[arg(long, default_value_t = 3)]
    niter: u32,
    /// b=0 contribution as a percentage of the non-b=0 volumes (default 10).
    #[arg(long, default_value_t = 10.0)]
    bzero_pct: f64,
    /// WM SH-order strategy. `fixed` uses a single lmax (the `--lmax-wm`
    /// value) for every voxel — qsirecon parity. `path-bic` sweeps the
    /// candidates listed in `--lmax-wm-candidates` per voxel and picks
    /// the one minimising BIC. CSF/GM voxels auto-select lmax=0 (fast);
    /// only crossing-fiber WM benefits from lmax=8.
    #[arg(long, value_enum, default_value_t = LmaxWmStrategyFlag::Fixed)]
    lmax_wm_strategy: LmaxWmStrategyFlag,
    /// `--lmax-wm-strategy=fixed` only: WM SH order. Clamped to the WM
    /// response file's lmax.
    #[arg(long, default_value_t = 8)]
    lmax_wm: usize,
    /// `--lmax-wm-strategy=path-bic` only: comma-separated list of
    /// candidate even lmaxes (default: 0,2,4,6,8).
    #[arg(long, value_delimiter = ',', default_values_t = vec![0_usize, 2, 4, 6, 8])]
    lmax_wm_candidates: Vec<usize>,

    /// Inner ICLS active-set iterations.
    #[arg(long, default_value_t = 200)]
    icls_max_iter: usize,
    /// Inner ICLS constraint tolerance: a constraint is "satisfied" if
    /// `(C x)_i ≥ -tol`.
    #[arg(long, default_value_t = 1e-10)]
    icls_tol: f64,
    /// Inner ICLS Tikhonov stabiliser ε added to HᵀH diagonal to guarantee
    /// strict positive-definiteness.
    #[arg(long, default_value_t = 1e-10)]
    icls_epsilon: f64,

    /// Big delta Δ (seconds). Used by `qspace::GradientTable` for any
    /// downstream q-space derivation; SS3T itself does not need it.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Small delta δ (seconds).
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude (T/m), used only when deltas are estimated.
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,

    /// Also write per-voxel diagnostic NIfTIs (`_iters.nii.gz`,
    /// `_residual.nii.gz`, `_converged.nii.gz`) next to the WM output.
    #[arg(long)]
    diagnostics: bool,

    /// Keep bvecs in their FSL/image-axis frame instead of rotating them
    /// into world-RAS. Default rotation matches `cs-fit`'s convention.
    #[arg(long)]
    no_bvec_rotation: bool,

    /// Cap rayon's worker threads. If unset, picks up `$SLURM_CPUS_PER_TASK`,
    /// then `$RAYON_NUM_THREADS`, else uses one worker per logical CPU.
    #[arg(long)]
    threads: Option<usize>,

    /// Allow overwriting existing outputs (default: refuse).
    #[arg(long)]
    overwrite: bool,

    /// Suppress progress heartbeat and per-step summary lines.
    #[arg(long)]
    quiet: bool,

    /// Seconds between heartbeat lines during the per-voxel fit (default 30).
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance captured into the sidecar JSON.
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let provenance_builder = ProvenanceBuilder::new("cs-ss3t", args.provenance);
    let (threads, source) = cs_dmri::configure_rayon_threads(args.threads)?;
    if !args.quiet {
        let n = if threads == 0 { effective_thread_count() } else { threads };
        eprintln!("[cs-ss3t] threads={} source={}", n, source.as_str());
    }

    let bvec_frame = if args.no_bvec_rotation {
        BvecFrame::ImageAxis
    } else {
        BvecFrame::WorldRas
    };
    let dwi = load_dwi(
        &args.dwi,
        &args.bval,
        &args.bvec,
        args.mask.as_deref(),
        args.big_delta,
        args.small_delta,
        Some(args.gmax),
        bvec_frame,
    )
    .with_context(|| "failed to load DWI bundle")?;

    let mask_voxels = dwi.mask.iter().filter(|&&v| v).count();
    if !args.quiet {
        eprintln!(
            "[cs-ss3t] DWI shape {:?}, mask voxels: {}",
            dwi.shape(),
            mask_voxels
        );
    }

    let responses = Ss3tResponses {
        wm: TissueResponse::parse_mrtrix_txt(&args.response_wm)
            .with_context(|| format!("parse WM response {:?}", args.response_wm))?,
        gm: TissueResponse::parse_mrtrix_txt(&args.response_gm)
            .with_context(|| format!("parse GM response {:?}", args.response_gm))?,
        csf: TissueResponse::parse_mrtrix_txt(&args.response_csf)
            .with_context(|| format!("parse CSF response {:?}", args.response_csf))?,
    };

    let lmax_strategy = match args.lmax_wm_strategy {
        LmaxWmStrategyFlag::Fixed => LmaxWmStrategy::Fixed(args.lmax_wm),
        LmaxWmStrategyFlag::PathBic => LmaxWmStrategy::PathBic(args.lmax_wm_candidates.clone()),
    };
    let cfg = Ss3tConfig {
        niter: args.niter,
        bzero_pct: args.bzero_pct,
        lmax_wm: lmax_strategy,
        icls: IclsConfig {
            max_iter: args.icls_max_iter,
            tol: args.icls_tol,
            epsilon: args.icls_epsilon,
        },
    };

    let interval = Duration::from_secs(args.progress_interval_secs.max(1));
    let heartbeat = Heartbeat::new("cs-ss3t", mask_voxels, interval, args.quiet);

    let result = fit_volume_ss3t_reporting(
        &dwi,
        &responses,
        &cfg,
        Ss3tFitConfig { compute_diagnostics: args.diagnostics },
        || heartbeat.tick(),
    )
    .with_context(|| "ss3t fit failed")?;

    heartbeat.finish();

    let threads_used = effective_thread_count();
    let provenance = provenance_builder.map(|b| b.finish(threads_used));

    let sidecar = Ss3tSidecar {
        method: "ss3t".to_string(),
        niter: cfg.niter,
        bzero_pct: cfg.bzero_pct,
        lmax_wm: result.plan.candidates.iter().copied().max().unwrap_or(0),
        lmax_wm_candidates: result.plan.candidates.clone(),
        bzero_sw: result.plan.bzero_sw,
        icls: Ss3tSolverParams {
            max_iter: cfg.icls.max_iter,
            tol: cfg.icls.tol,
            epsilon: cfg.icls.epsilon,
        },
        n_sphere_dirs: odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras().len(),
        sphere_id: "dsistudio_odf8".to_string(),
        responses: ResponseProvenance {
            wm_path: args.response_wm.clone(),
            gm_path: args.response_gm.clone(),
            csf_path: args.response_csf.clone(),
        },
        n_coefficients: NCoefficients {
            wm: result.plan.max_n_sh_wm,
            gm: 1,
            csf: 1,
        },
        bvec_frame: dwi.bvec_frame,
        provenance,
    };
    let sidecar_path = sidecar_for(&args.output_wm);
    let sidecar_json = serde_json::to_string_pretty(&sidecar)
        .map_err(|e| CsDmriError::Other(format!("sidecar serialize: {e}")))?;

    write_4d_f32(&args.output_wm, &args.dwi, &result.wm, args.overwrite)
        .with_context(|| format!("write WM FOD {:?}", args.output_wm))?;
    write_4d_f32(&args.output_gm, &args.dwi, &result.gm, args.overwrite)
        .with_context(|| format!("write GM {:?}", args.output_gm))?;
    write_4d_f32(&args.output_csf, &args.dwi, &result.csf, args.overwrite)
        .with_context(|| format!("write CSF {:?}", args.output_csf))?;
    atomic_write(&sidecar_path, args.overwrite, |tmp| {
        std::fs::write(tmp, &sidecar_json).map_err(CsDmriError::from)
    })
    .with_context(|| format!("write sidecar {:?}", sidecar_path))?;

    if !args.quiet {
        eprintln!(
            "[cs-ss3t] wrote WM ({} SH coeffs) → {}",
            result.plan.max_n_sh_wm,
            args.output_wm.display()
        );
        eprintln!("[cs-ss3t] wrote GM → {}", args.output_gm.display());
        eprintln!("[cs-ss3t] wrote CSF → {}", args.output_csf.display());
        eprintln!("[cs-ss3t] wrote sidecar → {}", sidecar_path.display());
    }

    if args.diagnostics {
        if let Some(iters) = &result.iterations {
            let p = sibling_path(&args.output_wm, "_iters.nii.gz");
            write_3d_u32_as_f32(&p, &args.dwi, iters, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-ss3t] wrote iter map → {}", p.display());
            }
        }
        if let Some(resid) = &result.residual_l2 {
            let p = sibling_path(&args.output_wm, "_residual.nii.gz");
            write_3d_f32(&p, &args.dwi, resid, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-ss3t] wrote residual map → {}", p.display());
            }
        }
        if let Some(conv) = &result.converged {
            let p = sibling_path(&args.output_wm, "_converged.nii.gz");
            write_3d_u8(&p, &args.dwi, conv, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-ss3t] wrote converged map → {}", p.display());
            }
        }
        if let Some(lmax) = &result.chosen_lmax {
            let p = sibling_path(&args.output_wm, "_lmax_chosen.nii.gz");
            write_3d_u8(&p, &args.dwi, lmax, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-ss3t] wrote chosen lmax map → {}", p.display());
            }
        }
        if let Some(bic) = &result.bic {
            let p = sibling_path(&args.output_wm, "_bic.nii.gz");
            write_3d_f32(&p, &args.dwi, bic, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-ss3t] wrote BIC map → {}", p.display());
            }
        }
    }

    Ok(())
}

fn sidecar_for(wm_path: &Path) -> PathBuf {
    let s = wm_path.to_string_lossy();
    let stem = if let Some(stripped) = s.strip_suffix(".nii.gz") {
        stripped
    } else if let Some(stripped) = s.strip_suffix(".nii") {
        stripped
    } else {
        s.as_ref()
    };
    PathBuf::from(format!("{}.json", stem))
}

