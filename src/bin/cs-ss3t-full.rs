// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-ss3t-full`: end-to-end SS3T pipeline in one command.
//!
//! Chains [`cs-response`](../cs-response.rs), [`cs-ss3t`](../cs-ss3t.rs), and
//! [`cs-mtnorm`](../cs-mtnorm.rs) so a single invocation goes from raw
//! single-shell DWI to normalised WM/GM/CSF tissue maps without any
//! external MRtrix tooling.
//!
//! Stages can be skipped:
//! - `--response-{wm,gm,csf} PATH` → skip response estimation, use the
//!   supplied `.txt` files.
//! - `--no-normalize` → skip the mtnormalise stage.
//!
//! All intermediate buffers stay in memory; nothing is written to disk
//! until the final outputs.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use cs_dmri::dti::{DtiFitConfig, RestoreConfig, fit_volume_restore_reporting};
use cs_dmri::io::aux::{sibling_path, write_3d_f32, write_3d_u32_as_f32, write_3d_u8, write_4d_f32};
use cs_dmri::io::dwi::load_dwi;
use cs_dmri::multitissue::mtnormalise::{
    MtnormaliseConfig, mtnormalise, target_sum_mrtrix_default,
};
use cs_dmri::multitissue::response_estimation::{
    DhollanderConfig, DhollanderSelectConfig, estimate_responses, write_response_txt,
};
use cs_dmri::multitissue::ss3t::{Ss3tConfig, Ss3tResponses, LmaxWmStrategy};
use cs_dmri::multitissue::TissueResponse;
use cs_dmri::multitissue::volume::{Ss3tFitConfig, fit_volume_ss3t_reporting};
use cs_dmri::qspace::{BvecFrame, TORTOISE_DEFAULT_GMAX};
use cs_dmri::solver::icls::IclsConfig;
use cs_dmri::{
    Heartbeat, ProvenanceBuilder, ProvenanceMode, configure_rayon_threads, effective_thread_count,
};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "End-to-end SS3T pipeline (response estimation → SS3T → mtnormalise)"
)]
struct Cli {
    /// 4D DWI NIfTI input (one b=0 + one DWI shell).
    #[arg(long)]
    dwi: PathBuf,
    /// FSL bval file.
    #[arg(long)]
    bval: PathBuf,
    /// FSL bvec file.
    #[arg(long)]
    bvec: PathBuf,
    /// Optional brain mask NIfTI.
    #[arg(long)]
    mask: Option<PathBuf>,

    /// Output WM FOD NIfTI (after normalisation, unless `--no-normalize`).
    /// Required unless `--odx` is given.
    #[arg(long)]
    output_wm: Option<PathBuf>,
    /// Output GM compartment NIfTI. Required unless `--odx` is given.
    #[arg(long)]
    output_gm: Option<PathBuf>,
    /// Output CSF compartment NIfTI. Required unless `--odx` is given.
    #[arg(long)]
    output_csf: Option<PathBuf>,

    /// Write a single ODX file bundling WM SH (canonical), GM/CSF (in `sh/`),
    /// brain mask, response functions (in header), and WM peaks. When set,
    /// the three NIfTI outputs are skipped. Use a `.odx` extension for an
    /// archive; any other extension/path becomes a directory tree.
    #[arg(long, value_name = "PATH")]
    odx: Option<PathBuf>,

    /// Skip response estimation: use this WM `.txt` file instead.
    /// Provide all three of `--response-{wm,gm,csf}` to skip estimation.
    #[arg(long, requires = "response_gm", requires = "response_csf")]
    response_wm: Option<PathBuf>,
    /// Skip response estimation: use this GM `.txt` file.
    #[arg(long)]
    response_gm: Option<PathBuf>,
    /// Skip response estimation: use this CSF `.txt` file.
    #[arg(long)]
    response_csf: Option<PathBuf>,

    /// Optional: write the estimated responses to disk for inspection or
    /// reuse. Only honored when responses are estimated (no
    /// `--response-*` overrides).
    #[arg(long)]
    write_responses_to: Option<PathBuf>,

    /// Skip the final mtnormalise step. Useful when downstream tools
    /// expect raw SS3T outputs or have their own normalisation.
    #[arg(long)]
    no_normalize: bool,

    // ---- Response estimation knobs (Dhollander) ----
    /// Erosion passes applied to the brain mask before tissue selection.
    #[arg(long, default_value_t = 3)]
    dh_erode: usize,
    /// FA threshold for the crude WM vs GM-CSF split.
    #[arg(long, default_value_t = 0.2)]
    dh_fa: f64,
    /// Final single-fibre WM voxels, as a percentage of refined WM.
    #[arg(long, default_value_t = 0.5)]
    dh_sfwm: f64,
    /// Final GM voxels, as a percentage of refined GM.
    #[arg(long, default_value_t = 2.0)]
    dh_gm: f64,
    /// Final CSF voxels, as a percentage of refined CSF.
    #[arg(long, default_value_t = 10.0)]
    dh_csf: f64,
    /// Use the pre-2026 threshold-triple tissue selection (top-N%-MD CSF,
    /// FA+dominance WM) instead of MRtrix's staged signal-decay-metric
    /// algorithm. Only for reproducing older runs: its CSF class includes
    /// partial-volume voxels, which depresses the CSF response amplitude.
    #[arg(long)]
    legacy_tissue_selection: bool,

    /// FA above this counts a voxel as a WM single-fibre candidate.
    /// `--legacy-tissue-selection` only.
    #[arg(long, default_value_t = 0.7)]
    fa_wm_threshold: f64,
    /// Eigenvalue ratio gate for "single fibre" (suppresses crossings).
    #[arg(long, default_value_t = 2.0)]
    fiber_dominance_ratio: f64,
    /// Top-N percent of MD voxels classified as CSF.
    #[arg(long, default_value_t = 2.5)]
    md_csf_pct: f64,

    // ---- SS3T knobs ----
    /// SS3T outer iterations.
    #[arg(long, default_value_t = 3)]
    niter: u32,
    /// b=0 contribution percentage.
    #[arg(long, default_value_t = 10.0)]
    bzero_pct: f64,
    /// Maximum WM SH order.
    #[arg(long, default_value_t = 8)]
    lmax_wm: usize,

    // ---- mtnormalise knobs ----
    /// Polynomial order for the bias-field fit (default 3 = 20 monomials).
    #[arg(long, default_value_t = 3)]
    mtnorm_poly_order: usize,
    /// Use the median observed l=0 sum as mtnormalise's target instead of
    /// the MRtrix default (`1/sqrt(4π) ≈ 0.282`). Preserves input scale.
    #[arg(long)]
    mtnorm_target_median: bool,
    /// Apply the per-tissue balance factors to the output, like MRtrix3
    /// `mtnormalise -balanced`. Off by default, matching MRtrix.
    #[arg(long)]
    mtnorm_balanced: bool,

    // ---- DTI / RESTORE knobs ----
    /// RESTORE max iterations for the underlying DTI fit (used only when
    /// estimating responses).
    #[arg(long, default_value_t = 50)]
    restore_max_iter: usize,
    /// RESTORE convergence tolerance.
    #[arg(long, default_value_t = 1e-6)]
    restore_tol: f64,

    // ---- Standard ----
    /// Big delta Δ (seconds), recorded for provenance only.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Small delta δ (seconds).
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude (T/m).
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,

    /// Also write `_iters`, `_residual`, `_converged` sibling NIfTIs.
    #[arg(long)]
    diagnostics: bool,

    /// Keep bvecs in image-axis frame.
    #[arg(long)]
    no_bvec_rotation: bool,

    /// Cap rayon's worker threads.
    #[arg(long)]
    threads: Option<usize>,

    /// Allow overwriting existing outputs.
    #[arg(long)]
    overwrite: bool,

    /// Suppress per-step summary lines.
    #[arg(long)]
    quiet: bool,

    /// Seconds between heartbeat lines during long parallel loops.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance mode (currently informational only).
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    if args.odx.is_none()
        && (args.output_wm.is_none() || args.output_gm.is_none() || args.output_csf.is_none())
    {
        anyhow::bail!(
            "must pass either --odx <path> or all three of --output-wm/--output-gm/--output-csf"
        );
    }
    let provenance_builder = ProvenanceBuilder::new("cs-ss3t-full", args.provenance);
    let (threads, source) = configure_rayon_threads(args.threads)?;
    if !args.quiet {
        let n = if threads == 0 { effective_thread_count() } else { threads };
        eprintln!("[cs-ss3t-full] threads={} source={}", n, source.as_str());
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
            "[cs-ss3t-full] DWI shape {:?}, mask voxels: {}",
            dwi.shape(),
            mask_voxels
        );
    }

    // ---- Stage 1: responses (estimate or load) ----
    let interval = Duration::from_secs(args.progress_interval_secs.max(1));
    let responses = if args.response_wm.is_some() {
        let wm_path = args.response_wm.as_ref().unwrap();
        let gm_path = args.response_gm.as_ref().unwrap();
        let csf_path = args.response_csf.as_ref().unwrap();
        if !args.quiet {
            eprintln!(
                "[cs-ss3t-full] stage 1/3: loading provided responses (skipping estimation)"
            );
        }
        Ss3tResponses {
            wm: TissueResponse::parse_mrtrix_txt(wm_path)
                .with_context(|| format!("parse {:?}", wm_path))?,
            gm: TissueResponse::parse_mrtrix_txt(gm_path)
                .with_context(|| format!("parse {:?}", gm_path))?,
            csf: TissueResponse::parse_mrtrix_txt(csf_path)
                .with_context(|| format!("parse {:?}", csf_path))?,
        }
    } else {
        if !args.quiet {
            eprintln!(
                "[cs-ss3t-full] stage 1/3: estimating responses via Dhollander 2016 (RESTORE DTI + tissue selection)"
            );
        }
        let restore_cfg = RestoreConfig {
            max_iter: args.restore_max_iter,
            tol: args.restore_tol,
            ..RestoreConfig::default()
        };
        let dti_hb = Heartbeat::new("dti", mask_voxels, interval, args.quiet);
        let dti = fit_volume_restore_reporting(
            &dwi,
            &restore_cfg,
            DtiFitConfig { compute_diagnostics: false },
            || dti_hb.tick(),
        )
        .with_context(|| "DTI fit failed")?;
        dti_hb.finish();
        let cfg = DhollanderConfig {
            fa_wm_threshold: args.fa_wm_threshold,
            fiber_dominance_ratio: args.fiber_dominance_ratio,
            md_csf_pct: args.md_csf_pct,
            lmax_wm: args.lmax_wm,
            legacy_selection: args.legacy_tissue_selection,
            stages: DhollanderSelectConfig {
                erode: args.dh_erode,
                fa: args.dh_fa,
                sfwm_pct: args.dh_sfwm,
                gm_pct: args.dh_gm,
                csf_pct: args.dh_csf,
                ..DhollanderSelectConfig::default()
            },
        };
        let estimate =
            estimate_responses(&dwi, &dti, &cfg).with_context(|| "response estimation failed")?;
        if !args.quiet {
            let d = &estimate.diagnostics;
            eprintln!(
                "[cs-ss3t-full]   tissue voxels: WM={} ({:.1}%)  GM={} ({:.1}%)  CSF={} ({:.1}%)",
                d.n_wm_voxels,
                100.0 * d.n_wm_voxels as f64 / d.n_brain_voxels as f64,
                d.n_gm_voxels,
                100.0 * d.n_gm_voxels as f64 / d.n_brain_voxels as f64,
                d.n_csf_voxels,
                100.0 * d.n_csf_voxels as f64 / d.n_brain_voxels as f64,
            );
        }
        if let Some(dir) = &args.write_responses_to {
            std::fs::create_dir_all(dir).ok();
            let wm_p = dir.join("wm_response.txt");
            let gm_p = dir.join("gm_response.txt");
            let csf_p = dir.join("csf_response.txt");
            write_response_txt(&estimate.wm, &wm_p)
                .with_context(|| format!("write {:?}", wm_p))?;
            write_response_txt(&estimate.gm, &gm_p)
                .with_context(|| format!("write {:?}", gm_p))?;
            write_response_txt(&estimate.csf, &csf_p)
                .with_context(|| format!("write {:?}", csf_p))?;
            if !args.quiet {
                eprintln!("[cs-ss3t-full]   wrote responses → {}", dir.display());
            }
        }
        Ss3tResponses {
            wm: estimate.wm,
            gm: estimate.gm,
            csf: estimate.csf,
        }
    };

    // ---- Stage 2: SS3T ----
    if !args.quiet {
        eprintln!("[cs-ss3t-full] stage 2/3: SS3T iterative tissue decomposition");
    }
    let cfg = Ss3tConfig {
        niter: args.niter,
        bzero_pct: args.bzero_pct,
        lmax_wm: LmaxWmStrategy::Fixed(args.lmax_wm),
        icls: IclsConfig::default(),
    };
    let ss3t_hb = Heartbeat::new("ss3t", mask_voxels, interval, args.quiet);
    let mut result = fit_volume_ss3t_reporting(
        &dwi,
        &responses,
        &cfg,
        Ss3tFitConfig { compute_diagnostics: args.diagnostics },
        || ss3t_hb.tick(),
    )
    .with_context(|| "ss3t fit failed")?;
    ss3t_hb.finish();

    // ---- Stage 3: mtnormalise (optional) ----
    if !args.no_normalize {
        if !args.quiet {
            eprintln!("[cs-ss3t-full] stage 3/3: mtnormalise (polynomial bias-field correction)");
        }
        let mtcfg = MtnormaliseConfig {
            poly_order: args.mtnorm_poly_order,
            target_sum: if args.mtnorm_target_median {
                None
            } else {
                Some(target_sum_mrtrix_default())
            },
            apply_balance: args.mtnorm_balanced,
            ..MtnormaliseConfig::default()
        };
        let diag = mtnormalise(&mut result.wm, &mut result.gm, &mut result.csf, &dwi.mask, &mtcfg)
            .with_context(|| "mtnormalise failed")?;
        if !args.quiet {
            eprintln!(
                "[cs-ss3t-full]   target_sum={:.4}  fit voxels={}  mean |log residual|={:.3e}",
                diag.target_sum_used, diag.n_fit_voxels, diag.mean_abs_log_residual
            );
        }
    } else if !args.quiet {
        eprintln!("[cs-ss3t-full] stage 3/3: skipped (--no-normalize)");
    }

    // ---- Write outputs ----
    if let Some(odx_path) = &args.odx {
        let directory = odx_path.extension().and_then(|s| s.to_str()) != Some("odx");
        cs_dmri::io::odx_out::write_ss3t_odx(
            odx_path,
            &args.dwi,
            &dwi.mask,
            &result.wm,
            &result.gm,
            &result.csf,
            args.lmax_wm,
            &responses,
            args.overwrite,
            directory,
        )
        .with_context(|| format!("write ODX {:?}", odx_path))?;
        if !args.quiet {
            eprintln!("[cs-ss3t-full] wrote ODX → {}", odx_path.display());
        }
    } else {
        let wm_path = args.output_wm.as_ref().unwrap();
        let gm_path = args.output_gm.as_ref().unwrap();
        let csf_path = args.output_csf.as_ref().unwrap();
        write_4d_f32(wm_path, &args.dwi, &result.wm, args.overwrite)
            .with_context(|| format!("write WM {:?}", wm_path))?;
        write_4d_f32(gm_path, &args.dwi, &result.gm, args.overwrite)
            .with_context(|| format!("write GM {:?}", gm_path))?;
        write_4d_f32(csf_path, &args.dwi, &result.csf, args.overwrite)
            .with_context(|| format!("write CSF {:?}", csf_path))?;
        if !args.quiet {
            eprintln!("[cs-ss3t-full] wrote WM → {}", wm_path.display());
            eprintln!("[cs-ss3t-full] wrote GM → {}", gm_path.display());
            eprintln!("[cs-ss3t-full] wrote CSF → {}", csf_path.display());
        }

        if args.diagnostics {
            if let Some(iters) = &result.iterations {
                let p = sibling_path(wm_path, "_iters.nii.gz");
                write_3d_u32_as_f32(&p, &args.dwi, iters, args.overwrite)?;
            }
            if let Some(resid) = &result.residual_l2 {
                let p = sibling_path(wm_path, "_residual.nii.gz");
                write_3d_f32(&p, &args.dwi, resid, args.overwrite)?;
            }
            if let Some(conv) = &result.converged {
                let p = sibling_path(wm_path, "_converged.nii.gz");
                write_3d_u8(&p, &args.dwi, conv, args.overwrite)?;
            }
        }
    }

    let _provenance = provenance_builder.map(|b| b.finish(effective_thread_count()));

    Ok(())
}
