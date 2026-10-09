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
use std::sync::RwLock;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use cs_dmri::dti::RestoreConfig;
use cs_dmri::io::aux::{sibling_path, write_3d_f32, write_3d_u32_as_f32, write_3d_u8, write_4d_f32};
use cs_dmri::io::dwi::load_dwi;
use cs_dmri::multitissue::mtnormalise::{MtnormaliseConfig, target_sum_mrtrix_default};
use cs_dmri::multitissue::pipeline::{
    PipelineObserver, PipelineStage, ResponseSource, Ss3tPipelineConfig, ss3t_pipeline,
};
use cs_dmri::multitissue::response_estimation::{
    DhollanderConfig, DhollanderSelectConfig, write_response_txt,
};
use cs_dmri::multitissue::ss3t::{Ss3tConfig, Ss3tResponses, LmaxWmStrategy};
use cs_dmri::multitissue::TissueResponse;
use cs_dmri::qspace::{BvecFrame, TORTOISE_DEFAULT_GMAX};
use cs_dmri::solver::icls::IclsConfig;
use cs_dmri::{
    Heartbeat, ProvenanceBuilder, ProvenanceMode, configure_rayon_threads, effective_thread_count,
};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Single-shell three-tissue pipeline: response function estimation, SS3T-CSD and multi-tissue intensity normalisation"
)]
struct Cli {
    /// 4D DWI NIfTI input (b=0 volumes and one diffusion-weighted shell).
    #[arg(long)]
    dwi: PathBuf,
    /// FSL bval file.
    #[arg(long)]
    bval: PathBuf,
    /// FSL bvec file.
    #[arg(long)]
    bvec: PathBuf,
    /// Brain mask NIfTI. If omitted, a mask is computed from the mean b=0
    /// image.
    #[arg(long)]
    mask: Option<PathBuf>,

    /// Output white matter FOD NIfTI, normalised unless `--no-normalize` is
    /// given. Required unless `--odx` is given.
    #[arg(long)]
    output_wm: Option<PathBuf>,
    /// Output grey matter compartment NIfTI. Required unless `--odx` is
    /// given.
    #[arg(long)]
    output_gm: Option<PathBuf>,
    /// Output CSF compartment NIfTI. Required unless `--odx` is given.
    #[arg(long)]
    output_csf: Option<PathBuf>,

    /// Write a single ODX file containing the white matter SH coefficients,
    /// the grey matter and CSF compartments (under `sh/`), the brain mask,
    /// the response functions (in the header) and white matter peaks. The
    /// NIfTI outputs are then not written. A path with a `.odx` extension is
    /// written as a zip archive; any other path is written as a directory.
    #[arg(long, value_name = "PATH")]
    odx: Option<PathBuf>,

    /// White matter response in MRtrix `.txt` format. If all three of
    /// `--response-{wm,gm,csf}` are given, response estimation is skipped.
    #[arg(long, requires = "response_gm", requires = "response_csf")]
    response_wm: Option<PathBuf>,
    /// Grey matter response in MRtrix `.txt` format (see --response-wm).
    #[arg(long)]
    response_gm: Option<PathBuf>,
    /// CSF response in MRtrix `.txt` format (see --response-wm).
    #[arg(long)]
    response_csf: Option<PathBuf>,

    /// Directory in which to write the estimated responses
    /// (`wm_response.txt`, `gm_response.txt`, `csf_response.txt`). Ignored
    /// when responses are supplied with `--response-*`.
    #[arg(long)]
    write_responses_to: Option<PathBuf>,

    /// Do not apply multi-tissue intensity normalisation; the SS3T-CSD
    /// outputs are written as fitted.
    #[arg(long)]
    no_normalize: bool,

    // ---- Response estimation knobs (Dhollander) ----
    /// Number of erosion passes applied to the brain mask before tissue
    /// selection. Not used with --legacy-tissue-selection.
    #[arg(long, default_value_t = 3)]
    dh_erode: usize,
    /// FA threshold for the initial separation of white matter from grey
    /// matter and CSF. Not used with --legacy-tissue-selection.
    #[arg(long, default_value_t = 0.2)]
    dh_fa: f64,
    /// Number of single-fibre white matter voxels selected, as a percentage of
    /// the refined white matter. Not used with --legacy-tissue-selection.
    #[arg(long, default_value_t = 0.5)]
    dh_sfwm: f64,
    /// Number of grey matter voxels selected, as a percentage of the refined
    /// grey matter. Not used with --legacy-tissue-selection.
    #[arg(long, default_value_t = 2.0)]
    dh_gm: f64,
    /// Number of CSF voxels selected, as a percentage of the refined CSF. Not
    /// used with --legacy-tissue-selection.
    #[arg(long, default_value_t = 10.0)]
    dh_csf: f64,
    /// Use the earlier threshold-based tissue selection instead of the staged
    /// selection based on a signal decay metric. CSF voxels are those in the
    /// top --md-csf-pct percent of MD; single-fibre white matter voxels have
    /// FA above --fa-wm-threshold and eigenvalue ratio above
    /// --fiber-dominance-ratio; the remaining voxels are grey matter. The CSF
    /// class selected in this way can include partial-volume voxels.
    #[arg(long)]
    legacy_tissue_selection: bool,

    /// FA above which a voxel is a single-fibre white matter candidate. Used
    /// only with --legacy-tissue-selection.
    #[arg(long, default_value_t = 0.7)]
    fa_wm_threshold: f64,
    /// Minimum eigenvalue ratio λ₁ / mean(λ₂, λ₃) for a single-fibre white
    /// matter voxel; 0 disables the test. Used only with
    /// --legacy-tissue-selection.
    #[arg(long, default_value_t = 2.0)]
    fiber_dominance_ratio: f64,
    /// Percentage of brain voxels with the highest MD that are selected as
    /// CSF. Used only with --legacy-tissue-selection.
    #[arg(long, default_value_t = 2.5)]
    md_csf_pct: f64,

    // ---- SS3T knobs ----
    /// Number of SS3T outer iterations. Must be at least 2.
    #[arg(long, default_value_t = 3)]
    niter: u32,
    /// Weight of the b=0 volumes in the SS3T fit, as a percentage of the
    /// diffusion-weighted volumes. Must be positive.
    #[arg(long, default_value_t = 10.0)]
    bzero_pct: f64,
    /// Maximum SH order of the white matter FOD and response.
    #[arg(long, default_value_t = 8)]
    lmax_wm: usize,

    // ---- mtnormalise knobs ----
    /// Order of the polynomial bias field model in the normalisation step
    /// (order 3 has 20 terms).
    #[arg(long, default_value_t = 3)]
    mtnorm_poly_order: usize,
    /// In the normalisation step, use the median of the observed sums of l=0
    /// coefficients as the target instead of 1/√(4π). The global scale of
    /// the input is preserved.
    #[arg(long)]
    mtnorm_target_median: bool,
    /// Multiply each output tissue by its balance factor, as in MRtrix3
    /// `mtnormalise -balanced`.
    #[arg(long)]
    mtnorm_balanced: bool,

    // ---- DTI / RESTORE knobs ----
    /// Maximum number of RESTORE reweighting iterations in the tensor fit
    /// used for response estimation.
    #[arg(long, default_value_t = 50)]
    restore_max_iter: usize,
    /// RESTORE convergence tolerance on the relative change in tensor
    /// coefficients.
    #[arg(long, default_value_t = 1e-6)]
    restore_tol: f64,

    // ---- Standard ----
    /// Diffusion time Δ (big delta), in seconds. Not used by the pipeline;
    /// accepted for consistency with `cs-fit`.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Gradient pulse duration δ (small delta), in seconds. Not used by the
    /// pipeline.
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude, in T/m. Used only when Δ and δ are
    /// estimated.
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,

    /// Also write per-voxel maps of iteration count (`_iters.nii.gz`),
    /// residual (`_residual.nii.gz`) and convergence (`_converged.nii.gz`)
    /// next to `--output-wm`. Not written with `--odx`.
    #[arg(long)]
    diagnostics: bool,

    /// Fit with b-vectors in the image-axis (FSL) frame. By default b-vectors
    /// are rotated into world (RAS) coordinates before fitting.
    #[arg(long)]
    no_bvec_rotation: bool,

    /// Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used,
    /// then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU.
    #[arg(long)]
    threads: Option<usize>,

    /// Overwrite existing output files. Without this flag, existing outputs
    /// cause an error.
    #[arg(long)]
    overwrite: bool,

    /// Suppress periodic progress and per-step summary messages.
    #[arg(long)]
    quiet: bool,

    /// Interval between progress messages, in seconds.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance mode. Accepted for consistency with the other tools; `cs-
    /// ss3t-full` writes no provenance record.
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
    cs_dmri::qc::report_input_qc("cs-ss3t-full", &dwi, args.quiet);

    // ---- Stages 1-3: responses → SS3T → mtnormalise ----
    let interval = Duration::from_secs(args.progress_interval_secs.max(1));
    let response_source = if args.response_wm.is_some() {
        let wm_path = args.response_wm.as_ref().unwrap();
        let gm_path = args.response_gm.as_ref().unwrap();
        let csf_path = args.response_csf.as_ref().unwrap();
        ResponseSource::Provided(Ss3tResponses {
            wm: TissueResponse::parse_mrtrix_txt(wm_path)
                .with_context(|| format!("parse {:?}", wm_path))?,
            gm: TissueResponse::parse_mrtrix_txt(gm_path)
                .with_context(|| format!("parse {:?}", gm_path))?,
            csf: TissueResponse::parse_mrtrix_txt(csf_path)
                .with_context(|| format!("parse {:?}", csf_path))?,
        })
    } else {
        ResponseSource::Estimate {
            restore: RestoreConfig {
                max_iter: args.restore_max_iter,
                tol: args.restore_tol,
                ..RestoreConfig::default()
            },
            dhollander: DhollanderConfig {
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
            },
        }
    };
    let estimating = matches!(response_source, ResponseSource::Estimate { .. });
    let pipeline_cfg = Ss3tPipelineConfig {
        responses: response_source,
        ss3t: Ss3tConfig {
            niter: args.niter,
            bzero_pct: args.bzero_pct,
            lmax_wm: LmaxWmStrategy::Fixed(args.lmax_wm),
            icls: IclsConfig::default(),
        },
        compute_diagnostics: args.diagnostics,
        normalize: (!args.no_normalize).then(|| MtnormaliseConfig {
            poly_order: args.mtnorm_poly_order,
            target_sum: if args.mtnorm_target_median {
                None
            } else {
                Some(target_sum_mrtrix_default())
            },
            apply_balance: args.mtnorm_balanced,
            ..MtnormaliseConfig::default()
        }),
    };
    if !args.quiet && !estimating {
        eprintln!("[cs-ss3t-full] stage 1/3: loading provided responses (skipping estimation)");
    }
    let observer = StageProgress {
        current: RwLock::new(None),
        mask_voxels,
        interval,
        quiet: args.quiet,
    };
    let out = ss3t_pipeline(&dwi, &pipeline_cfg, &observer).with_context(|| "SS3T pipeline failed")?;
    observer.finish_current();
    for w in &out.warnings {
        eprintln!("[ss3t] {w}");
    }

    if let Some((estimate, _dti)) = &out.estimate {
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
            for (tissue, name) in [
                (&estimate.wm, "wm_response.txt"),
                (&estimate.gm, "gm_response.txt"),
                (&estimate.csf, "csf_response.txt"),
            ] {
                let p = dir.join(name);
                write_response_txt(tissue, &p).with_context(|| format!("write {:?}", p))?;
            }
            if !args.quiet {
                eprintln!("[cs-ss3t-full]   wrote responses → {}", dir.display());
            }
        }
    }
    match &out.normalization {
        Some(diag) if !args.quiet => eprintln!(
            "[cs-ss3t-full]   target_sum={:.4}  fit voxels={}  mean |log residual|={:.3e}",
            diag.target_sum_used, diag.n_fit_voxels, diag.mean_abs_log_residual
        ),
        None if !args.quiet => eprintln!("[cs-ss3t-full] stage 3/3: skipped (--no-normalize)"),
        _ => {}
    }
    let responses = out.responses;
    let result = out.fit;

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
            // The effective lmax, not `--lmax-wm`: SS3T clamps to the WM
            // response's lmax, and the ODX SH order must match the WM channels.
            result.plan.plans[0].lmax_wm,
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

/// Prints the stage banners and runs one heartbeat per per-voxel stage.
struct StageProgress {
    current: RwLock<Option<Heartbeat>>,
    mask_voxels: usize,
    interval: Duration,
    quiet: bool,
}

impl StageProgress {
    fn finish_current(&self) {
        if let Some(hb) = self.current.write().unwrap().take() {
            hb.finish();
        }
    }
}

impl PipelineObserver for StageProgress {
    fn stage(&self, stage: PipelineStage) {
        self.finish_current();
        let (banner, label) = match stage {
            PipelineStage::Dti => (
                Some("stage 1/3: estimating responses via Dhollander 2016 (RESTORE DTI + tissue selection)"),
                Some("dti"),
            ),
            PipelineStage::ResponseEstimation => (None, None),
            PipelineStage::Ss3t => (Some("stage 2/3: SS3T iterative tissue decomposition"), Some("ss3t")),
            PipelineStage::Normalize => {
                (Some("stage 3/3: mtnormalise (polynomial bias-field correction)"), None)
            }
        };
        if let (Some(b), false) = (banner, self.quiet) {
            eprintln!("[cs-ss3t-full] {b}");
        }
        if let Some(label) = label {
            *self.current.write().unwrap() =
                Some(Heartbeat::new(label, self.mask_voxels, self.interval, self.quiet));
        }
    }

    fn voxel(&self) {
        if let Some(hb) = self.current.read().unwrap().as_ref() {
            hb.tick();
        }
    }
}
