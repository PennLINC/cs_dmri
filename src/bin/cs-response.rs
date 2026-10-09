// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-response`: WM/GM/CSF response-function estimation for SS3T.
//!
//! Implements the Dhollander 2016 unsupervised three-tissue response
//! algorithm (ISMRM Workshop abstract) using cs_dmri's RESTORE DTI fit
//! for the underlying scalar maps. Outputs three MRtrix-format `.txt`
//! files that drop directly into `cs-ss3t --response-{wm,gm,csf}`.
//!
//! License posture: the per-tissue averaging is cs_dmri's own, written from
//! the Dhollander 2016 abstract. The voxel selection it calls is a port of
//! MRtrix's `dwi2response dhollander` (MPL-2.0), confined to
//! `src/multitissue/dhollander.rs`.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use cs_dmri::dti::{DtiFitConfig, RestoreConfig, fit_volume_restore_reporting};
use cs_dmri::io::aux::{write_3d_u8};
use cs_dmri::io::dwi::load_dwi;
use cs_dmri::multitissue::response_estimation::{
    DhollanderConfig, DhollanderSelectConfig, estimate_responses, write_response_txt,
};
use cs_dmri::qspace::{BvecFrame, TORTOISE_DEFAULT_GMAX};
use cs_dmri::{
    Heartbeat, ProvenanceBuilder, ProvenanceMode, configure_rayon_threads, effective_thread_count,
};

#[derive(Parser, Debug)]
#[command(version, about = "Estimate white matter, grey matter and CSF response functions (Dhollander et al., 2016)")]
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
    /// Brain mask NIfTI. If omitted, a mask is computed by thresholding the
    /// mean b=0 image.
    #[arg(long)]
    mask: Option<PathBuf>,

    /// Output single-fibre white matter response (MRtrix `.txt` format).
    #[arg(long)]
    output_wm: PathBuf,
    /// Output grey matter response (MRtrix `.txt` format, one column).
    #[arg(long)]
    output_gm: PathBuf,
    /// Output CSF response (MRtrix `.txt` format, one column).
    #[arg(long)]
    output_csf: PathBuf,

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
    /// Maximum SH order of the white matter response.
    #[arg(long, default_value_t = 8)]
    lmax_wm: usize,

    /// Maximum number of RESTORE reweighting iterations in the tensor fit.
    #[arg(long, default_value_t = 50)]
    restore_max_iter: usize,
    /// RESTORE convergence tolerance on the relative change in tensor
    /// coefficients.
    #[arg(long, default_value_t = 1e-6)]
    restore_tol: f64,
    /// Geman-McClure weight below which a measurement is counted as an
    /// outlier in the tensor fit's outlier fraction. Does not affect the fit.
    #[arg(long, default_value_t = 0.04)]
    restore_outlier_threshold: f64,

    /// Diffusion time Δ (big delta), in seconds. Not used by response
    /// estimation; accepted for consistency with `cs-fit`.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Gradient pulse duration δ (small delta), in seconds. Not used by
    /// response estimation.
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude, in T/m. Used only when Δ and δ are
    /// estimated.
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,

    /// Also write the selected voxels of each tissue as mask NIfTIs
    /// (`<output_wm_stem>_mask_wm.nii.gz`, `_mask_gm.nii.gz`,
    /// `_mask_csf.nii.gz`).
    #[arg(long)]
    diagnostics: bool,

    /// Use b-vectors in the image-axis (FSL) frame. By default b-vectors are
    /// rotated into world (RAS) coordinates.
    #[arg(long)]
    no_bvec_rotation: bool,

    /// Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used,
    /// then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU.
    #[arg(long)]
    threads: Option<usize>,

    /// Overwrite existing `--diagnostics` mask NIfTIs. Without this flag,
    /// existing masks cause an error. Response files are always written.
    #[arg(long)]
    overwrite: bool,

    /// Suppress periodic progress and per-step summary messages.
    #[arg(long)]
    quiet: bool,

    /// Interval between progress messages during the voxel-wise tensor fit,
    /// in seconds.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance mode. Accepted for consistency with the other tools; `cs-
    /// response` writes no provenance record.
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let provenance_builder = ProvenanceBuilder::new("cs-response", args.provenance);
    let (threads, source) = configure_rayon_threads(args.threads)?;
    if !args.quiet {
        let n = if threads == 0 { effective_thread_count() } else { threads };
        eprintln!("[cs-response] threads={} source={}", n, source.as_str());
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
            "[cs-response] DWI shape {:?}, mask voxels: {}",
            dwi.shape(),
            mask_voxels
        );
    }
    cs_dmri::qc::report_input_qc("cs-response", &dwi, args.quiet);

    // Step 1: RESTORE DTI fit.
    if !args.quiet {
        eprintln!("[cs-response] step 1: RESTORE DTI fit...");
    }
    let restore_cfg = RestoreConfig {
        max_iter: args.restore_max_iter,
        tol: args.restore_tol,
        outlier_threshold: args.restore_outlier_threshold,
        ..RestoreConfig::default()
    };
    let interval = Duration::from_secs(args.progress_interval_secs.max(1));
    let heartbeat = Heartbeat::new("cs-response", mask_voxels, interval, args.quiet);
    let dti = fit_volume_restore_reporting(
        &dwi,
        &restore_cfg,
        DtiFitConfig { compute_diagnostics: false },
        || heartbeat.tick(),
    )
    .with_context(|| "DTI fit failed")?;
    heartbeat.finish();

    // Step 2: Dhollander tissue selection + per-tissue averaging.
    if !args.quiet {
        eprintln!("[cs-response] step 2: tissue selection + per-shell averaging...");
    }
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

    // Step 3: write three .txt files.
    write_response_txt(&estimate.wm, &args.output_wm)
        .with_context(|| format!("write WM response {:?}", args.output_wm))?;
    write_response_txt(&estimate.gm, &args.output_gm)
        .with_context(|| format!("write GM response {:?}", args.output_gm))?;
    write_response_txt(&estimate.csf, &args.output_csf)
        .with_context(|| format!("write CSF response {:?}", args.output_csf))?;

    let _provenance = provenance_builder.map(|b| b.finish(effective_thread_count()));

    if !args.quiet {
        let d = &estimate.diagnostics;
        eprintln!(
            "[cs-response] tissue voxels: WM={} ({:.1}%), GM={} ({:.1}%), CSF={} ({:.1}%) of {} brain",
            d.n_wm_voxels,
            100.0 * d.n_wm_voxels as f64 / d.n_brain_voxels as f64,
            d.n_gm_voxels,
            100.0 * d.n_gm_voxels as f64 / d.n_brain_voxels as f64,
            d.n_csf_voxels,
            100.0 * d.n_csf_voxels as f64 / d.n_brain_voxels as f64,
            d.n_brain_voxels
        );
        eprintln!(
            "[cs-response] MD CSF threshold = {:.4e}",
            d.md_csf_threshold
        );
        eprintln!(
            "[cs-response] wrote WM response → {}",
            args.output_wm.display()
        );
        eprintln!(
            "[cs-response] wrote GM response → {}",
            args.output_gm.display()
        );
        eprintln!(
            "[cs-response] wrote CSF response → {}",
            args.output_csf.display()
        );
    }

    if args.diagnostics {
        let wm_mask_path = sibling_mask(&args.output_wm, "_mask_wm.nii.gz");
        let gm_mask_path = sibling_mask(&args.output_wm, "_mask_gm.nii.gz");
        let csf_mask_path = sibling_mask(&args.output_wm, "_mask_csf.nii.gz");
        write_3d_u8(&wm_mask_path, &args.dwi, &estimate.wm_mask, args.overwrite)
            .with_context(|| format!("write WM mask {:?}", wm_mask_path))?;
        write_3d_u8(&gm_mask_path, &args.dwi, &estimate.gm_mask, args.overwrite)
            .with_context(|| format!("write GM mask {:?}", gm_mask_path))?;
        write_3d_u8(
            &csf_mask_path,
            &args.dwi,
            &estimate.csf_mask,
            args.overwrite,
        )
        .with_context(|| format!("write CSF mask {:?}", csf_mask_path))?;
        if !args.quiet {
            eprintln!("[cs-response] wrote WM mask → {}", wm_mask_path.display());
            eprintln!("[cs-response] wrote GM mask → {}", gm_mask_path.display());
            eprintln!(
                "[cs-response] wrote CSF mask → {}",
                csf_mask_path.display()
            );
        }
    }

    Ok(())
}

/// Build a sibling-mask path next to a `.txt` response file.
fn sibling_mask(response_txt: &std::path::Path, suffix: &str) -> PathBuf {
    let s = response_txt.to_string_lossy();
    let stem = if let Some(stripped) = s.strip_suffix(".txt") {
        stripped
    } else {
        s.as_ref()
    };
    PathBuf::from(format!("{}{}", stem, suffix))
}
