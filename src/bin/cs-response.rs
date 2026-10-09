// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-response`: WM/GM/CSF response-function estimation for SS3T.
//!
//! Implements the Dhollander 2016 unsupervised three-tissue response
//! algorithm (ISMRM Workshop abstract) using cs-dmri's RESTORE DTI fit
//! for the underlying scalar maps. Outputs three MRtrix-format `.txt`
//! files that drop directly into `cs-ss3t --response-{wm,gm,csf}`.
//!
//! License posture: the per-tissue averaging is cs-dmri's own, written from
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
#[command(version, about = "Dhollander-2016 three-tissue response estimation (cs-dmri)")]
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
    /// Optional brain mask NIfTI. Auto-thresholded from b0 mean if absent.
    #[arg(long)]
    mask: Option<PathBuf>,

    /// Output WM single-fibre response (MRtrix `.txt` format).
    #[arg(long)]
    output_wm: PathBuf,
    /// Output GM response (MRtrix `.txt` format, 1 column).
    #[arg(long)]
    output_gm: PathBuf,
    /// Output CSF response (MRtrix `.txt` format, 1 column).
    #[arg(long)]
    output_csf: PathBuf,

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
    /// Eigenvalue ratio λ₁ / mean(λ₂, λ₃): above this counts as
    /// "single-fibre" (suppresses crossing voxels). Default 2.0; set
    /// to 0 to skip.
    #[arg(long, default_value_t = 2.0)]
    fiber_dominance_ratio: f64,
    /// Top-N percent of MD values are CSF candidates (default 2.5%).
    #[arg(long, default_value_t = 2.5)]
    md_csf_pct: f64,
    /// Maximum SH order for the WM response.
    #[arg(long, default_value_t = 8)]
    lmax_wm: usize,

    /// RESTORE max reweighting iterations for the underlying DTI fit.
    #[arg(long, default_value_t = 50)]
    restore_max_iter: usize,
    /// RESTORE convergence tolerance.
    #[arg(long, default_value_t = 1e-6)]
    restore_tol: f64,
    /// Geman-McClure weight cutoff for the DTI outlier-fraction map (only
    /// affects diagnostics).
    #[arg(long, default_value_t = 0.04)]
    restore_outlier_threshold: f64,

    /// Big delta Δ (seconds). Recorded for sidecar parity; not used.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Small delta δ (seconds).
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude (T/m); only used when deltas are estimated.
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,

    /// Also write per-tissue selection masks as sibling NIfTIs
    /// (`<output_wm_stem>_mask_wm.nii.gz` etc.) for QC.
    #[arg(long)]
    diagnostics: bool,

    /// Keep bvecs in image-axis frame instead of rotating to world-RAS.
    #[arg(long)]
    no_bvec_rotation: bool,

    /// Cap rayon's worker threads.
    #[arg(long)]
    threads: Option<usize>,

    /// Allow overwriting existing outputs.
    #[arg(long)]
    overwrite: bool,

    /// Suppress progress heartbeat and per-step summary lines.
    #[arg(long)]
    quiet: bool,

    /// Seconds between heartbeat lines during the per-voxel DTI fit.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance mode (currently informational only — `cs-response` doesn't
    /// emit a sidecar JSON; provenance is encoded in the `.txt` header
    /// comment line).
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
