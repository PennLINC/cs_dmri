// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-dti`: robust diffusion-tensor fit via RESTORE.
//!
//! Loads a 4D DWI bundle and fits the diffusion tensor at every masked voxel
//! using the RESTORE algorithm (Chang, Jones & Pierpaoli, MRM 2005). Designed
//! for clinical-quality DWI where a subset of volumes per voxel suffer
//! motion-induced signal dropouts. Outputs FA, MD, S₀, and an outlier-fraction
//! NIfTI; optionally the full tensor and principal-eigenvector volumes.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use cs_dmri::dti::{DtiFitConfig, RestoreConfig, fit_volume_restore_reporting};
use cs_dmri::io::aux::{sibling_path, write_3d_f32, write_3d_u8, write_3d_u32_as_f32, write_4d_f32};
use cs_dmri::io::dwi::load_dwi;
use cs_dmri::qspace::{BvecFrame, TORTOISE_DEFAULT_GMAX};
use cs_dmri::{
    Heartbeat, ProvenanceBuilder, ProvenanceMode, configure_rayon_threads, effective_thread_count,
};

#[derive(Parser, Debug)]
#[command(version, about = "Robust diffusion tensor fit with RESTORE (Chang et al., MRM 2005)")]
struct Cli {
    /// 4D DWI NIfTI input.
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

    /// Output fractional anisotropy (FA) NIfTI (3D, range [0, 1]).
    #[arg(long)]
    output_fa: PathBuf,
    /// Output mean diffusivity (MD) NIfTI (3D), in the inverse of the b-value
    /// units.
    #[arg(long)]
    output_md: PathBuf,
    /// Output S₀ NIfTI (3D): fitted signal at b=0, in the units of the input.
    #[arg(long)]
    output_s0: PathBuf,
    /// Output outlier-fraction NIfTI (3D, range [0, 1]): the fraction of
    /// measurements in each voxel classified as outliers (see
    /// --outlier-threshold).
    #[arg(long)]
    output_outlier_fraction: PathBuf,

    /// Output tensor NIfTI (4D, six components in the order Dxx, Dxy, Dxz,
    /// Dyy, Dyz, Dzz).
    #[arg(long)]
    output_tensor: Option<PathBuf>,
    /// Output principal eigenvector NIfTI (4D, three components).
    #[arg(long)]
    output_principal_dir: Option<PathBuf>,

    /// Maximum number of RESTORE reweighting iterations.
    #[arg(long, default_value_t = 50)]
    max_iter: usize,
    /// RESTORE convergence tolerance on the relative change in tensor
    /// coefficients.
    #[arg(long, default_value_t = 1e-6)]
    tol: f64,
    /// Geman-McClure weight below which a measurement is counted as an
    /// outlier in the outlier-fraction map. Does not affect the fit.
    #[arg(long, default_value_t = 0.04)]
    outlier_threshold: f64,

    /// Diffusion time Δ (big delta), in seconds. Not used by the tensor fit;
    /// accepted for consistency with `cs-fit`.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Gradient pulse duration δ (small delta), in seconds. Not used by the
    /// tensor fit.
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude, in T/m. Used only when Δ and δ are
    /// estimated.
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,

    /// Also write per-voxel iteration-count (`_iters`) and convergence
    /// (`_converged`) maps next to the FA output.
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

    /// Interval between progress messages during the voxel-wise fit, in
    /// seconds.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance mode. Accepted for consistency with the other tools; `cs-dti`
    /// writes no provenance record.
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let provenance_builder = ProvenanceBuilder::new("cs-dti", args.provenance);
    let (threads, source) = configure_rayon_threads(args.threads)?;
    if !args.quiet {
        let n = if threads == 0 { effective_thread_count() } else { threads };
        eprintln!("[cs-dti] threads={} source={}", n, source.as_str());
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
            "[cs-dti] DWI shape {:?}, mask voxels: {}",
            dwi.shape(),
            mask_voxels
        );
    }
    cs_dmri::qc::report_input_qc("cs-dti", &dwi, args.quiet);

    let cfg = RestoreConfig {
        max_iter: args.max_iter,
        tol: args.tol,
        outlier_threshold: args.outlier_threshold,
        ..RestoreConfig::default()
    };

    let interval = Duration::from_secs(args.progress_interval_secs.max(1));
    let heartbeat = Heartbeat::new("cs-dti", mask_voxels, interval, args.quiet);

    let result = fit_volume_restore_reporting(
        &dwi,
        &cfg,
        DtiFitConfig { compute_diagnostics: args.diagnostics },
        || heartbeat.tick(),
    )
    .with_context(|| "DTI fit failed")?;

    heartbeat.finish();

    let threads_used = effective_thread_count();
    let _provenance = provenance_builder.map(|b| b.finish(threads_used));

    write_3d_f32(&args.output_fa, &args.dwi, &result.fa, args.overwrite)
        .with_context(|| format!("write FA {:?}", args.output_fa))?;
    write_3d_f32(&args.output_md, &args.dwi, &result.md, args.overwrite)
        .with_context(|| format!("write MD {:?}", args.output_md))?;
    write_3d_f32(&args.output_s0, &args.dwi, &result.s0, args.overwrite)
        .with_context(|| format!("write S0 {:?}", args.output_s0))?;
    write_3d_f32(
        &args.output_outlier_fraction,
        &args.dwi,
        &result.outlier_fraction,
        args.overwrite,
    )
    .with_context(|| format!("write outlier fraction {:?}", args.output_outlier_fraction))?;

    if let Some(p) = &args.output_tensor {
        write_4d_f32(p, &args.dwi, &result.tensor, args.overwrite)
            .with_context(|| format!("write tensor {:?}", p))?;
    }
    if let Some(p) = &args.output_principal_dir {
        write_4d_f32(p, &args.dwi, &result.principal_dir, args.overwrite)
            .with_context(|| format!("write principal direction {:?}", p))?;
    }

    if !args.quiet {
        eprintln!("[cs-dti] wrote FA → {}", args.output_fa.display());
        eprintln!("[cs-dti] wrote MD → {}", args.output_md.display());
        eprintln!("[cs-dti] wrote S0 → {}", args.output_s0.display());
        eprintln!(
            "[cs-dti] wrote outlier fraction → {}",
            args.output_outlier_fraction.display()
        );
        // Quick sanity stats over the mask.
        let mut fa_sum = 0.0_f64;
        let mut outlier_sum = 0.0_f64;
        let mut n = 0usize;
        for ((idx, &fa), &m) in result.fa.indexed_iter().zip(dwi.mask.iter()) {
            let _ = idx;
            if m {
                fa_sum += fa as f64;
                n += 1;
            }
        }
        for ((idx, &of), &m) in result.outlier_fraction.indexed_iter().zip(dwi.mask.iter()) {
            let _ = idx;
            if m {
                outlier_sum += of as f64;
            }
        }
        if n > 0 {
            eprintln!(
                "[cs-dti] mean masked FA = {:.4}, mean outlier fraction = {:.4}",
                fa_sum / n as f64,
                outlier_sum / n as f64
            );
        }
    }

    if args.diagnostics {
        if let Some(iters) = &result.iterations {
            let p = sibling_path(&args.output_fa, "_iters.nii.gz");
            write_3d_u32_as_f32(&p, &args.dwi, iters, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-dti] wrote iter map → {}", p.display());
            }
        }
        if let Some(conv) = &result.converged {
            let p = sibling_path(&args.output_fa, "_converged.nii.gz");
            write_3d_u8(&p, &args.dwi, conv, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-dti] wrote converged map → {}", p.display());
            }
        }
    }

    Ok(())
}
