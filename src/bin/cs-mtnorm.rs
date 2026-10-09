// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-mtnorm`: native-Rust multi-tissue intensity normalization.
//!
//! Rust port of the log-domain algorithm behind MRtrix3's `mtnormalise`
//! (Raffelt 2017 / Dhollander 2021), including per-tissue balance factors
//! and iterative IQR outlier rejection. Useful as the final step in a
//! single-shell SS3T pipeline so downstream tractography / FOD-based
//! analyses see tissue values on a consistent scale across the brain.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{Array3, Array4};
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

use cs_dmri::io::output::{sibling_path, write_3d_f32, write_4d_f32};
use cs_dmri::multitissue::mtnormalise::{
    MtnormaliseConfig, mtnormalise, target_sum_mrtrix_default,
};
use cs_dmri::CsDmriError;

#[derive(Parser, Debug)]
#[command(version, about = "Multi-tissue intensity normalization and bias field correction in the log domain (Raffelt et al., 2017; Dhollander et al., 2021)")]
struct Cli {
    /// Input white matter FOD NIfTI (4D, one volume per SH coefficient).
    #[arg(long)]
    in_wm: PathBuf,
    /// Input gray matter NIfTI (3D, or 4D with one volume).
    #[arg(long)]
    in_gm: PathBuf,
    /// Input CSF NIfTI (3D, or 4D with one volume).
    #[arg(long)]
    in_csf: PathBuf,
    /// Brain mask NIfTI (3D, binary).
    #[arg(long)]
    mask: PathBuf,

    /// Output normalized white matter FOD NIfTI (4D, one volume per SH
    /// coefficient).
    #[arg(long)]
    out_wm: PathBuf,
    /// Output normalized gray matter NIfTI (4D, one volume).
    #[arg(long)]
    out_gm: PathBuf,
    /// Output normalized CSF NIfTI (4D, one volume).
    #[arg(long)]
    out_csf: PathBuf,

    /// Order of the polynomial bias field model (order 3 has 20 terms).
    #[arg(long, default_value_t = 3)]
    poly_order: usize,

    /// Target value for the sum over tissues of the normalized l=0 SH
    /// coefficients in each voxel. Default: 1/√(4π), as in MRtrix3
    /// `mtnormalise`.
    #[arg(long)]
    target_sum: Option<f64>,

    /// Use the median of the observed sums as the target. The global scale
    /// of the input is preserved and only the spatial bias field is removed.
    #[arg(long, conflicts_with = "target_sum")]
    target_median: bool,

    /// Number of outer iterations (bias field updates).
    #[arg(long, default_value_t = 15)]
    niter: usize,

    /// Maximum number of iterations of the inner balance-factor and
    /// outlier-rejection loop in each outer iteration.
    #[arg(long, default_value_t = 7)]
    balance_maxiter: usize,

    /// Multiply each output tissue by its balance factor, as in MRtrix3
    /// `mtnormalise -balanced`. Balance factors are used in bias field
    /// estimation in either case; this option applies them to the outputs,
    /// which changes the relative scale of the tissues.
    #[arg(long)]
    balanced: bool,

    /// Also write the estimated bias field next to the white matter output
    /// (`<wm_stem>_bias.nii.gz`).
    #[arg(long)]
    diagnostics: bool,

    /// Overwrite existing output files. Without this flag, existing outputs
    /// cause an error.
    #[arg(long)]
    overwrite: bool,

    /// Suppress per-step summary messages.
    #[arg(long)]
    quiet: bool,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let mut wm = load_4d(&args.in_wm).with_context(|| format!("read {:?}", args.in_wm))?;
    let mut gm = load_4d_promote(&args.in_gm).with_context(|| format!("read {:?}", args.in_gm))?;
    let mut csf =
        load_4d_promote(&args.in_csf).with_context(|| format!("read {:?}", args.in_csf))?;
    let mask = load_3d_mask(&args.mask).with_context(|| format!("read {:?}", args.mask))?;

    if !args.quiet {
        eprintln!(
            "[cs-mtnorm] WM {:?} ({} SH coefs)  GM {:?}  CSF {:?}  mask voxels: {}",
            &wm.shape()[..3],
            wm.shape()[3],
            &gm.shape()[..3],
            &csf.shape()[..3],
            mask.iter().filter(|&&m| m).count()
        );
    }

    let target_sum = if let Some(t) = args.target_sum {
        Some(t)
    } else if args.target_median {
        None
    } else {
        // Default: MRtrix3 mtnormalise convention.
        Some(target_sum_mrtrix_default())
    };

    let cfg = MtnormaliseConfig {
        poly_order: args.poly_order,
        target_sum,
        niter: args.niter,
        balance_maxiter: args.balance_maxiter,
        apply_balance: args.balanced,
    };
    let diag = mtnormalise(&mut wm, &mut gm, &mut csf, &mask, &cfg)
        .with_context(|| "mtnormalise failed")?;

    if !args.quiet {
        eprintln!(
            "[cs-mtnorm] target_sum={:.4}  poly_order={}  niter={}  fit voxels={}  mean |log residual|={:.3e}",
            diag.target_sum_used,
            args.poly_order,
            diag.iterations,
            diag.n_fit_voxels,
            diag.mean_abs_log_residual
        );
        eprintln!(
            "[cs-mtnorm] balance factors{}: b_WM={:.4}  b_GM={:.4}  b_CSF={:.4}",
            if args.balanced { " (applied to output)" } else { "" },
            diag.tissue_scales[0], diag.tissue_scales[1], diag.tissue_scales[2]
        );
    }

    write_4d_f32(&args.out_wm, &args.in_wm, &wm, args.overwrite)
        .with_context(|| format!("write {:?}", args.out_wm))?;
    write_4d_f32(&args.out_gm, &args.in_wm, &gm, args.overwrite)
        .with_context(|| format!("write {:?}", args.out_gm))?;
    write_4d_f32(&args.out_csf, &args.in_wm, &csf, args.overwrite)
        .with_context(|| format!("write {:?}", args.out_csf))?;

    if !args.quiet {
        eprintln!("[cs-mtnorm] wrote normalized WM → {}", args.out_wm.display());
        eprintln!("[cs-mtnorm] wrote normalized GM → {}", args.out_gm.display());
        eprintln!(
            "[cs-mtnorm] wrote normalized CSF → {}",
            args.out_csf.display()
        );
    }

    if args.diagnostics {
        let bias_path = sibling_path(&args.out_wm, "_bias.nii.gz");
        write_3d_f32(&bias_path, &args.in_wm, &diag.bias_field, args.overwrite)
            .with_context(|| format!("write {:?}", bias_path))?;
        if !args.quiet {
            eprintln!("[cs-mtnorm] wrote bias field → {}", bias_path.display());
        }
    }

    Ok(())
}

fn load_4d(path: &std::path::Path) -> Result<Array4<f32>> {
    let obj = ReaderOptions::new().read_file(path)?;
    let nd = obj.into_volume().into_ndarray::<f32>()?;
    Ok(nd
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| CsDmriError::Dimension(format!("expected 4D NIfTI: {e}")))?)
}

/// Load a 3D-or-4D NIfTI as Array4<f32> (promoting 3D to 4D-with-singleton).
fn load_4d_promote(path: &std::path::Path) -> Result<Array4<f32>> {
    let obj = ReaderOptions::new().read_file(path)?;
    let nd = obj.into_volume().into_ndarray::<f32>()?;
    match nd.ndim() {
        4 => Ok(nd
            .into_dimensionality::<ndarray::Ix4>()
            .expect("verified ndim=4 above")),
        3 => {
            let arr3 = nd
                .into_dimensionality::<ndarray::Ix3>()
                .expect("verified ndim=3 above");
            let s = arr3.shape();
            let (nx, ny, nz) = (s[0], s[1], s[2]);
            let mut arr4 = Array4::<f32>::zeros((nx, ny, nz, 1));
            for x in 0..nx {
                for y in 0..ny {
                    for z in 0..nz {
                        arr4[(x, y, z, 0)] = arr3[(x, y, z)];
                    }
                }
            }
            Ok(arr4)
        }
        d => Err(CsDmriError::Dimension(format!(
            "expected 3D or 4D NIfTI, got {d}D"
        ))
        .into()),
    }
}

fn load_3d_mask(path: &std::path::Path) -> Result<Array3<bool>> {
    let obj = ReaderOptions::new().read_file(path)?;
    let nd = obj.into_volume().into_ndarray::<f32>()?;
    let arr3 = nd
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| CsDmriError::Dimension(format!("mask must be 3D: {e}")))?;
    Ok(arr3.mapv(|v| v > 0.5))
}

