// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-odf`: project SHORE coefficients to Tournier-ordered ODF SH and write an ODX.
//!
//! Reads a coefficient NIfTI (+ JSON sidecar) produced by `cs-fit`, applies the
//! analytical SHORE → ODF SH operator, and writes a real ODX file containing the
//! mrtrix3-ordered SH coefficients per masked voxel.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use ndarray::Array3;
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

use cs_dmri::ShoreBasis;
use cs_dmri::basis::BasisMetadata;
use cs_dmri::io::coeffs::CoefficientsFile;
use cs_dmri::io::microstructure::{
    MicrostructureOptions, MicrostructureOutlierRejection, MicrostructureUnits,
    write_microstructure_nifti_siblings,
};
use cs_dmri::io::odx_out::{
    ShoreOdxOptions, ShoreOdxPeakOpts, ShoreToOdxOptions, finalize_and_write_odx,
    shore_coeffs_to_odx, SHORE_ODX_DEFAULT_NPEAKS, SHORE_ODX_DEFAULT_REL_THRESH,
    SHORE_ODX_DEFAULT_MIN_SEP_DEG,
};
use cs_dmri::{ProvenanceBuilder, ProvenanceMode, effective_thread_count};

use odx_rs::mrtrix_sh::ANISOTROPIC_POWER_NORM_FACTOR;
use odx_rs::reference_affine::read_reference_affine;

#[derive(Parser, Debug)]
#[command(version, about = "Project SHORE coefficients onto ODF spherical harmonics (MRtrix3/Tournier convention) and write an ODX file")]
struct Cli {
    /// Coefficient NIfTI written by cs-fit. The JSON sidecar is read from the
    /// same location.
    #[arg(long)]
    coeffs: PathBuf,
    /// Output .odx file, or output directory with --directory.
    #[arg(long)]
    output: PathBuf,
    /// Brain mask NIfTI. If omitted, all voxels with at least one non-zero
    /// coefficient are used.
    #[arg(long)]
    mask: Option<PathBuf>,
    /// Maximum (even) SH order. Default: the largest even integer ≤ the SHORE
    /// radial order.
    #[arg(long)]
    lmax: Option<u32>,
    /// Name of the SH field under sh/ in the ODX.
    #[arg(long, default_value = "coefficients")]
    name: String,
    /// Write the ODX as a directory instead of a zip archive.
    #[arg(long)]
    directory: bool,

    /// Do not compute the per-voxel anisotropic power map (Dell'Acqua et al.,
    /// 2014).
    #[arg(long)]
    no_anisotropic_power: bool,

    /// Normalization factor in the logarithm of the anisotropic power map.
    #[arg(long, default_value_t = ANISOTROPIC_POWER_NORM_FACTOR)]
    ap_norm_factor: f64,

    /// Do not apply global ODF normalization. By default the quantity
    /// QA = max(ODF) − min(ODF) is computed in each voxel, and all SH
    /// coefficients are divided by its maximum over the mask, as in DSI
    /// Studio. Relative amplitudes between voxels are preserved.
    #[arg(long)]
    no_global_normalize: bool,

    /// Do not copy the diagnostic maps written by `cs-fit --diagnostics`
    /// (`<stem>_r2.nii.gz`, `<stem>_rmse.nii.gz`, `<stem>_alpha.nii.gz`,
    /// `<stem>_bic.nii.gz`, `<stem>_sparsity.nii.gz`) into the ODX. By
    /// default, each of these found next to the coefficient NIfTI is stored
    /// as a per-voxel field.
    #[arg(long)]
    no_diagnostic_dpvs: bool,

    /// Do not extract ODF peaks (fixels). By default, local maxima of each ODF
    /// are located on the DSI Studio ODF8 hemisphere (321 vertices), filtered
    /// by relative amplitude and angular separation, and refined by Newton
    /// iteration on the continuous SH representation.
    #[arg(long)]
    no_peaks: bool,

    /// Maximum number of peaks per voxel.
    #[arg(long, default_value_t = SHORE_ODX_DEFAULT_NPEAKS)]
    peak_npeaks: usize,

    /// Discard peaks with amplitude below this fraction of the largest peak in
    /// the voxel.
    #[arg(long, default_value_t = SHORE_ODX_DEFAULT_REL_THRESH)]
    peak_relative_threshold: f32,

    /// Minimum angular separation between peaks, in degrees.
    #[arg(long, default_value_t = SHORE_ODX_DEFAULT_MIN_SEP_DEG)]
    peak_min_separation_deg: f32,

    /// Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used,
    /// then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU.
    #[arg(long)]
    threads: Option<usize>,

    /// Overwrite an existing output ODX, output directory or microstructure
    /// NIfTI. Without this flag, existing outputs cause an error.
    #[arg(long)]
    overwrite: bool,

    /// Suppress periodic progress and per-step summary messages.
    #[arg(long)]
    quiet: bool,

    /// Interval between progress messages, in seconds.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance recorded in the ODX as the extra value `cs_dmri_provenance`.
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,

    /// Do not compute propagator scalars from the SHORE coefficients: return
    /// to the origin, axis and plane probabilities (RTOP, RTAP, RTPP), mean
    /// squared displacement (MSD), q-space inverse variance (QIV) and
    /// non-Gaussianity (NG). By default these are computed in each voxel,
    /// outliers are set to NaN (see `--microstructure-outlier-factor`), and
    /// the maps are stored as per-voxel fields in the ODX together with
    /// per-scalar display ranges (`cs_dmri_microstructure_display`) and
    /// outlier counts (`cs_dmri_microstructure_outliers`). RTAP and RTPP are
    /// defined relative to the first peak direction and are NaN in voxels
    /// without a peak.
    #[arg(long)]
    no_microstructure: bool,

    /// Also write each propagator scalar to a NIfTI next to the ODX
    /// (`<output_stem>_rtop.nii.gz`, …) with the RAS+ affine.
    #[arg(long, alias = "microstructure")]
    microstructure_nifti: bool,

    /// Outlier threshold factor K: values of a propagator scalar greater
    /// than K times the 99th percentile of its finite values within the mask
    /// are set to NaN.
    #[arg(long, default_value_t = 10.0)]
    microstructure_outlier_factor: f32,

    /// Disable outlier rejection for propagator scalars. All finite values
    /// are retained.
    #[arg(long)]
    no_microstructure_outlier_rejection: bool,

    /// Length unit of the propagator scalars.
    #[arg(long, value_enum, default_value_t = ScalarUnits::Um)]
    scalar_units: ScalarUnits,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum ScalarUnits {
    /// q in 1/μm: RTOP in μm⁻³, RTAP in μm⁻², RTPP in μm⁻¹, MSD in μm², QIV
    /// in μm⁵. Default.
    Um,
    /// q in 1/mm: RTOP in mm⁻³, RTAP in mm⁻², RTPP in mm⁻¹, MSD in mm², QIV
    /// in mm⁵.
    Mm,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let provenance_builder = ProvenanceBuilder::new("cs-odf", args.provenance);
    let (threads, source) = cs_dmri::configure_rayon_threads(args.threads)?;
    if !args.quiet {
        let n = if threads == 0 { effective_thread_count() } else { threads };
        eprintln!("[cs-odf] threads={} source={}", n, source.as_str());
    }

    let coeffs_file = CoefficientsFile::read(&args.coeffs)
        .with_context(|| format!("failed to read coefficients from {:?}", args.coeffs))?;
    let raw_coeffs = coeffs_file.coeffs;
    let meta = &coeffs_file.metadata;

    // `shore_coeffs_to_odx` canonicalizes everything to RAS+ so the on-disk
    // ODX matches what `odx convert` produces from the same NIfTI.
    let raw_affine = read_reference_affine(&args.coeffs)
        .map_err(|e| anyhow!("failed to read affine from {:?}: {e}", args.coeffs))?;
    let (radial_order, zeta) = match meta.basis {
        BasisMetadata::Shore { radial_order, zeta } => (radial_order, zeta),
    };
    let basis = ShoreBasis::new(radial_order, zeta);
    let raw_mask = args
        .mask
        .as_deref()
        .map(read_mask_nifti)
        .transpose()?;

    // Diagnostic siblings the user wants captured as DPVs (R², RMSE, α, BIC,
    // sparsity).
    let loaded: Vec<(&'static str, Array3<f32>)> = if args.no_diagnostic_dpvs {
        Vec::new()
    } else {
        let mut out: Vec<(&'static str, Array3<f32>)> = Vec::new();
        for (suffix, name) in [
            ("_r2.nii.gz", "r2"),
            ("_rmse.nii.gz", "rmse"),
            ("_alpha.nii.gz", "alpha"),
            ("_bic.nii.gz", "bic"),
            ("_sparsity.nii.gz", "sparsity"),
        ] {
            let p = sibling_diagnostic_path(&args.coeffs, suffix);
            if !p.exists() {
                continue;
            }
            if !args.quiet {
                eprintln!("[cs-odf] added DPV '{name}' from {}", p.display());
            }
            out.push((name, read_3d_f32_nifti(&p)?));
        }
        out
    };
    let diagnostic_dpvs: Vec<(&str, &Array3<f32>)> =
        loaded.iter().map(|(n, a)| (*n, a)).collect();

    let microstructure = if args.no_microstructure {
        None
    } else {
        if args.no_peaks && !args.quiet {
            eprintln!(
                "[cs-odf] --no-peaks: RTAP/RTPP per voxel will be NaN \
                 (rerun without --no-peaks for direction-aware scalars)"
            );
        }
        let units = match args.scalar_units {
            ScalarUnits::Um => MicrostructureUnits::Um,
            ScalarUnits::Mm => MicrostructureUnits::Mm,
        };
        let outlier_rejection = if args.no_microstructure_outlier_rejection {
            None
        } else {
            if !(args.microstructure_outlier_factor > 0.0
                && args.microstructure_outlier_factor.is_finite())
            {
                bail!(
                    "--microstructure-outlier-factor must be a positive finite number, got {}",
                    args.microstructure_outlier_factor
                );
            }
            Some(MicrostructureOutlierRejection {
                p99_factor: args.microstructure_outlier_factor,
            })
        };
        Some(MicrostructureOptions {
            units,
            outlier_rejection,
            quiet: args.quiet,
        })
    };

    let opts = ShoreToOdxOptions {
        lmax: args.lmax,
        odx: ShoreOdxOptions {
            field_name: args.name.clone(),
            global_normalize: !args.no_global_normalize,
            anisotropic_power: !args.no_anisotropic_power,
            ap_norm_factor: args.ap_norm_factor,
            peaks: if args.no_peaks {
                None
            } else {
                Some(ShoreOdxPeakOpts {
                    npeaks: args.peak_npeaks,
                    relative_threshold: args.peak_relative_threshold,
                    min_separation_deg: args.peak_min_separation_deg,
                })
            },
            progress_interval_secs: args.progress_interval_secs,
            quiet: args.quiet,
        },
        microstructure,
    };
    let out = shore_coeffs_to_odx(
        &raw_coeffs,
        raw_affine,
        raw_mask.as_ref(),
        &basis,
        &diagnostic_dpvs,
        &opts,
    )?;
    let mut built = out.build;

    if let (Some(scalars), true) = (out.microstructure.as_ref(), args.microstructure_nifti) {
        write_microstructure_nifti_siblings(
            &args.output,
            &out.affine,
            out.spatial,
            &built.masked_indices,
            scalars,
            args.overwrite,
            args.quiet,
        )?;
    }

    if let Some(builder_p) = provenance_builder {
        let prov = builder_p.finish(effective_thread_count());
        let value = serde_json::to_value(&prov)
            .map_err(|e| anyhow!("provenance serialize: {e}"))?;
        built.builder.set_extra_value("cs_dmri_provenance", value);
    }

    finalize_and_write_odx(built.builder, &args.output, args.overwrite, args.directory)
        .with_context(|| format!("write {:?}", args.output))?;

    if !args.quiet {
        eprintln!("[cs-odf] wrote {}", args.output.display());
    }
    Ok(())
}

fn read_3d_f32_nifti(path: &Path) -> Result<Array3<f32>> {
    let obj = ReaderOptions::new()
        .read_file(path)
        .with_context(|| format!("read {:?}", path))?;
    let nd = obj.into_volume().into_ndarray::<f32>()?;
    nd.into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| anyhow!("{:?} not 3D: {e}", path))
}

fn read_mask_nifti(path: &Path) -> Result<Array3<bool>> {
    let raw = read_3d_f32_nifti(path)
        .with_context(|| format!("failed to read mask {:?}", path))?;
    Ok(raw.mapv(|v| v > 0.0))
}

///`<coeffs>.nii.gz` + `_alpha.nii.gz` → `<coeffs>_alpha.nii.gz`. Mirrors
/// cs-fit's `sibling_path` helper so the two binaries agree on the layout.
fn sibling_diagnostic_path(coeffs_nifti: &std::path::Path, suffix: &str) -> PathBuf {
    let s = coeffs_nifti.to_string_lossy();
    let stem = if let Some(stripped) = s.strip_suffix(".nii.gz") {
        stripped
    } else if let Some(stripped) = s.strip_suffix(".nii") {
        stripped
    } else {
        s.as_ref()
    };
    PathBuf::from(format!("{}{}", stem, suffix))
}
