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
#[command(version, about = "Project SHORE coefficients to Tournier ODF SH and write ODX")]
struct Cli {
    /// Coefficient NIfTI from cs-fit (sidecar JSON read alongside).
    #[arg(long)]
    coeffs: PathBuf,
    /// Output .odx path (or directory when --directory is set).
    #[arg(long)]
    output: PathBuf,
    /// Optional brain mask NIfTI; defaults to "any nonzero coefficient voxel".
    #[arg(long)]
    mask: Option<PathBuf>,
    /// Maximum even SH order; defaults to largest even ≤ radial_order.
    #[arg(long)]
    lmax: Option<u32>,
    /// Field name under sh/ in the ODX. Default "coefficients" (the field name
    /// trxviz and odx-rs's mrtrix loader both expect for SH glyph rendering).
    #[arg(long, default_value = "coefficients")]
    name: String,
    /// Emit a directory tree instead of a zipped .odx archive.
    #[arg(long)]
    directory: bool,

    /// Skip the per-voxel anisotropic-power DPV (Dell'Acqua 2014). DPV is what
    /// most ODX viewers render as a slice background; only set this if you
    /// already have your own scalar map to display.
    #[arg(long)]
    no_anisotropic_power: bool,

    /// `norm_factor` for the AP log-shift; matches dipy's default 1e-5.
    #[arg(long, default_value_t = ANISOTROPIC_POWER_NORM_FACTOR)]
    ap_norm_factor: f64,

    /// Skip DSI-Studio-style global ODF normalization. By default we compute
    /// per-voxel QA = max(ODF) − min(ODF), take the global maximum across the
    /// brain, and divide every voxel's SH coefficients by that scalar. This
    /// preserves *relative* amplitudes between voxels (high-FA stays larger
    /// than low-FA) while capping the brightest peak at 1, which is what most
    /// ODF viewers assume when sizing glyphs.
    #[arg(long)]
    no_global_normalize: bool,

    /// Skip auto-loading of per-voxel diagnostic sibling NIfTIs (`<stem>_r2.nii.gz`,
    /// `<stem>_rmse.nii.gz`, `<stem>_alpha.nii.gz`, `<stem>_bic.nii.gz`,
    /// `<stem>_sparsity.nii.gz`) emitted by `cs-fit --diagnostics`. By default
    /// we look for these next to the coefficients NIfTI and copy each one that
    /// exists into the ODX as a DPV.
    #[arg(long)]
    no_diagnostic_dpvs: bool,

    /// Skip per-voxel peak (fixel) extraction. By default we delegate to
    /// `odx-rs::peak_finder::SpherePeakFinder::find_peaks_with_sh`: sample each
    /// ODF on the DSI Studio ODF8 hemisphere (321 vertices) to seed local
    /// maxima, prune with relative-threshold + separation-angle, then
    /// Newton-refine each accepted seed in continuous SH (mirroring MRtrix's
    /// `Math::SH::get_peak`) so the recorded peak directions are sub-vertex.
    /// Disable to keep the output SH-only.
    #[arg(long)]
    no_peaks: bool,

    /// Maximum number of peaks per voxel.
    #[arg(long, default_value_t = SHORE_ODX_DEFAULT_NPEAKS)]
    peak_npeaks: usize,

    /// Drop peaks below this fraction of the voxel's strongest peak.
    #[arg(long, default_value_t = SHORE_ODX_DEFAULT_REL_THRESH)]
    peak_relative_threshold: f32,

    /// Minimum angular separation (degrees) between accepted peaks.
    #[arg(long, default_value_t = SHORE_ODX_DEFAULT_MIN_SEP_DEG)]
    peak_min_separation_deg: f32,

    /// Cap rayon's worker threads. If unset, picks up `$SLURM_CPUS_PER_TASK`,
    /// then `$RAYON_NUM_THREADS`, else uses one worker per logical CPU.
    #[arg(long)]
    threads: Option<usize>,

    /// Allow overwriting an existing output ODX, output directory, or
    /// sibling microstructure NIfTI. Default: refuse.
    #[arg(long)]
    overwrite: bool,

    /// Suppress periodic progress heartbeat and per-step summary lines.
    #[arg(long)]
    quiet: bool,

    /// Seconds between heartbeat lines during long parallel loops. Default 30.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance captured into the ODX (as the extra value
    /// `cs_dmri_provenance`). `minimal` (default) emits no PHI surface;
    /// `full` adds argv, hostname, and wall-clock start time; `none` skips it.
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,

    /// Skip computing SHORE/MAPMRI propagator microstructure scalars (RTOP,
    /// RTAP, RTPP, MSD, QIV, NG). By default these are computed per voxel,
    /// fit-failure outliers are rejected to NaN (see
    /// `--microstructure-outlier-factor`), and the result is embedded as
    /// DPVs in the output ODX alongside per-scalar display-range stats
    /// (`cs_dmri_microstructure_display`) and reject counts
    /// (`cs_dmri_microstructure_outliers`). RTAP/RTPP need a first-peak
    /// direction; voxels without a detected peak come back as NaN.
    ///
    /// Closed-form derivations: see `cs-dmri/scripts/microstructure_math.md`.
    #[arg(long)]
    no_microstructure: bool,

    /// Also write each microstructure scalar to a sibling NIfTI of the ODX
    /// (`<output_stem>_rtop.nii.gz`, …) using the canonical RAS+ affine.
    /// Off by default — the DPVs in the ODX cover the visualization use case.
    #[arg(long, alias = "microstructure")]
    microstructure_nifti: bool,

    /// NaN microstructure-scalar values above `K × p99` per scalar. With
    /// `p99/median ≈ 8` for these scalars, K=10 corresponds to "anything
    /// more than ~80× the brain median is a fit failure, not signal."
    /// Bigger values reject only the most pathological outliers; smaller
    /// trims further into the upper tail. Set to a very large number
    /// (effectively `inf`) to keep all finite values; pair with
    /// `--no-microstructure-outlier-rejection` to disable entirely.
    #[arg(long, default_value_t = 10.0)]
    microstructure_outlier_factor: f32,

    /// Disable fit-failure rejection entirely — every finite microstructure
    /// value lands in the ODX, including the `5000× the median` outliers
    /// that arise from degenerate SHORE fits. Use only for debugging fit
    /// quality (you'll see the unphysical voxels intact).
    #[arg(long)]
    no_microstructure_outlier_rejection: bool,

    /// Length unit for emitted microstructure scalars.
    ///
    /// `um` (default) matches TORTOISE's `EstimateMAPMRI` output convention:
    /// q-vectors expressed in 1/μm, so RTOP is in /μm³, RTAP in /μm², RTPP in
    /// /μm, MSD in μm², QIV in μm⁵. Values fall in TORTOISE's familiar
    /// [0, ~few] range for brain tissue.
    ///
    /// `mm` keeps the dipy / cs-dmri-internal convention (q in 1/mm), which
    /// makes RTOP ~1e5/mm³ — physically equivalent, just larger numbers.
    #[arg(long, value_enum, default_value_t = ScalarUnits::Um)]
    scalar_units: ScalarUnits,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum ScalarUnits {
    /// TORTOISE convention: q in 1/μm, RTOP in /μm³, etc.
    Um,
    /// dipy / cs-dmri-internal convention: q in 1/mm, RTOP in /mm³, etc.
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
