// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-synth`: load coefficient NIfTI + new bval/bvec → synthesize 4D NIfTI.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use cs_dmri::basis::{Basis, BasisMetadata};
use cs_dmri::io::coeffs::CoefficientsFile;
use cs_dmri::qspace::{
    BvecFrame, GradientTable, TORTOISE_DEFAULT_GMAX, affine_rotation, rotate_bvecs,
};
use cs_dmri::{
    Heartbeat, ProvenanceBuilder, ProvenanceMode, ShoreBasis, atomic_write,
    effective_thread_count, synthesize_volume_reporting,
};
use nifti::writer::WriterOptions;
use odx_rs::reference_affine::read_reference_affine;

#[derive(Parser, Debug)]
#[command(version, about = "Synthesize diffusion-weighted images from SHORE coefficients for a given gradient table")]
struct Cli {
    /// Coefficient NIfTI written by `cs-fit`. The JSON sidecar is read from the
    /// matching `.json` file next to it.
    #[arg(long)]
    coeffs: PathBuf,
    /// FSL bval file of the gradient table to synthesize.
    #[arg(long)]
    bval: PathBuf,
    /// FSL bvec file of the gradient table to synthesize.
    #[arg(long)]
    bvec: PathBuf,
    /// Diffusion time Δ (big delta), in seconds. If omitted, the value in the
    /// coefficient sidecar is used.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Gradient pulse duration δ (small delta), in seconds. If omitted, the
    /// value in the coefficient sidecar is used.
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude, in T/m. Used only when Δ and δ are absent
    /// from the sidecar and must be estimated.
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,
    /// Output 4D DWI NIfTI.
    #[arg(long)]
    output: PathBuf,

    /// Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used,
    /// then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU.
    #[arg(long)]
    threads: Option<usize>,

    /// Overwrite an existing output NIfTI and sidecar JSON. Without this
    /// flag, existing outputs cause an error.
    #[arg(long)]
    overwrite: bool,

    /// Suppress periodic progress and per-step summary messages.
    #[arg(long)]
    quiet: bool,

    /// Interval between progress messages during synthesis, in seconds.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance recorded in a JSON sidecar next to the output; with `none`,
    /// no sidecar is written.
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let provenance_builder = ProvenanceBuilder::new("cs-synth", args.provenance);
    let (threads, source) = cs_dmri::configure_rayon_threads(args.threads)?;
    if !args.quiet {
        let n = if threads == 0 { effective_thread_count() } else { threads };
        eprintln!("[cs-synth] threads={} source={}", n, source.as_str());
    }
    let coeffs = CoefficientsFile::read(&args.coeffs)
        .with_context(|| format!("read {:?}", args.coeffs))?;
    if coeffs.metadata.is_single_shell() {
        eprintln!(
            "[cs-synth] warning: the coefficients come from single-shell data; signals at b-values \
             other than that shell are extrapolated by the regularization, not the measurements"
        );
    }

    let bvals = parse_bvals(&fs::read_to_string(&args.bval)?)?;
    let mut bvecs = parse_bvecs(&fs::read_to_string(&args.bvec)?)?;

    // The coefficients were fit in either image-axis or world-RAS; the new
    // bvecs are FSL/image-axis. Rotate them to match before sampling.
    if matches!(coeffs.metadata.bvec_frame, BvecFrame::WorldRas) {
        let affine = read_reference_affine(&args.coeffs).map_err(|e| {
            anyhow::anyhow!("failed to read affine from {:?}: {e}", args.coeffs)
        })?;
        let rotation = affine_rotation(&affine);
        rotate_bvecs(&mut bvecs, &rotation);
    }

    // Pull deltas from sidecar unless overridden.
    let big_delta = args.big_delta.or(Some(coeffs.metadata.big_delta_seconds));
    let small_delta = args.small_delta.or(Some(coeffs.metadata.small_delta_seconds));

    let gtab = GradientTable::new(bvals, bvecs, big_delta, small_delta, Some(args.gmax))?;

    let basis = match coeffs.metadata.basis {
        BasisMetadata::Shore { radial_order, zeta } => ShoreBasis::new(radial_order, zeta),
    };

    if coeffs.coeffs.shape()[3] != basis.n_coeffs() {
        bail!(
            "coefficient channel count {} does not match basis size {}",
            coeffs.coeffs.shape()[3],
            basis.n_coeffs()
        );
    }

    let nx = coeffs.coeffs.shape()[0];
    let interval = Duration::from_secs(args.progress_interval_secs.max(1));
    let hb = Heartbeat::new("cs-synth", nx, interval, args.quiet);
    let mut synth =
        synthesize_volume_reporting(&coeffs.coeffs, &basis, &gtab, || hb.tick());
    hb.finish();
    // DWI signal is physically non-negative; the SHORE basis can swing
    // slightly negative on samples it didn't see during fit. Clamp so
    // downstream tools and visual comparisons aren't surprised by sign.
    synth.mapv_inplace(|v| v.max(0.0));

    // Inherit the spatial metadata from the coefficients NIfTI; only the 4th
    // dimension changes.
    let mut header = nifti::NiftiHeader::from_file(&args.coeffs)
        .with_context(|| "read coefficients header")?;
    header.dim[0] = 4;
    header.dim[4] = synth.shape()[3] as u16;
    for i in 5..8 {
        header.dim[i] = 1;
    }
    header.datatype = 16;
    header.bitpix = 32;
    header.scl_slope = 0.0;
    header.scl_inter = 0.0;

    atomic_write(&args.output, args.overwrite, |tmp| {
        WriterOptions::new(tmp)
            .reference_header(&header)
            .write_nifti(&synth)
            .map_err(|e| cs_dmri::CsDmriError::Other(format!("write {tmp:?}: {e}")))
    })
    .with_context(|| format!("write {:?}", args.output))?;

    if let Some(builder_p) = provenance_builder {
        let prov = builder_p.finish(effective_thread_count());
        let sidecar_path = sidecar_path_for(&args.output);
        let json = serde_json::to_string_pretty(&prov)
            .with_context(|| "serialize provenance sidecar")?;
        atomic_write(&sidecar_path, args.overwrite, |tmp| {
            Ok(fs::write(tmp, &json)?)
        })
        .with_context(|| format!("write {:?}", sidecar_path))?;
        if !args.quiet {
            eprintln!("[cs-synth] wrote provenance sidecar: {}", sidecar_path.display());
        }
    }

    if !args.quiet {
        eprintln!(
            "[cs-synth] wrote {} from {} coefficients × {} gradients",
            args.output.display(),
            basis.n_coeffs(),
            synth.shape()[3],
        );
    }
    Ok(())
}

/// Mirrors `cs-fit`'s sidecar naming: `<stem>.json` next to the NIfTI.
fn sidecar_path_for(nifti_path: &Path) -> PathBuf {
    let s = nifti_path.to_string_lossy();
    let stem = if let Some(stripped) = s.strip_suffix(".nii.gz") {
        stripped
    } else if let Some(stripped) = s.strip_suffix(".nii") {
        stripped
    } else {
        s.as_ref()
    };
    PathBuf::from(format!("{}.json", stem))
}

fn parse_bvals(text: &str) -> Result<Vec<f64>> {
    text.split_whitespace()
        .map(|tok| {
            tok.parse::<f64>()
                .with_context(|| format!("bval token '{tok}'"))
        })
        .collect()
}

fn parse_bvecs(text: &str) -> Result<Vec<[f64; 3]>> {
    let lines: Vec<Vec<f64>> = text
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|t| t.parse::<f64>())
                .collect::<std::result::Result<Vec<f64>, _>>()
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| "parsing bvecs")?;
    let lines: Vec<Vec<f64>> = lines.into_iter().filter(|l| !l.is_empty()).collect();
    if lines.len() == 3 {
        let n = lines[0].len();
        if lines[1].len() != n || lines[2].len() != n {
            bail!("bvec rows have inconsistent lengths");
        }
        Ok((0..n)
            .map(|i| [lines[0][i], lines[1][i], lines[2][i]])
            .collect())
    } else if lines.iter().all(|l| l.len() == 3) {
        Ok(lines.iter().map(|l| [l[0], l[1], l[2]]).collect())
    } else {
        bail!("could not interpret bvec file")
    }
}
