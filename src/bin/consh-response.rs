// SPDX-License-Identifier: MIT OR Apache-2.0
//! `consh-response`: utilities for ConSH dense-b response functions.
//!
//! One subcommand, `mean`, pools per-scan responses into the common response
//! a quantitative study needs.
//!
//! This is the ConSH analogue of mrtrix3's `responsemean`. The motivation is
//! the same: fixel-based analysis and AFD quantification require that "one unit
//! of AFD" mean the same thing in every subject, which in turn requires a single
//! response shared across the study (Raffelt et al. 2012; Dhollander et al.
//! 2021; Smith et al. 2022). The difference is the format — a ConSH response is
//! a *dense-b* matrix with one zonal-SH row per b-grid point, rather than one
//! row per acquired shell — so `responsemean` cannot read it.
//!
//! # Magnitude compensation, matching `responsemean`'s default
//!
//! `responsemean`'s default derives a per-input multiplier — the geometric mean
//! over rows of `(group-mean l=0) / (input l=0)` — rescales each input by it,
//! and then averages. Direct averaging is its `-legacy` mode, described as
//! "without compensating for global magnitude differences between input files".
//!
//! That compensation exists because an mrtrix3 response carries per-scan
//! magnitude: `dwi2response` estimates it from raw signal, so WM PD·T2 (and, in
//! MRtrix's raw units, coil loading and receiver gain) are baked in, and
//! without rescaling the highest-magnitude inputs would dominate the mean.
//!
//! ConSH responses carry that same per-scan magnitude, minus the gain: every
//! tissue is divided by one common per-scan reference (the single-fiber-WM
//! median b=0) at estimation, which preserves the relative amplitudes between
//! tissues and leaves each scan's PD·T2 contrast and mean/median anchor ratio
//! in the coefficients. So this command applies exactly `responsemean`'s
//! default multiplier before pooling; `--legacy` pools directly, like
//! mrtrix3's `-legacy`.
//!
//! One guarded deviation: `responsemean` divides the log-multiplier by the
//! total row count and would fault on a non-positive l=0 term. A dense-b grid
//! has hundreds of rows and can legitimately fit ~0 signal at extreme b, so
//! rows with a non-positive l=0 term (in the input or the group mean) are
//! excluded from the multiplier and the divisor. When every row is positive —
//! the only case mrtrix3 defines — the two are identical.
//!
//! # Median
//!
//! mrtrix3 offers no median; `--statistic median` is an addition, not a match.
//! It is useful when one scan's single-fiber mask is bad enough to distort the
//! pool, which a mean cannot resist. The multiplier rescaling is applied before
//! the median as well. `mean` remains the default so results stay comparable to
//! `responsemean` unless robustness is explicitly asked for.
//!
//! Compatibility is checked rather than assumed: every input must share the
//! dense-b grid (row count) and `lmax` (column count), and each must have a
//! positive b=0 `l=0` term. A mismatch is an error, not a warning, because the
//! resulting kernel would silently produce fiber densities on incomparable
//! scales.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};

use cs_dmri::multitissue::response::TissueResponse;
use cs_dmri::multitissue::response_estimation::write_response_txt;
use cs_dmri::multitissue::response_pool::{
    PoolStatistic, check_poolable, deviations_pct, magnitude_multipliers, pool_tissue,
};

const TISSUES: [&str; 3] = ["wm", "gm", "csf"];

#[derive(Parser, Debug)]
#[command(version, about = "Utilities for ConSH dense-b response functions")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Pool per-scan dense-b responses into one common kernel.
    ///
    /// Inputs are directories written by `consh --write-responses-to`, each
    /// holding `{wm,gm,csf}_response_denseb.txt`. The output directory receives
    /// the pooled response in the same layout, ready for
    /// `consh --read-responses-from`.
    Mean(MeanArgs),
}

#[derive(Parser, Debug)]
struct MeanArgs {
    /// Response directories to pool (two or more).
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Output directory for the pooled response.
    #[arg(short, long)]
    output: PathBuf,
    /// Pooling statistic.
    ///
    /// `mean` matches `responsemean`'s default: each input is first rescaled by
    /// the geometric-mean multiplier that brings its l=0 terms toward the group
    /// mean l=0 (compensating per-scan magnitude — PD·T2 contrast and anchor
    /// ratio — that common-reference responses genuinely carry), then averaged.
    ///
    /// `median` has no mrtrix3 counterpart. It resists a single scan with a bad
    /// single-fiber mask, at the cost of discarding information when inputs are
    /// few. The same rescaling is applied first.
    #[arg(long, value_enum, default_value_t = Statistic::Mean)]
    statistic: Statistic,
    /// Skip the per-input magnitude rescaling and pool coefficients directly,
    /// like `responsemean -legacy`.
    #[arg(long)]
    legacy: bool,
    /// Overwrite the output directory if it already holds a response.
    #[arg(long)]
    overwrite: bool,
    /// Suppress the per-tissue spread report.
    #[arg(long)]
    quiet: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Statistic {
    Mean,
    Median,
}

impl From<Statistic> for PoolStatistic {
    fn from(s: Statistic) -> Self {
        match s {
            Statistic::Mean => Self::Mean,
            Statistic::Median => Self::Median,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Mean(args) => run_mean(args),
    }
}

fn run_mean(args: MeanArgs) -> Result<()> {
    if args.inputs.len() < 2 {
        bail!(
            "pooling needs at least two response directories; got {}",
            args.inputs.len()
        );
    }
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("create output directory {:?}", args.output))?;

    for tissue in TISSUES {
        let responses = load_tissue(&args.inputs, tissue)?;
        check_compatible(&responses, &args.inputs, tissue)?;
        let mults = magnitude_multipliers(&responses, args.legacy);
        let pooled = pool_tissue(&responses, &mults, args.statistic.into());

        let out = args.output.join(format!("{tissue}_response_denseb.txt"));
        if out.exists() && !args.overwrite {
            bail!("{:?} exists; pass --overwrite to replace it", out);
        }
        write_response_txt(&pooled, &out)
            .with_context(|| format!("write pooled {tissue} response to {:?}", out))?;

        if !args.quiet {
            report(tissue, &responses, &mults, &pooled);
        }
    }
    if !args.quiet {
        eprintln!(
            "[consh-response] pooled {} responses ({:?}) -> {}",
            args.inputs.len(),
            args.statistic,
            args.output.display()
        );
    }
    Ok(())
}

fn load_tissue(inputs: &[PathBuf], tissue: &str) -> Result<Vec<TissueResponse>> {
    inputs
        .iter()
        .map(|dir| {
            let p = dir.join(format!("{tissue}_response_denseb.txt"));
            TissueResponse::parse_mrtrix_txt(&p)
                .with_context(|| format!("read {tissue} response from {:?}", p))
        })
        .collect()
}

/// Reject inputs that cannot be pooled, naming each input's value.
fn check_compatible(responses: &[TissueResponse], inputs: &[PathBuf], tissue: &str) -> Result<()> {
    if let Err(problem) = check_poolable(responses) {
        bail!(
            "{tissue}: {problem}.\n{}",
            describe(responses, inputs, |r| format!(
                "rows {}, lmax {}, b=0 l=0 term {:.9}",
                r.n_shells(),
                r.lmax,
                r.coeffs.first().and_then(|x| x.first()).copied().unwrap_or(0.0)
            ))
        );
    }
    Ok(())
}

fn describe(
    responses: &[TissueResponse],
    inputs: &[PathBuf],
    f: impl Fn(&TissueResponse) -> String,
) -> String {
    responses
        .iter()
        .zip(inputs)
        .map(|(r, p)| format!("  {:<50} {}", short(p), f(r)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn short(p: &Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |s| s.to_string_lossy().into())
}

/// Report the magnitude multipliers applied and how far the (rescaled) inputs
/// sit from the pooled kernel.
///
/// The deviation is the diagnostic that matters when deciding whether one
/// shared kernel is defensible: a scan far from the pool will have its fiber
/// densities biased relative to the rest of the study. It is computed after
/// rescaling, so it reports residual *shape* spread; the multiplier range
/// reports the magnitude spread that was compensated.
fn report(tissue: &str, responses: &[TissueResponse], mults: &[f64], pooled: &TissueResponse) {
    let devs = deviations_pct(responses, mults, pooled);
    if devs.is_empty() {
        return;
    }
    let mean = devs.iter().sum::<f64>() / devs.len() as f64;
    let max = devs.iter().cloned().fold(f64::MIN, f64::max);
    let (m_lo, m_hi) = mults
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &m| (lo.min(m), hi.max(m)));
    eprintln!(
        "[consh-response] {tissue}: {} rows, lmax {}, magnitude multipliers {:.3}–{:.3}, \
         r_0 deviation from pool (rescaled): mean {:.2}%  max {:.2}%",
        pooled.n_shells(),
        pooled.lmax,
        m_lo,
        m_hi,
        mean,
        max
    );
}
