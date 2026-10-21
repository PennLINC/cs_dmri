// SPDX-License-Identifier: MIT OR Apache-2.0
//! `consh-response`: utilities for CONSH dense-b response functions.
//!
//! One subcommand, `mean`, pools per-scan responses into the common response
//! a quantitative study needs.
//!
//! This is the CONSH analogue of mrtrix3's `responsemean`. The motivation is
//! the same: fixel-based analysis and AFD quantification require that "one unit
//! of AFD" mean the same thing in every subject, which in turn requires a single
//! response shared across the study (Raffelt et al. 2012; Dhollander et al.
//! 2021; Smith et al. 2022). The difference is the format — a CONSH response is
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
//! CONSH responses carry that same per-scan magnitude, minus the gain: every
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

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};

use cs_dmri::multitissue::response::TissueResponse;
use cs_dmri::multitissue::response_estimation::write_response_txt;

const TISSUES: [&str; 3] = ["wm", "gm", "csf"];

#[derive(Parser, Debug)]
#[command(version, about = "Utilities for CONSH dense-b response functions")]
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
        let mults = multipliers(&responses, args.legacy);
        let pooled = pool(&responses, &mults, args.statistic);

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

/// Reject inputs that cannot be meaningfully averaged.
///
/// The dense-b grid is a function of the scan's own bmax and step, so responses
/// from different acquisitions have different row counts. Averaging across them
/// — or worse, silently truncating — would put fiber densities on incomparable
/// scales, so this is a hard error.
fn check_compatible(
    responses: &[TissueResponse],
    inputs: &[PathBuf],
    tissue: &str,
) -> Result<()> {
    let rows: BTreeSet<usize> = responses.iter().map(|r| r.n_shells()).collect();
    if rows.len() != 1 {
        bail!(
            "{tissue}: inputs have different dense-b grids (row counts {:?}). \
             All scans must share bmax and --step; see the G6 note in \
             docs/quantitative-consh.md.\n{}",
            rows,
            describe(responses, inputs, |r| r.n_shells().to_string())
        );
    }
    let lmax: BTreeSet<usize> = responses.iter().map(|r| r.lmax).collect();
    if lmax.len() != 1 {
        bail!(
            "{tissue}: inputs have different lmax {:?}; refit with a common --lmax-wm.\n{}",
            lmax,
            describe(responses, inputs, |r| r.lmax.to_string())
        );
    }
    let ncols: BTreeSet<usize> = responses
        .iter()
        .map(|r| r.coeffs.first().map_or(0, |row| row.len()))
        .collect();
    if ncols.len() != 1 {
        bail!("{tissue}: inputs have different coefficient counts {:?}", ncols);
    }
    // The b=0 magnitudes legitimately differ across scans (per-scan PD·T2
    // contrast and mean/median anchor ratio — that is what the multiplier
    // compensates), but each must at least be positive: a zero or negative b=0
    // row means the input was not produced by this pipeline (or was edited).
    if responses
        .iter()
        .any(|r| r.coeffs.first().and_then(|row| row.first()).copied().unwrap_or(0.0) <= 0.0)
    {
        bail!(
            "{tissue}: every input needs a positive b=0 l=0 term.\n{}",
            describe(responses, inputs, |r| format!(
                "{:.9}",
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

/// Per-input magnitude multipliers, exactly as `responsemean` computes them:
/// the geometric mean over rows of `(group arithmetic-mean l=0) / (input l=0)`.
///
/// Rows where either term is non-positive are excluded from both the sum and
/// the divisor (see the module docs); when every row is positive — the only
/// case mrtrix3 defines — this is identical to `responsemean`. `legacy` returns
/// all-ones, matching `responsemean -legacy`.
fn multipliers(responses: &[TissueResponse], legacy: bool) -> Vec<f64> {
    if legacy {
        return vec![1.0; responses.len()];
    }
    let nrows = responses[0].n_shells();
    let mean_lzero: Vec<f64> = (0..nrows)
        .map(|i| responses.iter().map(|r| r.coeffs[i][0]).sum::<f64>() / responses.len() as f64)
        .collect();
    responses
        .iter()
        .map(|r| {
            let mut log_sum = 0.0_f64;
            let mut n = 0usize;
            for (i, &m) in mean_lzero.iter().enumerate() {
                let s = r.coeffs[i][0];
                if m > 0.0 && s > 0.0 {
                    log_sum += (m / s).ln();
                    n += 1;
                }
            }
            if n > 0 { (log_sum / n as f64).exp() } else { 1.0 }
        })
        .collect()
}

fn pool(responses: &[TissueResponse], mults: &[f64], stat: Statistic) -> TissueResponse {
    let nrows = responses[0].n_shells();
    let ncols = responses[0].coeffs.first().map_or(0, |r| r.len());
    let mut coeffs = vec![vec![0.0f64; ncols]; nrows];
    let mut buf: Vec<f64> = Vec::with_capacity(responses.len());
    for (i, row) in coeffs.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            buf.clear();
            buf.extend(responses.iter().zip(mults).map(|(r, m)| m * r.coeffs[i][j]));
            *cell = match stat {
                Statistic::Mean => buf.iter().sum::<f64>() / buf.len() as f64,
                Statistic::Median => {
                    let mid = buf.len() / 2;
                    buf.select_nth_unstable_by(mid, f64::total_cmp);
                    if buf.len() % 2 == 1 {
                        buf[mid]
                    } else {
                        // Even count: average the two central order statistics,
                        // matching the usual definition.
                        let hi = buf[mid];
                        let lo = buf[..mid].iter().copied().fold(f64::MIN, f64::max);
                        0.5 * (lo + hi)
                    }
                }
            };
        }
    }
    TissueResponse { coeffs, lmax: responses[0].lmax }
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
    let r0: Vec<f64> = pooled.coeffs.iter().map(|r| r[0]).collect();
    let nz: Vec<usize> = (0..r0.len()).filter(|&i| r0[i].abs() > 1e-9).collect();
    if nz.is_empty() {
        return;
    }
    let devs: Vec<f64> = responses
        .iter()
        .zip(mults)
        .map(|(resp, m)| {
            nz.iter()
                .map(|&i| ((m * resp.coeffs[i][0] - r0[i]) / r0[i]).abs())
                .sum::<f64>()
                / nz.len() as f64
                * 100.0
        })
        .collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn resp(l0_rows: &[f64]) -> TissueResponse {
        TissueResponse { coeffs: l0_rows.iter().map(|&v| vec![v]).collect(), lmax: 0 }
    }

    #[test]
    fn multipliers_match_responsemean_definition() {
        // Two inputs with joint magnitude+shape differences; the multiplier is
        // the geometric mean over rows of (group-mean l=0) / (input l=0),
        // exactly as mrtrix3's responsemean computes it.
        let a = resp(&[1.0, 0.5]);
        let b = resp(&[4.0, 1.0]);
        let m = multipliers(&[a, b], false);
        let mean_l0 = [2.5_f64, 0.75];
        let expect_a = ((mean_l0[0] / 1.0).ln() / 2.0 + (mean_l0[1] / 0.5).ln() / 2.0).exp();
        let expect_b = ((mean_l0[0] / 4.0).ln() / 2.0 + (mean_l0[1] / 1.0).ln() / 2.0).exp();
        assert_abs_diff_eq!(m[0], expect_a, epsilon = 1e-12);
        assert_abs_diff_eq!(m[1], expect_b, epsilon = 1e-12);

        // And the pooled mean is the average of the rescaled inputs.
        let pooled = pool(&[resp(&[1.0, 0.5]), resp(&[4.0, 1.0])], &m, Statistic::Mean);
        assert_abs_diff_eq!(
            pooled.coeffs[0][0],
            (expect_a * 1.0 + expect_b * 4.0) / 2.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            pooled.coeffs[1][0],
            (expect_a * 0.5 + expect_b * 1.0) / 2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn pure_magnitude_difference_is_fully_compensated() {
        // Input B is 2× input A (same shape). The rescaled inputs must agree
        // exactly, so the pooled kernel equals both — the property that keeps a
        // high-PD·T2 scan from dominating the group kernel.
        let a = resp(&[2.0, 1.0, 0.25]);
        let b = resp(&[4.0, 2.0, 0.5]);
        let m = multipliers(&[a, b], false);
        for i in 0..3 {
            let ra = m[0] * [2.0, 1.0, 0.25][i];
            let rb = m[1] * [4.0, 2.0, 0.5][i];
            assert_abs_diff_eq!(ra, rb, epsilon = 1e-12);
        }
    }

    #[test]
    fn legacy_is_all_ones() {
        let m = multipliers(&[resp(&[1.0]), resp(&[4.0])], true);
        assert_eq!(m, vec![1.0, 1.0]);
    }

    #[test]
    fn nonpositive_rows_are_excluded_not_propagated() {
        // A dense-b grid can carry rows fitted to ~0 signal; those rows must
        // not poison the multiplier with ln(0) or ln(negative).
        let a = resp(&[1.0, 0.0, 0.5]);
        let b = resp(&[2.0, 0.0, 1.0]);
        let m = multipliers(&[a, b], false);
        assert!(m.iter().all(|v| v.is_finite() && *v > 0.0));
        // Rows 0 and 2 are pure ×2 magnitude, so compensation is exact.
        assert_abs_diff_eq!(m[0] * 1.0, m[1] * 2.0, epsilon = 1e-12);
    }
}
