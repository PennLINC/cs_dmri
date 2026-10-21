// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pooling of per-scan dense-b responses into one group response.
//!
//! The analogue of MRtrix3's `responsemean` for CONSH responses, which have
//! one zonal-SH row per dense-b grid point rather than one per shell. As in
//! `responsemean`'s default, each input is first rescaled by a magnitude
//! multiplier, the geometric mean over rows of `(group mean l=0) / (input
//! l=0)`, which compensates per-scan magnitude differences (PD·T2 contrast and
//! the ratio of the reference b=0 to the voxels' mean) before pooling. Rows
//! with a non-positive l=0 term, in the input or the group mean, are left out
//! of the multiplier; when every row is positive, the only case `responsemean`
//! defines, the two agree. `legacy` pools without rescaling, like
//! `responsemean -legacy`.
//!
//! The command-line form is `consh-response mean`.

use std::collections::BTreeSet;

use crate::multitissue::consh::ConshResponses;
use crate::multitissue::response::TissueResponse;
use crate::{CsDmriError, Result};

/// Pooling statistic for [`pool_responses`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolStatistic {
    /// Arithmetic mean of the rescaled inputs (`responsemean`'s default).
    #[default]
    Mean,
    /// Median of the rescaled inputs. Resists a single scan with a poor
    /// single-fiber selection, at the cost of efficiency with few inputs.
    Median,
}

/// One tissue's pooled response and how the inputs related to it.
#[derive(Debug, Clone)]
pub struct PooledTissue {
    pub response: TissueResponse,
    /// Magnitude multiplier applied to each input (all 1 with `legacy`).
    pub multipliers: Vec<f64>,
    /// Mean absolute deviation of each rescaled input's l=0 terms from the
    /// pooled response, in percent: the residual spread in response shape.
    pub deviation_pct: Vec<f64>,
}

/// Pool per-scan CONSH responses (two or more, on the same dense-b grid and
/// with the same WM order) into one group response.
pub fn pool_responses(
    inputs: &[ConshResponses],
    statistic: PoolStatistic,
    legacy: bool,
) -> Result<(ConshResponses, [PooledTissue; 3])> {
    if inputs.len() < 2 {
        return Err(CsDmriError::Other(format!(
            "pooling needs at least two responses; got {}",
            inputs.len()
        )));
    }
    let tissue = |name: &str, get: fn(&ConshResponses) -> &TissueResponse| -> Result<PooledTissue> {
        let responses: Vec<TissueResponse> = inputs.iter().map(|r| get(r).clone()).collect();
        check_poolable(&responses).map_err(|e| CsDmriError::Other(format!("{name}: {e}")))?;
        let multipliers = magnitude_multipliers(&responses, legacy);
        let response = pool_tissue(&responses, &multipliers, statistic);
        let deviation_pct = deviations_pct(&responses, &multipliers, &response);
        Ok(PooledTissue { response, multipliers, deviation_pct })
    };
    let wm = tissue("WM", |r| &r.wm)?;
    let gm = tissue("GM", |r| &r.gm)?;
    let csf = tissue("CSF", |r| &r.csf)?;
    let pooled = ConshResponses {
        wm: wm.response.clone(),
        gm: gm.response.clone(),
        csf: csf.response.clone(),
    };
    Ok((pooled, [wm, gm, csf]))
}

/// Why one tissue's responses cannot be pooled, if they cannot.
///
/// The dense-b grid follows each scan's maximum b and grid step, so responses
/// from different acquisitions have different row counts; averaging or
/// truncating them would put fiber densities on incomparable scales.
pub fn check_poolable(responses: &[TissueResponse]) -> std::result::Result<(), String> {
    let rows: BTreeSet<usize> = responses.iter().map(|r| r.n_shells()).collect();
    if rows.len() != 1 {
        return Err(format!(
            "inputs have different dense-b grids (row counts {rows:?}); all scans must share \
             the maximum b-value and grid step"
        ));
    }
    let lmax: BTreeSet<usize> = responses.iter().map(|r| r.lmax).collect();
    if lmax.len() != 1 {
        return Err(format!("inputs have different SH orders {lmax:?}; refit with a common order"));
    }
    let ncols: BTreeSet<usize> =
        responses.iter().map(|r| r.coeffs.first().map_or(0, |row| row.len())).collect();
    if ncols.len() != 1 {
        return Err(format!("inputs have different coefficient counts {ncols:?}"));
    }
    // b=0 magnitudes differ between scans (that is what the multiplier
    // compensates), but a zero or negative b=0 term means the input did not
    // come from CONSH response estimation.
    if responses
        .iter()
        .any(|r| r.coeffs.first().and_then(|row| row.first()).copied().unwrap_or(0.0) <= 0.0)
    {
        return Err("every input needs a positive b=0 l=0 term".to_string());
    }
    Ok(())
}

/// Per-input magnitude multipliers: the geometric mean over rows of
/// `(group arithmetic-mean l=0) / (input l=0)`, over rows where both are
/// positive. All ones with `legacy`.
pub fn magnitude_multipliers(responses: &[TissueResponse], legacy: bool) -> Vec<f64> {
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

/// Pool coefficient by coefficient after rescaling each input by its multiplier.
pub fn pool_tissue(responses: &[TissueResponse], mults: &[f64], stat: PoolStatistic) -> TissueResponse {
    let nrows = responses[0].n_shells();
    let ncols = responses[0].coeffs.first().map_or(0, |r| r.len());
    let mut coeffs = vec![vec![0.0f64; ncols]; nrows];
    let mut buf: Vec<f64> = Vec::with_capacity(responses.len());
    for (i, row) in coeffs.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            buf.clear();
            buf.extend(responses.iter().zip(mults).map(|(r, m)| m * r.coeffs[i][j]));
            *cell = match stat {
                PoolStatistic::Mean => buf.iter().sum::<f64>() / buf.len() as f64,
                PoolStatistic::Median => {
                    let mid = buf.len() / 2;
                    buf.select_nth_unstable_by(mid, f64::total_cmp);
                    if buf.len() % 2 == 1 {
                        buf[mid]
                    } else {
                        // Even count: average of the two central order statistics.
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

/// Mean absolute deviation (percent) of each rescaled input's l=0 terms from
/// the pooled response, over rows where the pooled term is not zero. A scan far
/// from the pool would have its fiber densities biased relative to the rest of
/// the study.
pub fn deviations_pct(responses: &[TissueResponse], mults: &[f64], pooled: &TissueResponse) -> Vec<f64> {
    let r0: Vec<f64> = pooled.coeffs.iter().map(|r| r[0]).collect();
    let nz: Vec<usize> = (0..r0.len()).filter(|&i| r0[i].abs() > 1e-9).collect();
    responses
        .iter()
        .zip(mults)
        .map(|(resp, m)| {
            if nz.is_empty() {
                return 0.0;
            }
            nz.iter().map(|&i| ((m * resp.coeffs[i][0] - r0[i]) / r0[i]).abs()).sum::<f64>()
                / nz.len() as f64
                * 100.0
        })
        .collect()
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
        let m = magnitude_multipliers(&[a, b], false);
        let mean_l0 = [2.5_f64, 0.75];
        let expect_a = ((mean_l0[0] / 1.0).ln() / 2.0 + (mean_l0[1] / 0.5).ln() / 2.0).exp();
        let expect_b = ((mean_l0[0] / 4.0).ln() / 2.0 + (mean_l0[1] / 1.0).ln() / 2.0).exp();
        assert_abs_diff_eq!(m[0], expect_a, epsilon = 1e-12);
        assert_abs_diff_eq!(m[1], expect_b, epsilon = 1e-12);

        // And the pooled mean is the average of the rescaled inputs.
        let pooled = pool_tissue(&[resp(&[1.0, 0.5]), resp(&[4.0, 1.0])], &m, PoolStatistic::Mean);
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
        let m = magnitude_multipliers(&[a, b], false);
        for i in 0..3 {
            let ra = m[0] * [2.0, 1.0, 0.25][i];
            let rb = m[1] * [4.0, 2.0, 0.5][i];
            assert_abs_diff_eq!(ra, rb, epsilon = 1e-12);
        }
    }

    #[test]
    fn legacy_is_all_ones() {
        let m = magnitude_multipliers(&[resp(&[1.0]), resp(&[4.0])], true);
        assert_eq!(m, vec![1.0, 1.0]);
    }

    #[test]
    fn nonpositive_rows_are_excluded_not_propagated() {
        // A dense-b grid can carry rows fitted to ~0 signal; those rows must
        // not poison the multiplier with ln(0) or ln(negative).
        let a = resp(&[1.0, 0.0, 0.5]);
        let b = resp(&[2.0, 0.0, 1.0]);
        let m = magnitude_multipliers(&[a, b], false);
        assert!(m.iter().all(|v| v.is_finite() && *v > 0.0));
        // Rows 0 and 2 are pure ×2 magnitude, so compensation is exact.
        assert_abs_diff_eq!(m[0] * 1.0, m[1] * 2.0, epsilon = 1e-12);
    }
}
