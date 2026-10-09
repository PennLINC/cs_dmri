// SPDX-License-Identifier: MPL-2.0
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file is a Rust port of the algorithm in MRtrix3's `mtnormalise`
 * (cpp/cmd/mtnormalise.cpp): the log-domain polynomial field fit, the
 * per-tissue balance factors, and the IQR outlier-rejection schedule.
 *
 * Copyright (c) 2008-2026 the MRtrix3 contributors (original work).
 * Copyright (c) 2026 the PennLINC developers team (port).
 *
 * Covered Software is provided under this License on an "as is" basis,
 * without warranty of any kind. See the Mozilla Public License v. 2.0.
 *
 * MPL-2.0 is a file-scoped copyleft: this file stays MPL-2.0 and its source
 * must remain available; the files that merely call into it do not become
 * MPL-covered. Do not copy chunks of this file into other modules.
 */

//! Multi-tissue log-domain intensity normalization — a Rust port of the
//! algorithm behind MRtrix3's `mtnormalise`, used by `cs-ss3t-full` and
//! `cs-mtnorm`.
//!
//! References:
//! - Raffelt, Dhollander, Tournier, Tabbara, Smith, Pierre & Connelly,
//!   *"Bias field correction and intensity normalisation for quantitative
//!   analysis of apparent fibre density"*, ISMRM 2017, 3541.
//! - Dhollander, Tabbara, Rosnarho-Tornstrand, Tournier, Raffelt & Connelly,
//!   *"Multi-tissue log-domain intensity and inhomogeneity normalisation for
//!   quantitative apparent fibre density"*, ISMRM 2021, 2472.
//!
//! ## Algorithm
//!
//! Per masked voxel `v`, let `x_t(v) ≥ 0` be the l=0 value of tissue `t`
//! (negative inputs are clamped to zero for estimation). The algorithm
//! estimates a smooth multiplicative field `f(v) = exp(polynomial(v))` and
//! per-tissue *balance factors* `b_t` such that the balanced tissue sum
//! matches a constant reference `T`:
//!
//! ```text
//!   Σ_t b_t · x_t(v) ≈ T · f(v)      over inlier voxels
//! ```
//!
//! Iteration structure (matching MRtrix3's `mtnormalise`):
//!
//! 1. Initial outlier pass with IQR range 3.0 on `log(Σ_t b_t x_t / f)`.
//! 2. `niter` (default 15) main iterations, each consisting of:
//!    - an inner loop (up to `balance_maxiter`, default 7): solve the
//!      balance factors by least squares — `min Σ_v w_v (Σ_t b_t x_t/f − 1)²`
//!      — then rescale so the geometric mean of `b` is 1, then re-detect
//!      outliers with IQR range 1.5; repeat while the inlier set changes;
//!    - a weighted least-squares fit of the polynomial log-field to
//!      `log(Σ_t b_t x_t) − log(T)` over the inlier voxels.
//! 3. Apply: `output_t(v) = B_t · x_t(v) / f(v)` over the *whole volume*,
//!    where `B_t = b_t` if `apply_balance` (MRtrix's `-balanced`) is set and
//!    `B_t = 1` otherwise (the MRtrix default: balance factors steer the
//!    field fit but are not baked into the output).
//!
//! The polynomial is evaluated on centered, scaled voxel coordinates rather
//! than scanner coordinates; the total-degree-`order` polynomial space is
//! closed under affine coordinate changes, so the fitted field is the same
//! while the normal equations stay well-conditioned.
//!
//! ## License posture
//!
//! Ported from MRtrix3's `cpp/cmd/mtnormalise.cpp` (MPL-2.0), which was
//! followed closely enough for numerical interchangeability (2026-08) that
//! this file is treated as a derivative and carries the MPL-2.0 notice above.

use std::f64::consts::PI;

use nalgebra::{Cholesky, DMatrix, DVector};
use ndarray::{Array3, Array4};

use crate::{CsDmriError, Result};

/// Tunable knobs for [`mtnormalise`].
#[derive(Debug, Clone, Copy)]
pub struct MtnormaliseConfig {
    /// Polynomial order for the spatial bias field. Default 3 (= 20
    /// monomials), matching MRtrix3.
    pub poly_order: usize,
    /// Reference value `T` the balanced tissue sum is normalized toward.
    /// The default (`None`) uses the median observed sum so the input's
    /// global scale is preserved. Set `Some(1.0/sqrt(4π))` (see
    /// [`target_sum_mrtrix_default`]) to match the MRtrix3 convention.
    pub target_sum: Option<f64>,
    /// Number of main iterations (field updates). MRtrix3 default: 15.
    pub niter: usize,
    /// Maximum iterations of the inner balance-factor / outlier-rejection
    /// loop per main iteration. MRtrix3 default: 7.
    pub balance_maxiter: usize,
    /// Multiply the *output* tissues by their balance factors, like
    /// MRtrix3's `-balanced` flag. The balance factors always steer the
    /// field estimation; this only controls whether they are applied to the
    /// output. MRtrix warns this has critical consequences for AFD
    /// normalization. Default `false`.
    pub apply_balance: bool,
}

impl Default for MtnormaliseConfig {
    fn default() -> Self {
        Self {
            poly_order: 3,
            target_sum: None,
            niter: 15,
            balance_maxiter: 7,
            apply_balance: false,
        }
    }
}

/// Diagnostics returned by [`mtnormalise`].
#[derive(Debug, Clone)]
pub struct MtnormaliseDiagnostics {
    /// The recovered field `f(v) = exp(polynomial(v))`, full volume.
    pub bias_field: Array3<f32>,
    /// Polynomial coefficients of `log f` in [`eval_poly_basis`] order at
    /// the final iteration.
    pub poly_coefs: Vec<f64>,
    /// Number of inlier voxels (weight 1) after the final outlier pass.
    pub n_fit_voxels: usize,
    /// Reference value used (resolved from `cfg.target_sum` or the median).
    pub target_sum_used: f64,
    /// Mean `|log(Σ_t b_t x_t / f) − log T|` over the final inlier voxels.
    pub mean_abs_log_residual: f64,
    /// Final per-tissue balance factors `[b_WM, b_GM, b_CSF]` (geometric
    /// mean 1). Applied to the output only when `cfg.apply_balance` is set.
    pub tissue_scales: [f64; 3],
    /// Number of main iterations run.
    pub iterations: usize,
    /// `exp(mean(log f))` over the final inlier voxels — MRtrix3's
    /// `lognorm_scale` header entry.
    pub lognorm_scale: f64,
}

const N_TISSUES: usize = 3;

/// Number of polynomial monomials for a 3D polynomial of given order.
/// `(o+1)(o+2)(o+3)/6`: 1, 4, 10, 20, 35, 56, … for orders 0, 1, 2, 3, 4, 5.
pub fn poly_basis_size(order: usize) -> usize {
    (order + 1) * (order + 2) * (order + 3) / 6
}

/// Evaluate every monomial `x^i y^j z^k` with `i+j+k ≤ order` at the given
/// point. Lexicographic order over `(i, j, k)`.
pub fn eval_poly_basis(coords: [f64; 3], order: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), poly_basis_size(order));
    let (x, y, z) = (coords[0], coords[1], coords[2]);
    let mut idx = 0;
    for i in 0..=order {
        let xi = x.powi(i as i32);
        for j in 0..=(order - i) {
            let yj = y.powi(j as i32);
            for k in 0..=(order - i - j) {
                let zk = z.powi(k as i32);
                out[idx] = xi * yj * zk;
                idx += 1;
            }
        }
    }
}

/// Total order over f64 that sorts non-finite values (NaN, -inf) first, so
/// they always land below the lower quartile like MRtrix's NaN-first
/// comparator.
fn cmp_nan_first(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.partial_cmp(b).unwrap(),
    }
}

/// Recompute the 0/1 inlier weights from the IQR rule on
/// `log(Σ_t b_t x_t / f)`. Quartiles are taken over *all* masked voxels
/// (current weights do not gate the quartile estimate). Returns the number
/// of weights that changed.
fn detect_outliers(
    outlier_range: f64,
    data: &[[f64; N_TISSUES]],
    field: &[f64],
    balance: &[f64; N_TISSUES],
    weights: &mut [f64],
) -> usize {
    let n = data.len();
    let summed_log: Vec<f64> = (0..n)
        .map(|v| {
            let s = balance[0] * data[v][0] + balance[1] * data[v][1] + balance[2] * data[v][2];
            (s / field[v]).ln()
        })
        .collect();

    let lower_idx = ((n as f64 * 0.25).round() as usize).min(n - 1);
    let upper_idx = ((n as f64 * 0.75).round() as usize).min(n - 1);
    let mut sorted = summed_log.clone();
    let (_, lq, _) = sorted.select_nth_unstable_by(lower_idx, cmp_nan_first);
    let lower_quartile = *lq;
    let (_, uq, _) = sorted.select_nth_unstable_by(upper_idx, cmp_nan_first);
    let upper_quartile = *uq;

    let iqr = upper_quartile - lower_quartile;
    let lower_threshold = lower_quartile - outlier_range * iqr;
    let upper_threshold = upper_quartile + outlier_range * iqr;

    let mut changed = 0usize;
    for (v, w) in weights.iter_mut().enumerate() {
        let sl = summed_log[v];
        let new_w = if sl.is_finite() && sl >= lower_threshold && sl <= upper_threshold {
            1.0
        } else {
            0.0
        };
        if new_w != *w {
            changed += 1;
        }
        *w = new_w;
    }
    changed
}

/// Solve `min_b Σ_{v: w_v=1} (Σ_t b_t x_t(v)/f(v) − 1)²`, then rescale so
/// the geometric mean of `b` is 1.
fn compute_balance_factors(
    data: &[[f64; N_TISSUES]],
    field: &[f64],
    weights: &[f64],
    balance: &mut [f64; N_TISSUES],
) -> Result<()> {
    let mut hth = DMatrix::<f64>::zeros(N_TISSUES, N_TISSUES);
    let mut rhs = DVector::<f64>::zeros(N_TISSUES);
    for v in 0..data.len() {
        if weights[v] == 0.0 {
            continue;
        }
        let s = [
            data[v][0] / field[v],
            data[v][1] / field[v],
            data[v][2] / field[v],
        ];
        for i in 0..N_TISSUES {
            rhs[i] += s[i];
            for j in i..N_TISSUES {
                hth[(i, j)] += s[i] * s[j];
                if i != j {
                    hth[(j, i)] += s[i] * s[j];
                }
            }
        }
    }
    // Tiny ridge keeps the solve defined when tissue columns are collinear
    // (e.g. perfectly proportional synthetic data); negligible otherwise.
    let max_diag = (0..N_TISSUES).map(|i| hth[(i, i)]).fold(0.0_f64, f64::max);
    let eps = max_diag.max(1.0) * 1e-12;
    for i in 0..N_TISSUES {
        hth[(i, i)] += eps;
    }
    let sol = Cholesky::new(hth)
        .ok_or_else(|| {
            CsDmriError::Other("mtnormalise: balance-factor normal equations not PD".into())
        })?
        .solve(&rhs);

    let b = [sol[0], sol[1], sol[2]];
    if b.iter().any(|&x| !x.is_finite() || x <= 0.0) {
        return Err(CsDmriError::Other(format!(
            "mtnormalise: non-positive tissue balance factor computed: {:?}",
            b
        )));
    }
    let log_mean = (b[0].ln() + b[1].ln() + b[2].ln()) / N_TISSUES as f64;
    let geo_mean = log_mean.exp();
    for (dst, &src) in balance.iter_mut().zip(b.iter()) {
        *dst = src / geo_mean;
    }
    Ok(())
}

/// Weighted LS fit of the polynomial log-field to
/// `log(Σ_t b_t x_t) − log T` over inlier voxels; refresh `field` at every
/// masked voxel from the new coefficients.
fn update_field(
    log_ref: f64,
    basis: &DMatrix<f64>,
    data: &[[f64; N_TISSUES]],
    balance: &[f64; N_TISSUES],
    weights: &[f64],
    field_coeffs: &mut DVector<f64>,
    field: &mut [f64],
) -> Result<()> {
    let n_basis = basis.ncols();
    let mut hth = DMatrix::<f64>::zeros(n_basis, n_basis);
    let mut rhs = DVector::<f64>::zeros(n_basis);
    for v in 0..data.len() {
        let w = weights[v];
        if w == 0.0 {
            continue;
        }
        let s = balance[0] * data[v][0] + balance[1] * data[v][1] + balance[2] * data[v][2];
        let y = if s > 0.0 { w * (s.ln() - log_ref) } else { 0.0 };
        let row = basis.row(v);
        for i in 0..n_basis {
            rhs[i] += row[i] * y;
            for j in i..n_basis {
                let dot = row[i] * row[j];
                hth[(i, j)] += dot;
                if i != j {
                    hth[(j, i)] += dot;
                }
            }
        }
    }
    let max_diag = (0..n_basis).map(|i| hth[(i, i)]).fold(0.0_f64, f64::max);
    let eps = max_diag.max(1.0) * 1e-12;
    for i in 0..n_basis {
        hth[(i, i)] += eps;
    }
    *field_coeffs = Cholesky::new(hth)
        .ok_or_else(|| {
            CsDmriError::Other(
                "mtnormalise: field normal equations not PD — try a lower poly_order".into(),
            )
        })?
        .solve(&rhs);

    for v in 0..data.len() {
        let log_f: f64 = (0..n_basis).map(|i| basis[(v, i)] * field_coeffs[i]).sum();
        field[v] = log_f.exp();
    }
    Ok(())
}

/// Multi-tissue intensity normalization. Mutates the three tissue arrays
/// in-place: every SH coefficient is divided by `f(v)` (and multiplied by
/// `b_t` when `cfg.apply_balance` is set) over the whole volume.
///
/// `wm` is 4D with `n_sh` channels (typically 45 at lmax=8). `gm` and `csf`
/// are 4D with 1 channel each. `mask` is 3D and selects the voxels used for
/// estimation; the field is applied everywhere, matching MRtrix3.
pub fn mtnormalise(
    wm: &mut Array4<f32>,
    gm: &mut Array4<f32>,
    csf: &mut Array4<f32>,
    mask: &Array3<bool>,
    cfg: &MtnormaliseConfig,
) -> Result<MtnormaliseDiagnostics> {
    let s = mask.shape();
    let (nx, ny, nz) = (s[0], s[1], s[2]);
    let wm_shape = wm.shape().to_vec();
    if wm_shape[..3] != [nx, ny, nz] {
        return Err(CsDmriError::Dimension(format!(
            "mtnormalise: WM spatial dims {:?} ≠ mask {:?}",
            &wm_shape[..3],
            [nx, ny, nz]
        )));
    }
    if gm.shape()[..3] != [nx, ny, nz] || csf.shape()[..3] != [nx, ny, nz] {
        return Err(CsDmriError::Dimension(
            "mtnormalise: GM / CSF spatial dims must match mask".into(),
        ));
    }
    if cfg.niter == 0 || cfg.balance_maxiter == 0 {
        return Err(CsDmriError::Other(
            "mtnormalise: niter and balance_maxiter must be ≥ 1".into(),
        ));
    }

    // Coordinate centring + scaling so polynomial values stay O(1)
    // (equivalent fit to MRtrix's scanner-coordinate basis, better
    // conditioned).
    let cx = (nx as f64 - 1.0) * 0.5;
    let cy = (ny as f64 - 1.0) * 0.5;
    let cz = (nz as f64 - 1.0) * 0.5;
    let scale = ((nx + ny + nz) as f64 / 6.0).max(1.0);
    let n_basis = poly_basis_size(cfg.poly_order);

    // ---- Collect every masked voxel (negatives clamped to 0 for the fit) ----
    let mut coord_index: Vec<(usize, usize, usize)> = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if mask[(x, y, z)] {
                    coord_index.push((x, y, z));
                }
            }
        }
    }
    let num_voxels = coord_index.len();
    if num_voxels == 0 {
        return Err(CsDmriError::Other(
            "mtnormalise: mask contains no voxels".into(),
        ));
    }
    let clamp0 = |v: f32| -> f64 {
        let v = v as f64;
        if v.is_finite() && v > 0.0 { v } else { 0.0 }
    };
    let data: Vec<[f64; N_TISSUES]> = coord_index
        .iter()
        .map(|&(x, y, z)| {
            [
                clamp0(wm[(x, y, z, 0)]),
                clamp0(gm[(x, y, z, 0)]),
                clamp0(csf[(x, y, z, 0)]),
            ]
        })
        .collect();

    // Resolve reference value T (explicit, or median observed sum).
    let target_sum = if let Some(t) = cfg.target_sum {
        if t <= 0.0 {
            return Err(CsDmriError::Other(format!(
                "mtnormalise: target_sum must be positive, got {}",
                t
            )));
        }
        t
    } else {
        let mut sums: Vec<f64> = data
            .iter()
            .map(|d| d[0] + d[1] + d[2])
            .filter(|&s| s > 0.0)
            .collect();
        if sums.is_empty() {
            return Err(CsDmriError::Other(
                "mtnormalise: no masked voxels with positive l=0 sum".into(),
            ));
        }
        let mid = sums.len() / 2;
        let (_, med, _) = sums.select_nth_unstable_by(mid, cmp_nan_first);
        *med
    };
    let log_ref = target_sum.ln();

    // ---- Basis matrix over masked voxels ----
    let mut basis = DMatrix::<f64>::zeros(num_voxels, n_basis);
    let mut basis_buf = vec![0.0_f64; n_basis];
    for (v, &(x, y, z)) in coord_index.iter().enumerate() {
        let coords = [
            (x as f64 - cx) / scale,
            (y as f64 - cy) / scale,
            (z as f64 - cz) / scale,
        ];
        eval_poly_basis(coords, cfg.poly_order, &mut basis_buf);
        for i in 0..n_basis {
            basis[(v, i)] = basis_buf[i];
        }
    }

    // ---- Main iteration (MRtrix structure) ----
    let mut weights: Vec<f64> = data
        .iter()
        .map(|d| {
            let s = d[0] + d[1] + d[2];
            if s.is_finite() && s > 0.0 { 1.0 } else { 0.0 }
        })
        .collect();
    let mut field = vec![1.0_f64; num_voxels];
    let mut field_coeffs = DVector::<f64>::zeros(n_basis);
    let mut balance = [1.0_f64; N_TISSUES];

    detect_outliers(3.0, &data, &field, &balance, &mut weights);

    let mut iters_done = 0usize;
    for iter in 0..cfg.niter {
        iters_done = iter + 1;

        let mut balance_iter = 1usize;
        loop {
            compute_balance_factors(&data, &field, &weights, &mut balance)?;
            let changed = detect_outliers(1.5, &data, &field, &balance, &mut weights);
            if changed == 0 || balance_iter >= cfg.balance_maxiter {
                break;
            }
            balance_iter += 1;
        }

        update_field(
            log_ref,
            &basis,
            &data,
            &balance,
            &weights,
            &mut field_coeffs,
            &mut field,
        )?;
    }

    // ---- Full-volume field ----
    let mut bias_field = Array3::<f32>::zeros((nx, ny, nz));
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let coords = [
                    (x as f64 - cx) / scale,
                    (y as f64 - cy) / scale,
                    (z as f64 - cz) / scale,
                ];
                eval_poly_basis(coords, cfg.poly_order, &mut basis_buf);
                let log_f: f64 = (0..n_basis).map(|i| field_coeffs[i] * basis_buf[i]).sum();
                bias_field[(x, y, z)] = log_f.exp() as f32;
            }
        }
    }

    // ---- Diagnostics over final inliers ----
    let mut n_fit = 0usize;
    let mut sum_log_field = 0.0_f64;
    let mut sum_abs_residual = 0.0_f64;
    for v in 0..num_voxels {
        if weights[v] == 0.0 {
            continue;
        }
        n_fit += 1;
        sum_log_field += field[v].ln();
        let s = balance[0] * data[v][0] + balance[1] * data[v][1] + balance[2] * data[v][2];
        if s > 0.0 {
            sum_abs_residual += ((s / field[v]).ln() - log_ref).abs();
        }
    }
    let lognorm_scale = if n_fit > 0 {
        (sum_log_field / n_fit as f64).exp()
    } else {
        1.0
    };
    let mean_abs_log_residual = if n_fit > 0 {
        sum_abs_residual / n_fit as f64
    } else {
        0.0
    };

    // ---- Apply over the whole volume ----
    let out_balance = if cfg.apply_balance {
        balance
    } else {
        [1.0; N_TISSUES]
    };
    let n_sh_wm = wm_shape[3];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let f = bias_field[(x, y, z)] as f64;
                if f.is_nan() || f <= 1e-12 {
                    continue;
                }
                let scale_wm = (out_balance[0] / f) as f32;
                let scale_gm = (out_balance[1] / f) as f32;
                let scale_csf = (out_balance[2] / f) as f32;
                for k in 0..n_sh_wm {
                    wm[(x, y, z, k)] *= scale_wm;
                }
                gm[(x, y, z, 0)] *= scale_gm;
                csf[(x, y, z, 0)] *= scale_csf;
            }
        }
    }

    let poly_coefs: Vec<f64> = (0..n_basis).map(|i| field_coeffs[i]).collect();

    Ok(MtnormaliseDiagnostics {
        bias_field,
        poly_coefs,
        n_fit_voxels: n_fit,
        target_sum_used: target_sum,
        mean_abs_log_residual,
        tissue_scales: balance,
        iterations: iters_done,
        lognorm_scale,
    })
}

/// Convenience: `target_sum = 1/sqrt(4π) ≈ 0.282`, the MRtrix3 `mtnormalise`
/// default. Makes the per-voxel sum of l=0 *coefficients* across normalized
/// tissues equal to `1/sqrt(4π)` — equivalently, the sum of amplitudes
/// `Y₀₀ · c₀₀` per voxel equals `1/(4π)`.
pub fn target_sum_mrtrix_default() -> f64 {
    1.0 / (4.0 * PI).sqrt()
}

/// Convenience: `target_sum = sqrt(4π) ≈ 3.545`. Sum of amplitudes (rather
/// than coefficients) equals 1 per voxel.
pub fn target_sum_sqrt_4pi() -> f64 {
    (4.0 * PI).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn poly_basis_size_known() {
        assert_eq!(poly_basis_size(0), 1);
        assert_eq!(poly_basis_size(1), 4);
        assert_eq!(poly_basis_size(2), 10);
        assert_eq!(poly_basis_size(3), 20);
    }

    #[test]
    fn eval_poly_basis_constant_term_first() {
        let mut buf = vec![0.0; poly_basis_size(2)];
        eval_poly_basis([0.7, -0.3, 0.5], 2, &mut buf);
        // First entry is 1 (i=j=k=0).
        assert_abs_diff_eq!(buf[0], 1.0, epsilon = 1e-12);
        // Total must equal poly_basis_size(2) = 10.
        assert_eq!(buf.len(), 10);
        // None should be NaN/Inf.
        for &v in &buf {
            assert!(v.is_finite());
        }
    }

    /// Deterministic per-voxel pseudo-random in [0,1) — keeps tests
    /// reproducible without a RNG dependency.
    fn hash01(x: usize, y: usize, z: usize, salt: u64) -> f64 {
        let mut h = (x as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add((y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F))
            .wrapping_add((z as u64).wrapping_mul(0x165667B19E3779F9))
            .wrapping_add(salt.wrapping_mul(0x27D4EB2F165667C5));
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51AFD7ED558CCD);
        h ^= h >> 33;
        (h >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Build a synthetic volume with spatially varying tissue fractions
    /// whose per-voxel sum is exactly `sum_scale · f(v)` for a smooth bias
    /// `f`, and return (wm, gm, csf, mask, true_field).
    fn synth_volume(
        n: usize,
        sum_scale: f64,
    ) -> (
        Array4<f32>,
        Array4<f32>,
        Array4<f32>,
        Array3<bool>,
        Array3<f64>,
    ) {
        let mut wm = Array4::<f32>::zeros((n, n, n, 1));
        let mut gm = Array4::<f32>::zeros((n, n, n, 1));
        let mut csf = Array4::<f32>::zeros((n, n, n, 1));
        let mut true_field = Array3::<f64>::zeros((n, n, n));
        let c = (n as f64 - 1.0) / 2.0;
        let scale = n as f64 / 2.0;
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let xc = (x as f64 - c) / scale;
                    let yc = (y as f64 - c) / scale;
                    let f = (0.3 * xc - 0.2 * yc + 0.1 * xc * yc).exp();
                    // Fractions vary voxel-to-voxel, WM-dominant on average.
                    let a = 0.4 + 0.5 * hash01(x, y, z, 1);
                    let b = (1.0 - a) * hash01(x, y, z, 2);
                    let cfr = 1.0 - a - b;
                    wm[(x, y, z, 0)] = (a * sum_scale * f) as f32;
                    gm[(x, y, z, 0)] = (b * sum_scale * f) as f32;
                    csf[(x, y, z, 0)] = (cfr * sum_scale * f) as f32;
                    true_field[(x, y, z)] = f;
                }
            }
        }
        let mask = Array3::<bool>::from_elem((n, n, n), true);
        (wm, gm, csf, mask, true_field)
    }

    /// Exact-sum synthetic data: balance factors should stay ≈1, the bias
    /// field should be recovered, and every output voxel's tissue sum should
    /// pin to the reference.
    #[test]
    fn pins_tissue_sum_to_reference_and_removes_bias() {
        let n = 16;
        let target = 1.0;
        let (mut wm, mut gm, mut csf, mask, true_field) = synth_volume(n, 2.5);
        let cfg = MtnormaliseConfig {
            target_sum: Some(target),
            ..Default::default()
        };
        let diag = mtnormalise(&mut wm, &mut gm, &mut csf, &mask, &cfg).unwrap();

        // Balance factors ≈ 1 (the exact-sum construction is solved by b=1).
        for &b in &diag.tissue_scales {
            assert_abs_diff_eq!(b, 1.0, epsilon = 1e-6);
        }
        // Geometric mean pinned to 1.
        let geo: f64 =
            (diag.tissue_scales.iter().map(|b| b.ln()).sum::<f64>() / 3.0).exp();
        assert_abs_diff_eq!(geo, 1.0, epsilon = 1e-12);

        // Output tissue sums pinned to the reference everywhere.
        let mut max_dev = 0.0_f64;
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let s = wm[(x, y, z, 0)] as f64
                        + gm[(x, y, z, 0)] as f64
                        + csf[(x, y, z, 0)] as f64;
                    max_dev = max_dev.max((s - target).abs());
                }
            }
        }
        assert!(max_dev < 0.02, "max |sum - target| = {max_dev}");

        // Recovered field matches the true one up to the global 2.5 scale.
        let mut max_field_dev = 0.0_f64;
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let ratio = diag.bias_field[(x, y, z)] as f64
                        / (2.5 * true_field[(x, y, z)]);
                    max_field_dev = max_field_dev.max((ratio - 1.0).abs());
                }
            }
        }
        assert!(max_field_dev < 0.02, "max field ratio dev = {max_field_dev}");
        assert!(diag.mean_abs_log_residual < 1e-2);
    }

    /// A global per-tissue miscalibration (WM doubled) must be absorbed by
    /// the balance factors during fitting; with `apply_balance` the output
    /// sums pin to the reference again.
    #[test]
    fn balance_factors_absorb_tissue_miscalibration() {
        let n = 16;
        let target = 1.0;
        let (mut wm, mut gm, mut csf, mask, _tf) = synth_volume(n, 1.0);
        wm.mapv_inplace(|v| v * 2.0);
        let cfg = MtnormaliseConfig {
            target_sum: Some(target),
            apply_balance: true,
            ..Default::default()
        };
        let diag = mtnormalise(&mut wm, &mut gm, &mut csf, &mask, &cfg).unwrap();

        // b_WM should be about half of b_GM / b_CSF.
        assert!(
            (diag.tissue_scales[0] * 2.0 / diag.tissue_scales[1] - 1.0).abs() < 0.05,
            "balance factors {:?} did not absorb the ×2 WM scale",
            diag.tissue_scales
        );
        // With -balanced semantics the output sums re-pin to the target.
        let mut devs: Vec<f64> = Vec::new();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let s = wm[(x, y, z, 0)] as f64
                        + gm[(x, y, z, 0)] as f64
                        + csf[(x, y, z, 0)] as f64;
                    devs.push((s - target).abs());
                }
            }
        }
        devs.sort_by(cmp_nan_first);
        let median_dev = devs[devs.len() / 2];
        assert!(median_dev < 0.05, "median |sum - target| = {median_dev}");
    }

    /// Voxels with wildly inflated sums must be rejected as outliers and
    /// not drag the field fit.
    #[test]
    fn outlier_voxels_are_rejected() {
        let n = 16;
        let target = 1.0;
        let (mut wm, mut gm, mut csf, mask, _tf) = synth_volume(n, 1.0);
        // Corrupt a small cluster with 10x values.
        let n_corrupt = 20usize;
        for i in 0..n_corrupt {
            let x = 2 + (i % 4);
            let y = 3 + ((i / 4) % 4);
            let z = 4 + (i / 16);
            wm[(x, y, z, 0)] *= 10.0;
            gm[(x, y, z, 0)] *= 10.0;
            csf[(x, y, z, 0)] *= 10.0;
        }
        let cfg = MtnormaliseConfig {
            target_sum: Some(target),
            ..Default::default()
        };
        let diag = mtnormalise(&mut wm, &mut gm, &mut csf, &mask, &cfg).unwrap();
        // The corrupted voxels are excluded from the fit.
        assert!(
            diag.n_fit_voxels <= n * n * n - n_corrupt,
            "expected ≥{n_corrupt} rejected voxels, kept {}",
            diag.n_fit_voxels
        );
        // Clean voxels still pin to target (median).
        let mut sums: Vec<f64> = Vec::new();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    sums.push(
                        wm[(x, y, z, 0)] as f64
                            + gm[(x, y, z, 0)] as f64
                            + csf[(x, y, z, 0)] as f64,
                    );
                }
            }
        }
        sums.sort_by(cmp_nan_first);
        let median = sums[sums.len() / 2];
        assert!(
            (median - target).abs() < 0.03,
            "median sum {median} strayed from target {target}"
        );
    }

    /// Default (no `apply_balance`): output = input / field only, so tissue
    /// ratios within a voxel are preserved exactly.
    #[test]
    fn default_output_preserves_within_voxel_ratios() {
        let n = 12;
        let (mut wm, mut gm, mut csf, mask, _tf) = synth_volume(n, 1.0);
        let wm0 = wm.clone();
        let gm0 = gm.clone();
        let csf0 = csf.clone();
        let cfg = MtnormaliseConfig {
            target_sum: Some(1.0),
            ..Default::default()
        };
        mtnormalise(&mut wm, &mut gm, &mut csf, &mask, &cfg).unwrap();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let r_before = wm0[(x, y, z, 0)] as f64 / gm0[(x, y, z, 0)].max(1e-9) as f64;
                    let r_after = wm[(x, y, z, 0)] as f64 / gm[(x, y, z, 0)].max(1e-9) as f64;
                    if r_before.is_finite() && r_before > 1e-6 {
                        assert!(
                            ((r_after / r_before) - 1.0).abs() < 1e-4,
                            "WM/GM ratio changed at ({x},{y},{z})"
                        );
                    }
                    let _ = csf0[(x, y, z, 0)];
                }
            }
        }
    }
}
