// SPDX-License-Identifier: MIT OR Apache-2.0
//! Constrained, smooth-in-b dense response fit.
//!
//! The kernel estimator in [`crate::multitissue::response_estimation`] fits each
//! dense-b grid row on its own, from the samples a Gaussian in b assigns it.
//! That works when q-space is densely sampled in b. It does not when it is not:
//!
//! - **Shelled data** turns every row into a copy of its nearest shell — a
//!   staircase in b, flat around each shell and jumping halfway between.
//! - **Sparse non-shelled data** (CS-DSI: a few dozen volumes over ~16 distinct
//!   b-values, a handful of directions each) leaves rows between b-values with
//!   little or no support, so they are noisy or carried forward.
//! - **Gradient deviation** (`--graddev`) spreads b-values off the nominal ones
//!   and past bmax, where support is thin and the rows come out implausible
//!   (amplitude rising with b).
//!
//! Each of those breaks the forward model as soon as a measurement's b falls
//! between rows. This fit instead estimates all rows *jointly*:
//!
//! ```text
//!   min  Σ_i ( y_i − A(b_i, θ_i) )²  +  λ Σ_k Σ_l (r_{k−1,l} − 2 r_{k,l} + r_{k+1,l})²
//!   s.t. r_{k,0} ≥ 0,   r_{k,0} ≥ r_{k+1,0}      (spherical mean ≥ 0, never grows with b)
//!        (−1)^{l/2} r_{k,l} ≥ 0   for l ≥ 2      (WM only)
//! ```
//!
//! where `A(b, θ)` interpolates linearly between the two rows bracketing `b` —
//! the same interpolation the deconvolution's `--b-interp` forward model uses —
//! and `A_k(θ) = Σ_l r_{k,l} √((4l+1)/4π) P_{2l}(cos θ)`. Rows at or below the
//! b=0 threshold are pinned to the measured b=0 row, and the smoothness term
//! bridges gaps between sampled b-values with a discrete spline instead of a
//! carried-forward row.
//!
//! **Why these constraints and not MRtrix `amp2response`'s.** Deconvolution only
//! ever uses the zonal coefficients, and by the convolution theorem the right
//! ones are the true response's projection onto `l ≤ lmax` — Gibbs ringing in
//! the amplitude included. `amp2response` constrains the *amplitude* (≥ 0,
//! monotone in angle); at high b the true profile is too sharp for lmax 8, its
//! projection rings, and forcing the ringing out biases every coefficient (and,
//! coupled across b, freezes rows into plateaus). The constraints above hold
//! *exactly* at any truncation order instead: `r_0` is the exact spherical mean,
//! which is non-negative and decays with b; and for any positive mixture of
//! axially symmetric Gaussian compartments with `D∥ ≥ D⊥` (sticks, zeppelins,
//! Watson-dispersed fibers) the zonal coefficients alternate as `(−1)^{l/2}`
//! (each is a positive confluent hypergeometric term, by Kummer's
//! transformation); a Rician floor only adds to `r_0`. They are what stops thin
//! or extrapolated rows (graddev's b-values past bmax) from going implausible.
//! Isotropic tissues use the same fit with only `l = 0`.

use nalgebra::{DMatrix, DVector};

use crate::math::assoc_legendre;
use crate::multitissue::consh::DenseBGrid;
use crate::solver::icls::{IclsConfig, IclsSolver};
use crate::{CsDmriError, Result};

/// Second-difference weight left in rows with no data nearby: enough to make
/// the gap a smooth interpolant (and the system well posed), too little to pull
/// the sampled rows bordering it.
const GAP_TENSION: f64 = 1e-3;

/// Knobs for [`fit_dense_rows`].
#[derive(Debug, Clone, Copy)]
pub struct SmoothResponseConfig {
    /// Smoothing length in b (s/mm²): where data are present the fit pools
    /// neighboring b-values over about the span a Gaussian of this σ would
    /// (the kernel estimator's `sigma_b`) — each b-value measured along only a
    /// few directions in sparse schemes, or from different voxels under
    /// gradient deviation — rather than fitting row by row. Across gaps wider
    /// than this, rows are interpolated instead.
    pub smoothing_b: f64,
    /// Cap on active-set iterations of the constrained solve.
    pub max_iter: usize,
}

impl Default for SmoothResponseConfig {
    fn default() -> Self {
        Self { smoothing_b: 150.0, max_iter: 100_000 }
    }
}

/// Zonal basis value `√((4j+1)/4π) · P_{2j}(c)` for `j = 0 … n_zonal−1`, the
/// normalization the response rows and [`crate::sh::zonal_factors_per_coeff`]
/// share.
fn zonal_basis(c: f64, n_zonal: usize, out: &mut [f64]) {
    for (j, o) in out.iter_mut().enumerate().take(n_zonal) {
        let l = 2 * j;
        *o = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt() * assoc_legendre(l as u32, 0, c);
    }
}

/// Fit every dense-b row of one tissue's zonal response jointly; see the module
/// docs. `pool_b` / `pool_y` are the pooled samples (b-value and normalized
/// signal); `pool_cos` their `|cos θ|` to the fiber axis, or `None` for an
/// isotropic tissue (then `lmax` must be 0); `pool_w` optional per-sample
/// least-squares weights (default 1). Rows with `grid.bs[k] ≤ b0_thr` are
/// pinned to `b0_row`. Returns one `lmax/2 + 1`-long row per grid point.
#[allow(clippy::too_many_arguments)]
pub fn fit_dense_rows(
    grid: &DenseBGrid,
    b0_thr: f64,
    b0_row: &[f64],
    lmax: usize,
    pool_b: &[f64],
    pool_cos: Option<&[f64]>,
    pool_y: &[f64],
    pool_w: Option<&[f64]>,
    cfg: &SmoothResponseConfig,
) -> Result<Vec<Vec<f64>>> {
    let nz = lmax / 2 + 1;
    let ng = grid.len();
    if pool_cos.is_none() && nz != 1 {
        return Err(CsDmriError::Other("isotropic response fit requires lmax 0".into()));
    }
    if ng < 3 || pool_b.is_empty() {
        return Err(CsDmriError::Other(format!(
            "smooth response fit: need ≥3 grid rows and some samples (rows {ng}, samples {})",
            pool_b.len()
        )));
    }
    let p = ng * nz;
    let idx = |k: usize, j: usize| k * nz + j;

    // ---- data term: normal equations of the row-interpolated prediction ----
    let mut g = DMatrix::<f64>::zeros(p, p);
    let mut c = DVector::<f64>::zeros(p);
    let mut z = vec![0.0; nz];
    for (i, (&b, &y)) in pool_b.iter().zip(pool_y).enumerate() {
        if b <= b0_thr {
            continue;
        }
        match pool_cos {
            Some(cos) => zonal_basis(cos[i], nz, &mut z),
            None => z[0] = (1.0 / (4.0 * std::f64::consts::PI)).sqrt(),
        }
        let sw = pool_w.map_or(1.0, |w| w[i]);
        if sw <= 0.0 {
            continue;
        }
        let (k, w) = grid.bracket(b);
        let rows = [(k, 1.0 - w), (k + 1, w)];
        for &(ka, wa) in &rows {
            if wa == 0.0 {
                continue;
            }
            for ja in 0..nz {
                let fa = wa * z[ja];
                c[idx(ka, ja)] += sw * fa * y;
                for &(kb, wb) in &rows {
                    if wb == 0.0 {
                        continue;
                    }
                    for jb in 0..nz {
                        g[(idx(ka, ja), idx(kb, jb))] += sw * fa * wb * z[jb];
                    }
                }
            }
        }
    }
    // Data weight of a sampled row: the mean diagonal over the rows that carry
    // samples. Everything is expressed in that unit, so the smoothing length and
    // the pin mean the same thing on every acquisition.
    let row_weight: Vec<f64> = (0..ng).map(|k| (0..nz).map(|j| g[(idx(k, j), idx(k, j))]).sum::<f64>()).collect();
    let peak = row_weight.iter().cloned().fold(0.0_f64, f64::max);
    let sampled: Vec<f64> = row_weight.iter().cloned().filter(|&w| w > 1e-9 * peak).collect();
    if sampled.is_empty() {
        return Err(CsDmriError::Other("smooth response fit: no diffusion-weighted samples".into()));
    }
    let data_scale = sampled.iter().sum::<f64>() / (sampled.len() * nz) as f64;
    g /= data_scale;
    c /= data_scale;

    // ---- smoothness: second differences across rows, per coefficient ----
    // A second-difference penalty λ against data weight w acts as a spline
    // whose half-power length is 2π (λ/w)^¼ rows; a Gaussian of σ rows cuts off
    // at ≈ 5.3 σ. Equating them: λ/w = (0.85 σ)⁴. The penalty is LOCAL: w is
    // the data weight within ±σ of the row (Gaussian-averaged), so where
    // b-values are dense the fit pools neighbors over ~σ (as the kernel
    // estimator does), while across a gap between sparse b-values (CS-DSI's
    // 600-wide steps, shells) only a light tension remains — interpolating the
    // gap without dragging the sampled rows on either side toward each other.
    let sigma_rows = (cfg.smoothing_b / grid.step).max(1e-6);
    let shape = (0.85 * sigma_rows).powi(4);
    let row_w: Vec<f64> = row_weight.iter().map(|w| w / data_scale / nz as f64).collect();
    let reach = (3.0 * sigma_rows).ceil() as isize;
    for k in 1..ng - 1 {
        let (mut num, mut den) = (0.0, 0.0);
        for d in -reach..=reach {
            let m = k as isize + d;
            if m < 0 || m >= ng as isize {
                continue;
            }
            let g = (-0.5 * (d as f64 / sigma_rows).powi(2)).exp();
            num += g * row_w[m as usize];
            den += g;
        }
        let lam = shape * num / den + GAP_TENSION;
        for j in 0..nz {
            let t = [(idx(k - 1, j), 1.0), (idx(k, j), -2.0), (idx(k + 1, j), 1.0)];
            for &(a, va) in &t {
                for &(b, vb) in &t {
                    g[(a, b)] += lam * va * vb;
                }
            }
        }
    }

    // ---- pin the b=0 rows (a heavy quadratic keeps every constraint
    // homogeneous, which the ICLS solver requires) ----
    const PIN: f64 = 1e6;
    let pinned: Vec<bool> = grid.bs.iter().map(|&b| b <= b0_thr).collect();
    for k in (0..ng).filter(|&k| pinned[k]) {
        for j in 0..nz {
            let v = b0_row.get(j).copied().unwrap_or(0.0);
            g[(idx(k, j), idx(k, j))] += PIN;
            c[idx(k, j)] += PIN * v;
        }
    }

    // ---- constraints C x ≥ 0 (exact at any lmax; see the module docs) ----
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();
    for k in (0..ng).filter(|&k| !pinned[k]) {
        rows.push(vec![(idx(k, 0), 1.0)]); // r_{k,0} ≥ 0
        for j in 1..nz {
            let sign = if j % 2 == 1 { -1.0 } else { 1.0 }; // (−1)^{l/2}, l = 2j
            rows.push(vec![(idx(k, j), sign)]);
        }
    }
    for k in 0..ng - 1 {
        if !pinned[k + 1] {
            rows.push(vec![(idx(k, 0), 1.0), (idx(k + 1, 0), -1.0)]); // r_{k,0} ≥ r_{k+1,0}
        }
    }
    let mut cmat = DMatrix::<f64>::zeros(rows.len(), p);
    for (r, entries) in rows.iter().enumerate() {
        for &(col, v) in entries {
            cmat[(r, col)] = v;
        }
    }

    // ---- solve: G = L Lᵀ, so ½‖Lᵀx − L⁻¹c‖² is the quadratic above ----
    let chol = nalgebra::Cholesky::new(g)
        .ok_or_else(|| CsDmriError::Other("smooth response fit: normal equations not positive definite".into()))?;
    let l = chol.l();
    let rhs = l
        .solve_lower_triangular(&c)
        .ok_or_else(|| CsDmriError::Other("smooth response fit: triangular solve failed".into()))?;
    let solver = IclsSolver::new(
        l.transpose(),
        cmat,
        IclsConfig { max_iter: cfg.max_iter, ..IclsConfig::default() },
    );
    let (x, diag) = solver.solve(&rhs);
    if !diag.converged {
        return Err(CsDmriError::Other(format!(
            "smooth response fit did not converge in {} iterations",
            diag.iterations
        )));
    }
    Ok((0..ng).map(|k| (0..nz).map(|j| x[idx(k, j)]).collect()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Amplitude of a zonal row at |cos θ| = c.
    fn amp(row: &[f64], c: f64) -> f64 {
        let mut z = vec![0.0; row.len()];
        zonal_basis(c, row.len(), &mut z);
        row.iter().zip(&z).map(|(r, z)| r * z).sum()
    }

    /// Stick-plus-ball WM signal: S/S0 at b, |cos θ| = c.
    fn wm_signal(b: f64, c: f64) -> f64 {
        0.7 * (-b * 2.2e-3 * c * c).exp() + 0.3 * (-b * 0.8e-3).exp()
    }

    /// Best lmax-`2(nz−1)` zonal representation of the true signal at b — what a
    /// perfect estimator could reach (the SH truncation is not its error).
    fn projected_row(b: f64, nz: usize) -> Vec<f64> {
        let cs: Vec<f64> = (0..=400).map(|i| i as f64 / 400.0).collect();
        let mut x = DMatrix::<f64>::zeros(cs.len(), nz);
        let mut y = DVector::<f64>::zeros(cs.len());
        let mut z = vec![0.0; nz];
        for (i, &c) in cs.iter().enumerate() {
            zonal_basis(c, nz, &mut z);
            for j in 0..nz {
                x[(i, j)] = z[j];
            }
            y[i] = wm_signal(b, c);
        }
        let r = (x.transpose() * &x).cholesky().unwrap().solve(&(x.transpose() * y));
        r.iter().copied().collect()
    }

    /// Samples at a CS-DSI-like set of sparse b-values, a few directions each
    /// relative to many fiber orientations (pooled over voxels).
    fn sparse_samples() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let bs = [250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 2000.0, 2250.0, 2500.0, 3250.0, 4000.0, 5000.0];
        let (mut pb, mut pc, mut py) = (Vec::new(), Vec::new(), Vec::new());
        for (i, &b) in bs.iter().enumerate() {
            for v in 0..400 {
                // deterministic spread of |cos θ| over [0, 1]
                let c = ((v * 37 + i * 11) % 401) as f64 / 400.0;
                pb.push(b);
                pc.push(c);
                py.push(wm_signal(b, c));
            }
        }
        (pb, pc, py)
    }

    #[test]
    fn sparse_b_fit_tracks_the_true_response_between_sampled_b_values() {
        let grid = DenseBGrid::new(5000.0, 50.0, 100.0);
        let (pb, pc, py) = sparse_samples();
        let b0 = [wm_signal(0.0, 0.0) * (4.0 * std::f64::consts::PI).sqrt(), 0.0, 0.0, 0.0, 0.0];
        let rows = fit_dense_rows(&grid, 50.0, &b0, 8, &pb, Some(&pc), &py, None, &SmoothResponseConfig::default()).unwrap();
        // At and between sampled b-values (e.g. 1750, 2850, 3600, 4500) the fit
        // must be as close to the true signal as the best lmax-8 representation
        // of it — the projection deconvolution needs — and, between them,
        // closer than the staircase (nearest sampled b's row) would be.
        let cs: Vec<f64> = (0..=50).map(|i| i as f64 / 50.0).collect();
        let rms = |f: &dyn Fn(f64) -> f64, b: f64| {
            (cs.iter().map(|&c| (f(c) - wm_signal(b, c)).powi(2)).sum::<f64>() / cs.len() as f64).sqrt()
        };
        for &b in &[1750.0, 2850.0, 3600.0, 4500.0, 1000.0, 3250.0] {
            let k = (b / 50.0) as usize;
            let best = projected_row(b, 5);
            let fit_err = rms(&|c| amp(&rows[k], c), b);
            let best_err = rms(&|c| amp(&best, c), b);
            // What the staircase gives at b: the best row of the nearest sampled b.
            let sampled = [250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 2000.0, 2250.0, 2500.0, 3250.0, 4000.0, 5000.0];
            let near = *sampled.iter().min_by(|a, c| (*a - b).abs().total_cmp(&(*c - b).abs())).unwrap();
            let stair = projected_row(near, 5);
            let stair_err = rms(&|c| amp(&stair, c), b);
            // Smoothing over ~σ_b=150 costs a little where the response curves
            // (the kernel estimator pays the same); ≤ 0.6% of S0.
            assert!(fit_err <= best_err + 6e-3, "b={b}: rms error {fit_err:.4} vs lmax-8 projection {best_err:.4}");
            if near != b {
                assert!(fit_err < stair_err, "b={b}: rms error {fit_err:.4} not below the staircase's {stair_err:.4}");
            }
        }
    }

    #[test]
    fn constraints_hold_on_every_row_even_past_the_data() {
        // Data only up to b=2000, grid to 4000 (graddev-style extension).
        let grid = DenseBGrid::new(4000.0, 50.0, 100.0);
        let (pb, pc, py): (Vec<f64>, Vec<f64>, Vec<f64>) = {
            let (b, c, y) = sparse_samples();
            let keep: Vec<usize> = (0..b.len()).filter(|&i| b[i] <= 2000.0).collect();
            (keep.iter().map(|&i| b[i]).collect(), keep.iter().map(|&i| c[i]).collect(), keep.iter().map(|&i| y[i]).collect())
        };
        let b0 = [(4.0 * std::f64::consts::PI).sqrt(), 0.0, 0.0, 0.0, 0.0];
        let rows = fit_dense_rows(&grid, 50.0, &b0, 8, &pb, Some(&pc), &py, None, &SmoothResponseConfig::default()).unwrap();
        for k in 0..rows.len() {
            assert!(rows[k][0] > -1e-9, "row {k}: spherical mean < 0");
            if k + 1 < rows.len() {
                assert!(rows[k + 1][0] <= rows[k][0] + 1e-9, "row {}: spherical mean rises with b", k + 1);
            }
            if k > 1 {
                for j in 1..5 {
                    let sign = if j % 2 == 1 { -1.0 } else { 1.0 };
                    assert!(sign * rows[k][j] > -1e-9, "row {k}, l={}: wrong sign {}", 2 * j, rows[k][j]);
                }
            }
        }
        assert!((rows[0][0] - b0[0]).abs() < 1e-4, "b=0 row must stay pinned");
    }

    #[test]
    fn isotropic_fit_is_monotone_and_smooth_across_a_shell_gap() {
        // Shelled iso samples at 1000 and 3000 only: the row at 2000 must lie
        // between them (no carried-forward staircase).
        let grid = DenseBGrid::new(3000.0, 50.0, 100.0);
        let (mut pb, mut py) = (Vec::new(), Vec::new());
        for &b in &[1000.0, 3000.0] {
            for _ in 0..200 {
                pb.push(b);
                py.push((-b * 1e-3_f64).exp() * (4.0 * std::f64::consts::PI).sqrt() / (4.0 * std::f64::consts::PI).sqrt());
            }
        }
        let b0 = [(4.0 * std::f64::consts::PI).sqrt()];
        let rows = fit_dense_rows(&grid, 50.0, &b0, 0, &pb, None, &py, None, &SmoothResponseConfig::default()).unwrap();
        let s = |k: usize| rows[k][0] / (4.0 * std::f64::consts::PI).sqrt();
        assert!(s(40) < s(20) && s(40) > s(60), "row at 2000 between its shells");
        assert!((s(40) - s(20)).abs() > 0.05, "not flat after the 1000 shell");
        for k in 1..rows.len() {
            assert!(rows[k][0] <= rows[k - 1][0] + 1e-9);
        }
    }
}
