// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared helpers for the optimization-correctness integration tests.
//!
//! Synthetic problem builders use `ChaCha8Rng` seeded with a fixed `u64`
//! so failures reproduce on any machine. The KKT-residual helpers are the
//! correctness witnesses: a passing FISTA call should drive these residuals
//! toward zero on the test problems.

#![allow(dead_code)]

use cs_dmri::basis::RegularizationDiagonals;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Sample a standard normal via Box–Muller. Pure-Rust, no `rand_distr` dep.
fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    loop {
        let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
        let u2: f64 = rng.gen_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        if z.is_finite() {
            return z;
        }
    }
}

/// Gaussian random design with iid `N(0, 1/m)` entries (so column norms are
/// ~1 and the minimum eigenvalue of `MᵀM/m` is bounded for `m ≳ n`).
pub fn make_design(m: usize, n: usize, seed: u64) -> DMatrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let scale = 1.0 / (m as f64).sqrt();
    DMatrix::<f64>::from_fn(m, n, |_, _| standard_normal(&mut rng) * scale)
}

/// Gaussian design pushed to a target condition number `kappa` via SVD
/// rescaling. Singular values are geometrically spaced from 1 down to `1/kappa`.
pub fn make_ill_conditioned_design(m: usize, n: usize, kappa: f64, seed: u64) -> DMatrix<f64> {
    let raw = make_design(m, n, seed);
    let svd = raw.svd(true, true);
    let u = svd.u.expect("U present");
    let vt = svd.v_t.expect("V_t present");
    let r = u.ncols().min(vt.nrows());
    // Geometric spacing from 1 to 1/kappa over r singular values.
    let log_step = if r > 1 {
        -(kappa.ln()) / (r as f64 - 1.0)
    } else {
        0.0
    };
    let s: DVector<f64> = DVector::from_fn(r, |i, _| (i as f64 * log_step).exp());
    let mut us = u.columns(0, r).into_owned();
    for (j, sj) in s.iter().enumerate() {
        let mut col = us.column_mut(j);
        col *= *sj;
    }
    us * vt.rows(0, r)
}

/// Square invertible design (random Gaussian, then rescale spectrum to
/// uniformly bounded singular values).
pub fn make_square_invertible_design(n: usize, seed: u64) -> DMatrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let raw = DMatrix::<f64>::from_fn(n, n, |_, _| standard_normal(&mut rng));
    let svd = raw.svd(true, true);
    let u = svd.u.expect("U present");
    let vt = svd.v_t.expect("V_t present");
    // Reset all singular values to 1 → orthonormal-by-construction matrix
    // with κ = 1; perturb slightly so it's not literally a rotation.
    let mut s = DVector::<f64>::from_element(n, 1.0);
    for i in 0..n {
        s[i] = 1.0 + 0.1 * ((i as f64) / (n as f64));
    }
    let mut us = u.columns(0, n).into_owned();
    for (j, sj) in s.iter().enumerate() {
        let mut col = us.column_mut(j);
        col *= *sj;
    }
    us * vt
}

/// Planted sparse coefficient vector. Returns β* with exactly `k` nonzeros at
/// random positions. Magnitudes are uniform on `[0.5, 2.0]`; signs are random
/// when `signed`, otherwise all positive (for non-negative LASSO tests).
pub fn make_planted_beta(n: usize, k: usize, signed: bool, seed: u64) -> DVector<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    // Fisher–Yates
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    let mut beta = DVector::<f64>::zeros(n);
    for &idx in indices.iter().take(k) {
        let mag = 0.5 + 1.5 * rng.gen::<f64>();
        let sign = if signed && rng.gen::<bool>() { -1.0 } else { 1.0 };
        beta[idx] = sign * mag;
    }
    beta
}

/// `y = Mβ + σ ε`, `ε ~ N(0, I_m)`.
pub fn make_noisy_signal(m: &DMatrix<f64>, beta: &DVector<f64>, sigma: f64, seed: u64) -> DVector<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let clean = m * beta;
    DVector::from_fn(clean.len(), |i, _| clean[i] + sigma * standard_normal(&mut rng))
}

/// LASSO objective in cs-dmri / sklearn convention:
///     F(β) = (1/(2m)) ‖Mβ − y‖² + α ‖β‖₁
pub fn lasso_objective(m: &DMatrix<f64>, y: &DVector<f64>, beta: &DVector<f64>, alpha: f64) -> f64 {
    let r = m * beta - y;
    let m_rows = m.nrows() as f64;
    0.5 / m_rows * r.norm_squared() + alpha * beta.iter().map(|b| b.abs()).sum::<f64>()
}

/// KKT residuals for LASSO with sklearn convention. Returns
/// `(max_active_residual, max_inactive_excess)` where both should be near 0
/// for the true minimizer.
///
/// - **active** (`|β_j| > zero_tol`): subgradient must vanish, so
///   `(1/m) M_jᵀ (Mβ − y) + α · sign(β_j) = 0`. We report
///   `max_j |…|` over the active set.
/// - **inactive** (`|β_j| ≤ zero_tol`): subdifferential is `[-α, α]`, so
///   the gradient component must lie in that band. We report
///   `max_j max(0, |(1/m) M_jᵀ (Mβ − y)| − α)`.
pub fn kkt_residual_lasso(
    m: &DMatrix<f64>,
    y: &DVector<f64>,
    beta: &DVector<f64>,
    alpha: f64,
) -> (f64, f64) {
    let m_rows = m.nrows() as f64;
    let r = m * beta - y;
    let g = (m.transpose() * &r) / m_rows;
    let zero_tol = 1e-10;
    let mut max_active = 0.0_f64;
    let mut max_inactive = 0.0_f64;
    for (j, b) in beta.iter().enumerate() {
        if b.abs() > zero_tol {
            let v = (g[j] + alpha * b.signum()).abs();
            if v > max_active {
                max_active = v;
            }
        } else {
            let excess = g[j].abs() - alpha;
            if excess > max_inactive {
                max_inactive = excess;
            }
        }
    }
    (max_active, max_inactive)
}

/// KKT residuals for non-negative LASSO. Subdifferential of `‖β‖₁` restricted
/// to β ≥ 0 collapses to a one-sided cone:
///
/// - **active** (β_j > 0): `(1/m) M_jᵀ r + α = 0`.
/// - **inactive** (β_j = 0): `(1/m) M_jᵀ r ≥ −α`, i.e. excess
///   `max(0, −((1/m) M_jᵀ r) − α)`.
pub fn kkt_residual_nonneg_lasso(
    m: &DMatrix<f64>,
    y: &DVector<f64>,
    beta: &DVector<f64>,
    alpha: f64,
) -> (f64, f64) {
    let m_rows = m.nrows() as f64;
    let r = m * beta - y;
    let g = (m.transpose() * &r) / m_rows;
    let zero_tol = 1e-10;
    let mut max_active = 0.0_f64;
    let mut max_inactive = 0.0_f64;
    for (j, b) in beta.iter().enumerate() {
        if *b > zero_tol {
            let v = (g[j] + alpha).abs();
            if v > max_active {
                max_active = v;
            }
        } else {
            // Constraint β_j = 0 active → require g[j] ≥ −α (else would
            // benefit from increasing β_j). Excess = max(0, −g[j] − α).
            let excess = -g[j] - alpha;
            if excess > max_inactive {
                max_inactive = excess;
            }
        }
    }
    (max_active, max_inactive)
}

/// Tikhonov normal-equation residual:
///     ‖(MᵀM + λ_n D_n + λ_l D_l) β − Mᵀy‖₂ / ‖Mᵀy‖₂
///
/// cs-dmri's TikhonovSolver minimizes `‖Mβ−y‖² + λ_n ⟨β, D_n β⟩ + λ_l ⟨β, D_l β⟩`
/// (no 1/(2m) scaling; see [src/solver/tikhonov.rs](../../src/solver/tikhonov.rs)).
pub fn tikhonov_normal_residual(
    m: &DMatrix<f64>,
    y: &DVector<f64>,
    beta: &DVector<f64>,
    lambda_primary: f64,
    lambda_secondary: f64,
    reg: &RegularizationDiagonals,
) -> f64 {
    let mt = m.transpose();
    let mut lhs = (&mt * m) * beta;
    for (i, &d) in reg.primary.iter().enumerate() {
        lhs[i] += lambda_primary * d * beta[i];
    }
    if let Some(sec) = &reg.secondary {
        for (i, &d) in sec.iter().enumerate() {
            lhs[i] += lambda_secondary * d * beta[i];
        }
    }
    let rhs = mt * y;
    let denom = rhs.norm().max(1e-30);
    (lhs - rhs).norm() / denom
}

/// Closed-form OLS via QR (`min ‖Mβ − y‖₂` for full column-rank M). Used as
/// the ground truth in α → 0 limit tests.
pub fn ols_solution(m: &DMatrix<f64>, y: &DVector<f64>) -> DVector<f64> {
    // (MᵀM)⁻¹ Mᵀy via Cholesky on the normal equations. Adequate for the
    // well-conditioned test problems; ill-conditioned cases use a different
    // ground-truth comparator (sklearn fixture).
    let mt = m.transpose();
    let normal = &mt * m;
    let rhs = &mt * y;
    let chol = normal
        .cholesky()
        .expect("OLS test problem must be full column rank");
    chol.solve(&rhs)
}

/// Set difference of supports; returns indices that are in `a` but not in `b`.
pub fn support_diff(a: &DVector<f64>, b: &DVector<f64>, zero_tol: f64) -> Vec<usize> {
    a.iter()
        .enumerate()
        .filter_map(|(i, ai)| {
            if ai.abs() > zero_tol && b[i].abs() <= zero_tol {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Indices where |β_j| > zero_tol.
pub fn support_of(beta: &DVector<f64>, zero_tol: f64) -> Vec<usize> {
    beta.iter()
        .enumerate()
        .filter_map(|(i, b)| if b.abs() > zero_tol { Some(i) } else { None })
        .collect()
}
