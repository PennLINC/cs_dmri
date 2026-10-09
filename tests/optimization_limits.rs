// SPDX-License-Identifier: MIT OR Apache-2.0
//! Boundary / limit-behavior tests for FISTA and Tikhonov.
//!
//! Each test pins a closed-form fact about the optimum (α → 0 reduces to
//! OLS, α ≥ α_max reduces to zero, square invertible M with α = 0 reduces to
//! M⁻¹y, the log α-path endpoints are exact). These guard against subtle
//! normalization or step-size drift that wouldn't break the planted-recovery
//! tests but would still corrupt downstream fits.

mod common;

use cs_dmri::solver::alpha::{alpha_max, log_path};
use cs_dmri::solver::fista::FistaSolver;
use cs_dmri::solver::tikhonov::TikhonovSolver;
use cs_dmri::solver::{Problem, Solver};
use nalgebra::{DMatrix, DVector};

use common::{
    make_design, make_noisy_signal, make_planted_beta, make_square_invertible_design,
    ols_solution,
};

fn fit_fista(
    design: &DMatrix<f64>,
    signal: &DVector<f64>,
    alpha: f64,
    non_negative: bool,
) -> DVector<f64> {
    let solver = FistaSolver::new(design.clone(), alpha, 200_000, 1e-12, non_negative);
    let problem = Problem { design, signal };
    let (coef, _) = solver.fit(&problem);
    coef
}

#[test]
fn fista_alpha_zero_matches_ols_overdetermined() {
    let m = make_design(200, 20, 0xA1);
    let beta_star = make_planted_beta(20, 6, true, 0xA2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xA3);
    let beta_ols = ols_solution(&m, &y);
    // α = 1e-12 is "effectively zero" for cs_dmri's FISTA; using a literal 0
    // would make `alpha.max(1e-12)` in cs-fit kick in anyway and is identical.
    let beta_fista = fit_fista(&m, &y, 1e-12, false);
    let diff = (&beta_fista - &beta_ols).iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(
        diff < 1e-3,
        "FISTA(α≈0) vs OLS infinity-norm gap {diff:.3e} above 1e-3"
    );
}

#[test]
fn fista_alpha_above_alpha_max_returns_zero() {
    let m = make_design(150, 40, 0xB1);
    let beta_star = make_planted_beta(40, 5, true, 0xB2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xB3);
    let amax = alpha_max(&m, &y);
    let beta_hat = fit_fista(&m, &y, 10.0 * amax, false);
    let max_abs = beta_hat.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_abs < 1e-12,
        "FISTA at α=10·α_max returned non-zero: max|β̂|={max_abs:.3e}"
    );
}

#[test]
fn fista_square_invertible_alpha_zero_recovers_inverse() {
    let n = 25;
    let m = make_square_invertible_design(n, 0xC1);
    let mut rng = rand_chacha::ChaCha8Rng::from_seed([0xC2; 32]);
    use rand::Rng;
    let y = DVector::<f64>::from_fn(n, |_, _| rng.gen_range(-1.0..1.0));
    // Ground truth: M⁻¹y via QR.
    let beta_inv = m
        .clone()
        .lu()
        .solve(&y)
        .expect("constructed M is invertible");
    let beta_fista = fit_fista(&m, &y, 1e-12, false);
    let diff = (&beta_fista - &beta_inv).iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(
        diff < 1e-4,
        "FISTA(α≈0) vs M⁻¹y infinity-norm gap {diff:.3e} above 1e-4"
    );
}

/// Build pure-ridge regularization (D = I, no secondary). Decouples the
/// Tikhonov-limit tests from the SHORE basis, which has zero-diagonal modes
/// that defeat λ → ∞ shrinkage by construction.
fn ridge_regularization(n: usize) -> cs_dmri::basis::RegularizationDiagonals {
    cs_dmri::basis::RegularizationDiagonals {
        primary: DVector::from_element(n, 1.0),
        secondary: None,
    }
}

#[test]
fn tikhonov_lambda_zero_matches_ols() {
    // Pure ridge on a well-conditioned overdetermined Gaussian design.
    let m = make_design(200, 30, 0xD1);
    let n = 30;
    let reg = ridge_regularization(n);
    let beta_star = make_planted_beta(n, 6, true, 0xD2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xD3);

    // λ = 1e-14 perturbs the normal equations only at round-off level.
    let solver = TikhonovSolver::new(m.clone(), &reg, 1e-14, 0.0);
    let problem = Problem { design: &m, signal: &y };
    let (beta_tik, _) = solver.fit(&problem);
    let beta_ols = ols_solution(&m, &y);
    let diff = (&beta_tik - &beta_ols).iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(
        diff < 1e-6,
        "Tikhonov(λ→0) vs OLS infinity-norm gap {diff:.3e} above 1e-6"
    );
}

#[test]
fn tikhonov_lambda_huge_shrinks_to_zero() {
    // Pure ridge — every coefficient is penalized, so λ → ∞ uniformly shrinks.
    let m = make_design(200, 30, 0xE1);
    let n = 30;
    let reg = ridge_regularization(n);
    let beta_star = make_planted_beta(n, 6, true, 0xE2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xE3);

    // ‖MᵀM‖ is O(n/m) for our scaling; λ = 1e10 dominates by 9 orders.
    let solver = TikhonovSolver::new(m.clone(), &reg, 1e10, 0.0);
    let problem = Problem { design: &m, signal: &y };
    let (beta_tik, _) = solver.fit(&problem);
    let max_abs = beta_tik.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_abs < 1e-6,
        "Tikhonov(λ→∞, ridge) should shrink to ~0; got max|β̂|={max_abs:.3e}"
    );
}

#[test]
fn alpha_max_log_path_endpoints_exact() {
    let m = make_design(120, 35, 0xF1);
    let beta_star = make_planted_beta(35, 5, true, 0xF2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xF3);
    let amax = alpha_max(&m, &y);
    let n = 20;
    let eps = 1e-3;
    let path = log_path(amax, n, eps);
    assert_eq!(path.len(), n);
    assert!((path[0] - amax).abs() < amax * 1e-12, "path[0] != α_max");
    assert!(
        (path[n - 1] - amax * eps).abs() < amax * eps * 1e-12,
        "path[-1] != α_max · eps"
    );
    for w in path.windows(2) {
        assert!(w[0] > w[1], "log α-path not strictly decreasing");
    }
}

use rand::SeedableRng;
