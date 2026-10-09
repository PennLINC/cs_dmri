// SPDX-License-Identifier: MIT OR Apache-2.0
//! KKT-residual correctness tests for FISTA (plain and non-negative) and the
//! Tikhonov normal equations.
//!
//! Passing these tests proves the solver returned the *true* minimizer of the
//! convex program (within tolerance), not just a point near it. They are the
//! gold-standard correctness witness for a LASSO solver — see
//! [scikit-learn's test_coordinate_descent.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/tests/test_coordinate_descent.py)
//! for the same pattern.

mod common;

use cs_dmri::ShoreBasis;
use cs_dmri::basis::Basis;
use cs_dmri::qspace::GradientTable;
use cs_dmri::solver::fista::FistaSolver;
use cs_dmri::solver::tikhonov::TikhonovSolver;
use cs_dmri::solver::{Problem, Solver};

use common::{
    kkt_residual_lasso, kkt_residual_nonneg_lasso, make_design,
    make_ill_conditioned_design, make_noisy_signal, make_planted_beta,
    tikhonov_normal_residual,
};

fn fit_fista(
    design: &nalgebra::DMatrix<f64>,
    signal: &nalgebra::DVector<f64>,
    alpha: f64,
    non_negative: bool,
) -> nalgebra::DVector<f64> {
    let solver = FistaSolver::new(design.clone(), alpha, 100_000, 1e-10, non_negative);
    let problem = Problem { design, signal };
    let (coef, _) = solver.fit(&problem);
    coef
}

#[test]
fn fista_kkt_well_conditioned() {
    let m = make_design(200, 50, 0xA1);
    let beta_star = make_planted_beta(50, 8, true, 0xA2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xA3);
    let alpha = 0.05;
    let beta_hat = fit_fista(&m, &y, alpha, false);
    let (active, inactive) = kkt_residual_lasso(&m, &y, &beta_hat, alpha);
    assert!(
        active < 1e-6,
        "active KKT residual {active:.3e} above 1e-6 (well-conditioned)"
    );
    assert!(
        inactive < 1e-6,
        "inactive KKT excess {inactive:.3e} above 1e-6 (well-conditioned)"
    );
}

#[test]
fn fista_kkt_ill_conditioned() {
    // κ ≈ 10⁴ exercises the gradient/step-size path; FISTA still converges
    // but more slowly. Loose KKT tolerance accommodates the slowdown without
    // hiding outright bugs.
    let m = make_ill_conditioned_design(150, 40, 1e4, 0xB1);
    let beta_star = make_planted_beta(40, 6, true, 0xB2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xB3);
    let alpha = 0.05;
    let beta_hat = fit_fista(&m, &y, alpha, false);
    let (active, inactive) = kkt_residual_lasso(&m, &y, &beta_hat, alpha);
    assert!(
        active < 1e-3,
        "active KKT residual {active:.3e} above 1e-3 (ill-conditioned)"
    );
    assert!(
        inactive < 1e-3,
        "inactive KKT excess {inactive:.3e} above 1e-3 (ill-conditioned)"
    );
}

#[test]
fn fista_kkt_underdetermined() {
    let m = make_design(30, 80, 0xC1);
    let beta_star = make_planted_beta(80, 5, true, 0xC2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xC3);
    let alpha = 0.05;
    let beta_hat = fit_fista(&m, &y, alpha, false);
    let (active, inactive) = kkt_residual_lasso(&m, &y, &beta_hat, alpha);
    assert!(active < 1e-6, "active KKT residual {active:.3e}");
    assert!(inactive < 1e-6, "inactive KKT excess {inactive:.3e}");
}

#[test]
fn fista_kkt_overdetermined() {
    let m = make_design(400, 20, 0xD1);
    let beta_star = make_planted_beta(20, 4, true, 0xD2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xD3);
    let alpha = 0.05;
    let beta_hat = fit_fista(&m, &y, alpha, false);
    let (active, inactive) = kkt_residual_lasso(&m, &y, &beta_hat, alpha);
    assert!(active < 1e-6, "active KKT residual {active:.3e}");
    assert!(inactive < 1e-6, "inactive KKT excess {inactive:.3e}");
}

#[test]
fn fista_kkt_at_alpha_max_returns_exact_zero() {
    // α = α_max ≡ ‖Mᵀy‖∞ / m is the threshold below which a non-zero
    // solution becomes optimal; at exactly α_max, β̂ = 0 satisfies KKT.
    use cs_dmri::solver::alpha_max;
    let m = make_design(120, 35, 0xE1);
    let beta_star = make_planted_beta(35, 5, true, 0xE2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xE3);
    let alpha = alpha_max(&m, &y);
    let beta_hat = fit_fista(&m, &y, alpha, false);
    let max_abs = beta_hat.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_abs < 1e-12,
        "FISTA at α=α_max returned non-zero coefficients: max|β̂|={max_abs:.3e}"
    );
}

#[test]
fn nonneg_fista_kkt_and_nonnegativity() {
    let m = make_design(180, 45, 0xF1);
    // Planted-positive β so the non-negative constraint is not active for
    // the true solution; FISTA should still satisfy NN-KKT.
    let beta_star = make_planted_beta(45, 7, false, 0xF2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xF3);
    let alpha = 0.05;
    let beta_hat = fit_fista(&m, &y, alpha, true);
    let min_b = beta_hat.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    assert!(
        min_b >= -1e-12,
        "non-negative FISTA returned negative coefficient: min={min_b:.3e}"
    );
    let (active, inactive) = kkt_residual_nonneg_lasso(&m, &y, &beta_hat, alpha);
    assert!(
        active < 1e-6,
        "NN-LASSO active KKT residual {active:.3e}"
    );
    assert!(
        inactive < 1e-6,
        "NN-LASSO inactive KKT excess {inactive:.3e}"
    );
}

#[test]
fn nonneg_fista_kkt_with_active_constraint() {
    // Plant a β* with mixed signs but fit non-negative — the constraint will
    // bind on negative-true coordinates. Tests the inactive-side KKT band.
    let m = make_design(200, 50, 0x11);
    let beta_star = make_planted_beta(50, 8, true, 0x12);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0x13);
    let alpha = 0.05;
    let beta_hat = fit_fista(&m, &y, alpha, true);
    let min_b = beta_hat.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    assert!(min_b >= -1e-12, "min coef {min_b:.3e}");
    let (active, inactive) = kkt_residual_nonneg_lasso(&m, &y, &beta_hat, alpha);
    assert!(active < 1e-5, "active KKT residual {active:.3e}");
    assert!(inactive < 1e-5, "inactive KKT excess {inactive:.3e}");
}

/// Tikhonov solver returns the closed-form minimizer of
/// `‖Mβ − y‖² + λ_n ⟨β, D_n β⟩ + λ_l ⟨β, D_l β⟩`. The minimizer satisfies
/// `(MᵀM + λ_n D_n + λ_l D_l) β = Mᵀy`. We assert the relative residual of
/// that linear system is at machine precision.
#[test]
fn tikhonov_normal_equations_well_conditioned() {
    // Use a real SHORE basis + synthetic gradient table so D_n / D_l are the
    // diagonals the production code actually sees.
    let basis = ShoreBasis::new(6, 700.0);
    let gtab = synthetic_gtab(60);
    let design = basis.design_matrix(&gtab);
    let reg = basis.regularization();
    let n = basis.n_coeffs();

    let beta_star = make_planted_beta(n, 4, true, 0x21);
    let y = make_noisy_signal(&design, &beta_star, 0.01, 0x22);

    let lambda_n = 1e-2;
    let lambda_l = 1e-2;
    let solver = TikhonovSolver::new(design.clone(), &reg, lambda_n, lambda_l);
    let problem = Problem {
        design: &design,
        signal: &y,
    };
    let (coef, _) = solver.fit(&problem);
    let rel_residual = tikhonov_normal_residual(&design, &y, &coef, lambda_n, lambda_l, &reg);
    assert!(
        rel_residual < 1e-10,
        "Tikhonov normal-equation residual {rel_residual:.3e} above 1e-10"
    );
}

#[test]
fn tikhonov_normal_equations_small_lambda() {
    let basis = ShoreBasis::new(6, 700.0);
    let gtab = synthetic_gtab(80);
    let design = basis.design_matrix(&gtab);
    let reg = basis.regularization();
    let n = basis.n_coeffs();

    let beta_star = make_planted_beta(n, 6, true, 0x31);
    let y = make_noisy_signal(&design, &beta_star, 0.005, 0x32);

    let lambda_n = 1e-8;
    let lambda_l = 1e-8;
    let solver = TikhonovSolver::new(design.clone(), &reg, lambda_n, lambda_l);
    let problem = Problem {
        design: &design,
        signal: &y,
    };
    let (coef, _) = solver.fit(&problem);
    let rel_residual = tikhonov_normal_residual(&design, &y, &coef, lambda_n, lambda_l, &reg);
    assert!(
        rel_residual < 1e-9,
        "Tikhonov small-λ residual {rel_residual:.3e} above 1e-9"
    );
}

/// Build a small synthetic gradient table (one b0 + three shells of 30
/// directions each via golden-angle spiral). Mirrors `tests/roundtrip.rs`'s
/// approach but kept inline here so the optimization tests don't depend on
/// the roundtrip module's helpers.
fn synthetic_gtab(per_shell: usize) -> GradientTable {
    let mut bvals = vec![0.0_f64];
    let mut bvecs = vec![[0.0_f64, 0.0, 0.0]];
    for &b in &[1000.0_f64, 2000.0, 3000.0] {
        for i in 0..per_shell {
            let phi = std::f64::consts::PI * (1.0 + 5.0_f64.sqrt());
            let z = 1.0 - (2.0 * i as f64 + 1.0) / (per_shell as f64);
            let r = (1.0 - z * z).max(0.0).sqrt();
            let theta = phi * i as f64;
            bvecs.push([r * theta.cos(), r * theta.sin(), z]);
            bvals.push(b);
        }
    }
    GradientTable::new(bvals, bvecs, Some(0.04), Some(0.012), Some(0.04)).unwrap()
}
