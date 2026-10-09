// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sparse-recovery tests on planted-support synthetic problems.
//!
//! These tests probe the *statistical* correctness of LASSO + path-BIC: given
//! a true k-sparse β* in a Gaussian noise model, recovery should improve as
//! noise decreases and BIC should pick a support of the right rough size.
//! They are looser than the KKT tests by design — recovery curves are only
//! exact in expectation, so assertions allow a few false positives or one
//! noise-floor miss.
//!
//! These complement [optimization_kkt.rs](optimization_kkt.rs): KKT proves
//! the optimizer hit the true LASSO minimum; recovery proves that minimum is
//! statistically meaningful for the planted problem.

mod common;

use cs_dmri::solver::alpha::{alpha_max, bic};
use cs_dmri::solver::fista::FistaSolver;
use cs_dmri::solver::{AlphaPath, AlphaStrategy, Problem, Solver};

use common::{
    make_design, make_noisy_signal, make_planted_beta, support_diff, support_of,
};

fn fit_fista(
    design: &nalgebra::DMatrix<f64>,
    signal: &nalgebra::DVector<f64>,
    alpha: f64,
) -> nalgebra::DVector<f64> {
    let solver = FistaSolver::new(design.clone(), alpha, 50_000, 1e-9, false);
    let problem = Problem { design, signal };
    let (coef, _) = solver.fit(&problem);
    coef
}

#[test]
fn fixed_alpha_recovers_planted_support() {
    // m=400, n=80, k=10. With Gaussian iid design and σ=0.05, the strong-
    // signal regime: support recovery should be exact-ish.
    let m = make_design(400, 80, 0xA1);
    let beta_star = make_planted_beta(80, 10, true, 0xA2);
    let y = make_noisy_signal(&m, &beta_star, 0.05, 0xA3);
    // α as a fraction of α_max (= ‖Mᵀy‖∞ / m in cs_dmri's sklearn-style
    // convention). 10% of α_max sits comfortably in the "active recovery"
    // regime — small enough to keep the planted support, large enough to
    // suppress most false positives.
    let alpha = 0.1 * alpha_max(&m, &y);
    let beta_hat = fit_fista(&m, &y, alpha);

    let true_supp = support_of(&beta_star, 1e-9);
    let est_supp = support_of(&beta_hat, 1e-9);
    let false_neg = support_diff(&beta_star, &beta_hat, 1e-9);
    let false_pos = support_diff(&beta_hat, &beta_star, 1e-9);

    assert!(
        false_neg.len() <= 1,
        "expected ≤1 false negative on the planted support; got {} (true={true_supp:?}, est={est_supp:?})",
        false_neg.len()
    );
    assert!(
        false_pos.len() <= 4,
        "expected ≤4 false positives; got {} (false_pos={false_pos:?})",
        false_pos.len()
    );

    // On the true support, the recovered magnitudes should be in the right
    // ballpark (LASSO shrinks slightly toward zero).
    let mut max_err_on_support = 0.0_f64;
    for &i in &true_supp {
        let err = (beta_hat[i] - beta_star[i]).abs();
        if err > max_err_on_support {
            max_err_on_support = err;
        }
    }
    assert!(
        max_err_on_support < 0.4,
        "max coefficient error on true support {max_err_on_support:.3} above 0.4"
    );
}

#[test]
fn path_bic_picks_correct_support_size() {
    let design = make_design(400, 80, 0xB1);
    let beta_star = make_planted_beta(80, 10, true, 0xB2);
    let y = make_noisy_signal(&design, &beta_star, 0.05, 0xB3);

    let mut solver = FistaSolver::new(design.clone(), 1.0, 50_000, 1e-9, false);
    let strategy = AlphaStrategy::PathBic {
        path: AlphaPath { n: 25, eps: 1e-3 },
    };
    let problem = Problem {
        design: &design,
        signal: &y,
    };
    let result = strategy.resolve(&mut solver, &problem);
    let chosen_support = result.coef.iter().filter(|c| **c != 0.0).count();

    // path-BIC tends to pick *slightly* smaller-than-truth supports because
    // BIC's k·log(n) penalty discourages weak coordinates. Allow a wide
    // [k − 4, k + 8] window so seed-sensitivity doesn't flake the test.
    assert!(
        (6..=18).contains(&chosen_support),
        "path-BIC chose support size {chosen_support}; expected 6..=18 (truth k=10)"
    );
}

#[test]
fn support_recovery_curve_is_monotone_in_noise() {
    // Loose smoke test: as σ grows, the number of correctly-recovered true
    // support coordinates should be non-increasing. Catches sign bugs in the
    // prox or in the gradient that would invert the noise/recovery relation.
    let design = make_design(400, 80, 0xC1);
    let beta_star = make_planted_beta(80, 10, true, 0xC2);
    let true_supp = support_of(&beta_star, 1e-9);

    let mut recovered_counts = Vec::new();
    for (i, &sigma) in [0.01_f64, 0.05, 0.1, 0.3].iter().enumerate() {
        let y = make_noisy_signal(&design, &beta_star, sigma, 0xC3 + i as u64);
        // α as a fixed fraction of α_max keeps the *relative* penalty
        // regime constant across noise levels; recovery degrades only
        // because of σ, not because of changing α.
        let alpha = 0.1 * alpha_max(&design, &y);
        let beta_hat = fit_fista(&design, &y, alpha);
        let est_supp = support_of(&beta_hat, 1e-9);
        let recovered = true_supp.iter().filter(|i| est_supp.contains(i)).count();
        recovered_counts.push((sigma, recovered));
    }
    eprintln!("recovery vs σ: {:?}", recovered_counts);
    // Allow one inversion (low-noise vs noisier may tie within ±1).
    let mut inversions = 0;
    for w in recovered_counts.windows(2) {
        if w[1].1 > w[0].1 {
            inversions += 1;
        }
    }
    assert!(
        inversions <= 1,
        "recovery vs σ not monotone: {recovered_counts:?}"
    );
}

#[test]
fn bic_score_formula_matches_definition() {
    // Pin the BIC convention: BIC = n · ln(RSS / n) + ln(n) · k_nnz.
    // A regression here would silently shift the path-BIC chosen α and
    // break support-size assertions across the test suite — easier to fail
    // here at the formula level.
    let design = make_design(120, 30, 0xD1);
    let beta_star = make_planted_beta(30, 5, true, 0xD2);
    let y = make_noisy_signal(&design, &beta_star, 0.05, 0xD3);

    let alpha = 0.05;
    let beta_hat = fit_fista(&design, &y, alpha);
    let residual = &design * &beta_hat - &y;
    let rss = residual.norm_squared();
    let k_nnz = beta_hat.iter().filter(|c| **c != 0.0).count();
    let n = design.nrows();

    let manual = (n as f64) * (rss / n as f64).ln() + (n as f64).ln() * (k_nnz as f64);
    let from_lib = bic(n, rss, k_nnz);
    assert!(
        (manual - from_lib).abs() < 1e-9,
        "BIC formula drift: manual={manual}, lib={from_lib}"
    );
}
