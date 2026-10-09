// SPDX-License-Identifier: MIT OR Apache-2.0
//! Data-driven α selection for L1 solvers.
//!
//! `AlphaConfigurable` extends `Solver` with the ability to swap α and walk a
//! warm-started regularization path. `AlphaStrategy` picks one of three modes
//! per voxel: a fixed α, an α_max-relative ratio, or BIC-minimizing path
//! search. New criteria (AIC / discrepancy / SURE / L-curve) plug in as extra
//! enum variants without touching the trait.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use super::{FitDiagnostics, Problem, Solver};

/// L1 solvers that can be re-tuned to a different α and (optionally) walk a
/// warm-started regularization path.
pub trait AlphaConfigurable: Solver + Clone {
    fn alpha(&self) -> f64;
    fn set_alpha(&mut self, alpha: f64);

    /// Fit a sequence of α values. Default impl cold-starts each fit; path-
    /// aware solvers (FISTA) override to warm-start from the previous α's
    /// coefficient vector — typically a 5–20× speedup along a log-spaced path.
    ///
    /// `alphas_descending` is expected (but not required) to be in decreasing
    /// order; warm-starting only helps when α changes smoothly.
    fn fit_path(
        &mut self,
        problem: &Problem<'_>,
        alphas_descending: &[f64],
    ) -> Vec<(DVector<f64>, FitDiagnostics)> {
        alphas_descending
            .iter()
            .map(|&a| {
                self.set_alpha(a);
                self.fit(problem)
            })
            .collect()
    }
}

/// Smallest α at which `c = 0` is the optimum of
/// `(1/(2n)) ‖Mc − s‖² + α ‖c‖₁`.
///
/// Derivation: the KKT condition at c = 0 is `|(1/n) (Mᵀs)_j| ≤ α` for all j,
/// so `α_max = ‖Mᵀs‖∞ / n`.
pub fn alpha_max(design: &DMatrix<f64>, signal: &DVector<f64>) -> f64 {
    if design.nrows() == 0 {
        return 0.0;
    }
    let xty = design.transpose() * signal;
    xty.iter().fold(0.0_f64, |m, &x| m.max(x.abs())) / design.nrows() as f64
}

/// Log-spaced descending α path from `alpha_max` down to `alpha_max · eps`,
/// inclusive of both endpoints, with `n` entries (n ≥ 2).
pub fn log_path(alpha_max: f64, n: usize, eps: f64) -> Vec<f64> {
    assert!(n >= 2, "log_path requires n ≥ 2");
    assert!(alpha_max > 0.0, "log_path requires positive alpha_max");
    assert!(eps > 0.0 && eps < 1.0, "log_path requires 0 < eps < 1");
    let log_max = alpha_max.ln();
    let log_min = (alpha_max * eps).ln();
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (log_max + t * (log_min - log_max)).exp()
        })
        .collect()
}

/// Path layout for `AlphaStrategy::PathBic`. Independent of any specific α
/// scale — `α_max` is computed per voxel.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AlphaPath {
    pub n: usize,
    pub eps: f64,
}

impl AlphaPath {
    pub fn new(n: usize, eps: f64) -> Self {
        Self { n, eps }
    }
}

impl Default for AlphaPath {
    fn default() -> Self {
        Self { n: 20, eps: 1e-3 }
    }
}

/// BIC for one point on the regularization path. `n_grads` is the number of
/// observations (signal length); `k_nonzero` counts strictly-nonzero
/// coefficient entries (FISTA's soft-threshold returns exact zeros, so no
/// tolerance is needed).
///
/// `BIC = n · ln(RSS / n) + ln(n) · k`. RSS is clamped from below by a small
/// floor to keep the log finite when a fit happens to be near-exact.
#[inline]
pub fn bic(n_grads: usize, rss: f64, k_nonzero: usize) -> f64 {
    let n = n_grads as f64;
    let rss_clamped = rss.max(f64::MIN_POSITIVE);
    n * (rss_clamped / n).ln() + n.ln() * k_nonzero as f64
}

/// Pick the α index along the path that minimizes BIC. Ties broken by the
/// largest α (most parsimonious).
pub fn bic_argmin(
    n_grads: usize,
    path_results: &[(DVector<f64>, FitDiagnostics)],
    alphas_descending: &[f64],
) -> usize {
    debug_assert_eq!(path_results.len(), alphas_descending.len());
    let mut best_idx = 0_usize;
    let mut best_bic = f64::INFINITY;
    for (i, ((coef, diag), &alpha)) in path_results.iter().zip(alphas_descending).enumerate() {
        let _ = alpha;
        let k = coef.iter().filter(|c| **c != 0.0).count();
        let rss = diag.residual_l2 * diag.residual_l2;
        let b = bic(n_grads, rss, k);
        // Strictly less so that on ties we keep the earlier (larger-α) entry,
        // which is more parsimonious since the path is descending.
        if b < best_bic {
            best_bic = b;
            best_idx = i;
        }
    }
    best_idx
}

/// L2-residual-anchored α selector. Pick the largest α (smallest index along
/// the descending path) whose residual SS is within `(1 + slack) · rss_l2`.
/// If no point along the path satisfies that bound, fall back to the α with
/// minimum RSS — the closest-to-L2-fit α we measured.
///
/// This is a "1-SE rule"-style selector adapted to the SHORE / dMRI setting:
/// instead of letting BIC trade fit for sparsity (which mis-fires on high-b
/// or low-FA voxels — see the cs-dmri sampling-scheme harness), we bound
/// fit looseness directly against a Tikhonov reference and keep the most
/// parsimonious fit that meets the bound.
pub fn l2_anchored_argmax(
    path_results: &[(DVector<f64>, FitDiagnostics)],
    rss_l2: f64,
    slack: f64,
) -> usize {
    debug_assert!(slack >= 0.0, "slack must be non-negative");
    let threshold = (1.0 + slack) * rss_l2;
    // Path is α-descending; first acceptable index = largest acceptable α.
    for (i, (_, diag)) in path_results.iter().enumerate() {
        let rss = diag.residual_l2 * diag.residual_l2;
        if rss <= threshold {
            return i;
        }
    }
    // Fallback: argmin RSS along the path.
    let mut best_idx = 0_usize;
    let mut best_rss = f64::INFINITY;
    for (i, (_, diag)) in path_results.iter().enumerate() {
        let rss = diag.residual_l2 * diag.residual_l2;
        if rss < best_rss {
            best_rss = rss;
            best_idx = i;
        }
    }
    best_idx
}

/// One-voxel result returned by `AlphaStrategy::resolve`.
pub struct VoxelAlphaResult {
    pub coef: DVector<f64>,
    pub alpha: f64,
    pub residual_l2: f64,
    pub iterations: u32,
    pub converged: bool,
    /// BIC at the chosen α. `0.0` for the degenerate (all-zero-signal) voxel.
    pub bic: f64,
}

impl VoxelAlphaResult {
    fn zero(n_coeffs: usize) -> Self {
        Self {
            coef: DVector::<f64>::zeros(n_coeffs),
            alpha: 0.0,
            residual_l2: 0.0,
            iterations: 0,
            converged: true,
            bic: 0.0,
        }
    }
}

/// Per-voxel α-selection strategy. New criteria slot in as extra variants
/// (AIC, discrepancy principle, L-curve, SURE, …) without API churn.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum AlphaStrategy {
    Fixed { alpha: f64 },
    AlphaMaxRatio { ratio: f64 },
    PathBic { path: AlphaPath },
    /// Walk the same α path as `PathBic`, but per voxel pick the largest α
    /// whose RSS is within `(1+slack) · RSS_L2` of a Tikhonov reference fit.
    /// Falls back to argmin(RSS) if no path α meets the slack constraint.
    /// Requires the orchestrator to supply a reference L2 solver.
    PathL2Anchored { path: AlphaPath, slack: f64 },
}

impl AlphaStrategy {
    pub fn resolve<S: AlphaConfigurable>(
        &self,
        solver: &mut S,
        problem: &Problem<'_>,
    ) -> VoxelAlphaResult {
        let n_coeffs = problem.design.ncols();
        match *self {
            AlphaStrategy::Fixed { alpha } => {
                solver.set_alpha(alpha);
                let (coef, diag) = solver.fit(problem);
                let k = coef.iter().filter(|c| **c != 0.0).count();
                let rss = diag.residual_l2 * diag.residual_l2;
                let b = bic(problem.signal.len(), rss, k);
                VoxelAlphaResult {
                    coef,
                    alpha,
                    residual_l2: diag.residual_l2,
                    iterations: diag.iterations,
                    converged: diag.converged,
                    bic: b,
                }
            }
            AlphaStrategy::AlphaMaxRatio { ratio } => {
                let amax = alpha_max(problem.design, problem.signal);
                if amax == 0.0 {
                    return VoxelAlphaResult::zero(n_coeffs);
                }
                let alpha = amax * ratio;
                solver.set_alpha(alpha);
                let (coef, diag) = solver.fit(problem);
                let k = coef.iter().filter(|c| **c != 0.0).count();
                let rss = diag.residual_l2 * diag.residual_l2;
                let b = bic(problem.signal.len(), rss, k);
                VoxelAlphaResult {
                    coef,
                    alpha,
                    residual_l2: diag.residual_l2,
                    iterations: diag.iterations,
                    converged: diag.converged,
                    bic: b,
                }
            }
            AlphaStrategy::PathBic { path } => {
                let amax = alpha_max(problem.design, problem.signal);
                if amax == 0.0 {
                    return VoxelAlphaResult::zero(n_coeffs);
                }
                let alphas = log_path(amax, path.n, path.eps);
                let results = solver.fit_path(problem, &alphas);
                let idx = bic_argmin(problem.signal.len(), &results, &alphas);
                let (coef, diag) = results.into_iter().nth(idx).unwrap();
                let k = coef.iter().filter(|c| **c != 0.0).count();
                let rss = diag.residual_l2 * diag.residual_l2;
                let b = bic(problem.signal.len(), rss, k);
                VoxelAlphaResult {
                    coef,
                    alpha: alphas[idx],
                    residual_l2: diag.residual_l2,
                    iterations: diag.iterations,
                    converged: diag.converged,
                    bic: b,
                }
            }
            // PathL2Anchored requires a reference L2 solver; orchestrator
            // must call `resolve_with_l2_ref` instead. Treat a stray
            // `resolve` call as a configuration bug.
            AlphaStrategy::PathL2Anchored { .. } => {
                panic!(
                    "AlphaStrategy::PathL2Anchored requires a Tikhonov reference; \
                     call resolve_with_l2_ref(..) instead of resolve(..)"
                );
            }
        }
    }

    /// Like `resolve` but accepts a reference L2 solver used by the
    /// `PathL2Anchored` variant. Other variants ignore the L2 reference and
    /// behave identically to `resolve`.
    pub fn resolve_with_l2_ref<S: AlphaConfigurable, L: Solver>(
        &self,
        solver: &mut S,
        l2_ref: &L,
        problem: &Problem<'_>,
    ) -> VoxelAlphaResult {
        let n_coeffs = problem.design.ncols();
        match *self {
            AlphaStrategy::PathL2Anchored { path, slack } => {
                let amax = alpha_max(problem.design, problem.signal);
                if amax == 0.0 {
                    return VoxelAlphaResult::zero(n_coeffs);
                }
                let alphas = log_path(amax, path.n, path.eps);
                let results = solver.fit_path(problem, &alphas);
                let (_, l2_diag) = l2_ref.fit(problem);
                let rss_l2 = l2_diag.residual_l2 * l2_diag.residual_l2;
                let idx = l2_anchored_argmax(&results, rss_l2, slack);
                let (coef, diag) = results.into_iter().nth(idx).unwrap();
                let k = coef.iter().filter(|c| **c != 0.0).count();
                let rss = diag.residual_l2 * diag.residual_l2;
                let b = bic(problem.signal.len(), rss, k);
                VoxelAlphaResult {
                    coef,
                    alpha: alphas[idx],
                    residual_l2: diag.residual_l2,
                    iterations: diag.iterations,
                    converged: diag.converged,
                    bic: b,
                }
            }
            _ => self.resolve(solver, problem),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::fista::FistaSolver;
    use approx::assert_abs_diff_eq;
    use nalgebra::DMatrix;

    fn toy_problem() -> (DMatrix<f64>, DVector<f64>) {
        // A modestly over-determined problem with a known sparse generator.
        let m = DMatrix::<f64>::from_row_slice(
            10,
            5,
            &[
                1.0, 0.0, 0.5, -0.2, 0.1, 0.0, 1.0, -0.1, 0.3, -0.4, -0.3, 0.4, 1.0, 0.0, 0.2,
                0.2, -0.1, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, -0.2, -0.5, 0.5, -0.5, 0.5, 0.3,
                0.5, -0.5, 0.5, -0.5, -0.1, 0.1, 0.2, -0.3, 0.4, 0.0, 0.7, 0.0, 0.1, -0.2, 0.6,
                -0.4, 0.3, 0.2, -0.1, 0.0,
            ],
        );
        let true_c = DVector::<f64>::from_row_slice(&[2.0, 0.0, -1.5, 0.0, 0.0]);
        let s = &m * &true_c;
        (m, s)
    }

    #[test]
    fn alpha_max_kkt_zeros_solution() {
        let (m, s) = toy_problem();
        let amax = alpha_max(&m, &s);
        // At alpha = alpha_max, the FISTA optimum must be exactly zero.
        let solver = FistaSolver::new(m.clone(), amax, 500, 1e-9, false);
        let (c, _) = solver.fit(&Problem {
            design: &m,
            signal: &s,
        });
        for v in c.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn alpha_just_below_max_escapes_zero() {
        let (m, s) = toy_problem();
        let amax = alpha_max(&m, &s);
        let solver = FistaSolver::new(m.clone(), amax * 0.5, 2000, 1e-9, false);
        let (c, _) = solver.fit(&Problem {
            design: &m,
            signal: &s,
        });
        let nz = c.iter().filter(|v| **v != 0.0).count();
        assert!(nz > 0, "expected nonzero solution below alpha_max, got c={c:?}");
    }

    #[test]
    fn log_path_is_monotone_descending_with_correct_endpoints() {
        let p = log_path(10.0, 5, 1e-3);
        assert_eq!(p.len(), 5);
        assert_abs_diff_eq!(p[0], 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p[4], 10.0 * 1e-3, epsilon = 1e-12);
        for w in p.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn warm_started_path_matches_cold_start() {
        // The warm-started fit_path should converge to the same point as
        // independent cold-start fits at each alpha (within FISTA tolerance).
        let (m, s) = toy_problem();
        let amax = alpha_max(&m, &s);
        let alphas = log_path(amax, 6, 1e-2);
        let problem = Problem {
            design: &m,
            signal: &s,
        };

        let mut warm_solver = FistaSolver::new(m.clone(), amax, 5000, 1e-10, false);
        let warm_results = warm_solver.fit_path(&problem, &alphas);

        for (i, &a) in alphas.iter().enumerate() {
            let cold = FistaSolver::new(m.clone(), a, 5000, 1e-10, false);
            let (cold_coef, _) = cold.fit(&problem);
            let warm_coef = &warm_results[i].0;
            let diff = (warm_coef - &cold_coef).norm();
            let scale = warm_coef.norm().max(cold_coef.norm()).max(1e-12);
            assert!(
                diff / scale < 1e-3,
                "warm vs cold disagree at alpha={a}: warm={warm_coef:?}, cold={cold_coef:?}, rel={}",
                diff / scale
            );
        }
    }

    #[test]
    fn path_bic_recovers_sparse_support_under_noise() {
        // Build a very small synthetic problem with a known 2-of-5 sparse
        // ground truth, fit via PathBic, and assert the chosen support
        // overlaps the truth.
        let (m, s_clean) = toy_problem();
        // Add small Gaussian-like noise (deterministic): tiny perturbations
        // that should not flip support choice.
        let noise = DVector::<f64>::from_row_slice(&[
            0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.01, -0.01,
        ]);
        let s = &s_clean + noise;
        let mut solver = FistaSolver::new(m.clone(), 0.0, 5000, 1e-10, false);
        let strategy = AlphaStrategy::PathBic {
            path: AlphaPath { n: 30, eps: 1e-4 },
        };
        let result = strategy.resolve(&mut solver, &Problem { design: &m, signal: &s });
        let true_support = [0_usize, 2];
        let recovered: Vec<usize> = result
            .coef
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(i, _)| i)
            .collect();
        let overlap = true_support
            .iter()
            .filter(|&&i| recovered.contains(&i))
            .count();
        assert_eq!(
            overlap,
            true_support.len(),
            "BIC missed true support: recovered={recovered:?}"
        );
        assert!(result.alpha > 0.0);
    }
}
