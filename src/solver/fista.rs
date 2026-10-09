// SPDX-License-Identifier: MIT OR Apache-2.0
//! L1-regularized least squares via FISTA (Beck & Teboulle, 2009).
//!
//! Solves: argmin_c (1/(2n)) ‖Mc − S‖² + α ‖c‖₁
//!
//! The 1/(2n) scaling matches sklearn's `Lasso` / `LassoCV` objective so the
//! same `α` value produces the same fits as qsirecon's current implementation.

use nalgebra::{DMatrix, DVector};

use super::alpha::AlphaConfigurable;
use super::{FitDiagnostics, Problem, Solver};

/// FISTA L1-regularized least-squares solver.
#[derive(Clone)]
pub struct FistaSolver {
    /// Sparsity weight α matching sklearn's `Lasso(alpha=…)` convention.
    pub alpha: f64,
    /// Maximum FISTA iterations.
    pub max_iter: u32,
    /// Convergence tolerance on the relative coefficient change.
    pub tol: f64,
    /// If true, project coefficients onto the non-negative orthant after
    /// each soft-threshold (matches sklearn's `Lasso(positive=True)`).
    pub non_negative: bool,
    /// Cached Lipschitz constant of the gradient of the smooth part:
    /// L = ‖M‖₂² / n_samples.
    lipschitz: f64,
    /// (1/n_samples) Mᵀ M, used in the gradient computation.
    /// Not stored — we recompute MᵀM·c per iteration via `M` directly to
    /// keep memory down and let nalgebra's BLAS-style ops dominate.
    design: DMatrix<f64>,
    mt: DMatrix<f64>,
    n_samples_inv: f64,
}

impl FistaSolver {
    pub fn new(design: DMatrix<f64>, alpha: f64, max_iter: u32, tol: f64, non_negative: bool) -> Self {
        let n_samples = design.nrows() as f64;
        let n_samples_inv = 1.0 / n_samples;
        // L = (1/n) · σ_max(M)² ; bound it by a Frobenius-norm proxy that's cheaper
        // and an upper bound on the spectral norm — slightly conservative step size,
        // but still convergent. (Frobenius >= spectral norm.)
        // For a tighter bound we compute the full SVD once.
        let svd = design.clone().svd(false, false);
        let sigma_max = svd
            .singular_values
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        let lipschitz = sigma_max * sigma_max * n_samples_inv;
        let mt = design.transpose();
        Self {
            alpha,
            max_iter,
            tol,
            non_negative,
            lipschitz: lipschitz.max(f64::EPSILON),
            design,
            mt,
            n_samples_inv,
        }
    }

    pub fn design(&self) -> &DMatrix<f64> {
        &self.design
    }

    /// Override the warm-start coefficients used as the FISTA initial point.
    pub fn fit_with_warm_start(
        &self,
        problem: &Problem<'_>,
        warm_start: Option<&DVector<f64>>,
    ) -> (DVector<f64>, FitDiagnostics) {
        let n_coeffs = self.design.ncols();
        let mut x = match warm_start {
            Some(w) if w.len() == n_coeffs => w.clone(),
            _ => DVector::<f64>::zeros(n_coeffs),
        };
        let mut y = x.clone();
        let mut t = 1.0_f64;
        // Step size 1/L ; gradient of f(c) = (1/(2n)) ‖Mc - s‖²:
        //   ∇f(c) = (1/n) · Mᵀ (Mc - s)
        let step = 1.0 / self.lipschitz;
        let threshold = step * self.alpha;

        let mut iters = 0_u32;
        let mut converged = false;
        for it in 0..self.max_iter {
            // gradient at y: g = (1/n) · Mᵀ (M y − s)
            let m_y = &self.design * &y;
            let resid = &m_y - problem.signal;
            let grad = &self.mt * &resid * self.n_samples_inv;

            // proximal step: x_next = soft_threshold(y - step · grad, threshold)
            let candidate = &y - step * grad;
            let mut x_next = candidate;
            for i in 0..x_next.len() {
                let v = x_next[i];
                let s = if v > threshold {
                    v - threshold
                } else if v < -threshold {
                    v + threshold
                } else {
                    0.0
                };
                x_next[i] = if self.non_negative { s.max(0.0) } else { s };
            }

            // momentum
            let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
            let momentum = (t - 1.0) / t_next;
            let y_next = &x_next + momentum * (&x_next - &x);

            // convergence: relative change in x
            let dx = (&x_next - &x).norm();
            let denom = x_next.norm().max(1e-12);
            let rel_change = dx / denom;

            x = x_next;
            y = y_next;
            t = t_next;
            iters = it + 1;
            if rel_change < self.tol {
                converged = true;
                break;
            }
        }

        let residual = problem.signal - &self.design * &x;
        let resid_l2 = residual.norm();

        (
            x,
            FitDiagnostics {
                iterations: iters,
                residual_l2: resid_l2,
                converged,
                alpha: self.alpha,
                regularization_kind: 1,
            },
        )
    }
}

impl Solver for FistaSolver {
    fn fit(&self, problem: &Problem<'_>) -> (DVector<f64>, FitDiagnostics) {
        self.fit_with_warm_start(problem, None)
    }
}

impl AlphaConfigurable for FistaSolver {
    fn alpha(&self) -> f64 {
        self.alpha
    }

    fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Walk the supplied (descending) α slice, warm-starting each fit from the
    /// previous α's coefficient vector. The first fit is cold-started from
    /// zero; for α near α_max that's a no-op since the optimum is exactly
    /// zero, and subsequent αs converge in 10–30 iters instead of 100–300.
    fn fit_path(
        &mut self,
        problem: &Problem<'_>,
        alphas_descending: &[f64],
    ) -> Vec<(DVector<f64>, FitDiagnostics)> {
        let mut warm: Option<DVector<f64>> = None;
        let mut out = Vec::with_capacity(alphas_descending.len());
        for &a in alphas_descending {
            self.alpha = a;
            let (coef, diag) = self.fit_with_warm_start(problem, warm.as_ref());
            warm = Some(coef.clone());
            out.push((coef, diag));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::DMatrix;

    #[test]
    fn recovers_sparse_signal() {
        // Slightly over-determined sparse problem.
        let m = DMatrix::<f64>::from_row_slice(
            8,
            4,
            &[
                1.0, 0.0, 0.5, -0.2, 0.0, 1.0, -0.1, 0.3, -0.3, 0.4, 1.0, 0.0, 0.2, -0.1, 0.0, 1.0,
                0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.1, 0.2, -0.3,
                0.4,
            ],
        );
        let true_c = DVector::<f64>::from_row_slice(&[2.0, 0.0, -1.5, 0.0]);
        let s = &m * &true_c;

        let solver = FistaSolver::new(m.clone(), 1e-3, 2000, 1e-9, false);
        let problem = Problem {
            design: &m,
            signal: &s,
        };
        let (coef, diag) = solver.fit(&problem);
        // With a small alpha and a clean problem we should recover something close.
        assert_abs_diff_eq!(coef[0], 2.0, epsilon = 0.05);
        assert!(coef[1].abs() < 0.05);
        assert_abs_diff_eq!(coef[2], -1.5, epsilon = 0.05);
        assert!(coef[3].abs() < 0.05);
        assert_eq!(diag.regularization_kind, 1);
    }

    #[test]
    fn zero_alpha_matches_least_squares() {
        let m = DMatrix::<f64>::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0]);
        let true_c = DVector::<f64>::from_row_slice(&[2.0, -3.0]);
        let s = &m * &true_c;
        let solver = FistaSolver::new(m.clone(), 0.0, 5000, 1e-12, false);
        let problem = Problem {
            design: &m,
            signal: &s,
        };
        let (coef, _) = solver.fit(&problem);
        assert_abs_diff_eq!(coef[0], 2.0, epsilon = 1e-3);
        assert_abs_diff_eq!(coef[1], -3.0, epsilon = 1e-3);
    }
}
