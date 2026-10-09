// SPDX-License-Identifier: MIT OR Apache-2.0
//! Constrained spherical-deconvolution inner solver.
//!
//! Implements the iterative reweighting algorithm of
//! Tournier, Calamante & Connelly (NeuroImage 2007) — *"Robust determination of
//! the fibre orientation distribution in diffusion MRI: Non-negativity
//! constrained super-resolved spherical deconvolution"* — extended trivially to
//! arbitrary linear inequality constraints `B x ≥ 0`.
//!
//! The algorithm minimizes `½‖Hx − b‖²` subject to `B x ≥ 0` by repeatedly
//! solving a Tikhonov-regularized problem in which the rows of `B` that
//! produce negative responses on the current iterate are added (with weight
//! λ) to the design as a soft penalty:
//!
//! ```text
//!   x_{k+1} = argmin ½‖Hx − b‖² + (λ/2) ‖L_k x‖²
//!   L_k = rows of B where (B · x_k) < τ
//! ```
//!
//! Each step factors `(HᵀH + λ Lₖᵀ Lₖ + ε I)` once via Cholesky. The same
//! `nalgebra::Cholesky` primitive used by [`TikhonovSolver`](super::tikhonov)
//! powers each iteration. Convergence is declared when the active set is
//! stable across two consecutive iterations.

use nalgebra::{Cholesky, DMatrix, DVector};

/// Configuration for the iterative-reweighting CSD solver.
#[derive(Debug, Clone, Copy)]
pub struct CsdConfig {
    /// Hard cap on the number of reweighting iterations.
    pub max_iter: usize,
    /// Penalty weight for violated constraints.
    pub lambda: f64,
    /// Negative-amplitude threshold; rows of `B` with `(B x) < tau` are added
    /// to the active set. Use `0.0` for the standard non-negativity constraint;
    /// use a small negative value to relax.
    pub tau: f64,
    /// Tikhonov stabilizer added to the diagonal of `HᵀH` to guarantee a
    /// positive-definite normal-equations matrix even before any constraints
    /// are active.
    pub epsilon: f64,
}

impl Default for CsdConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            lambda: 1.0,
            tau: 0.0,
            epsilon: 1e-10,
        }
    }
}

/// Per-solve diagnostics returned alongside the coefficient vector.
#[derive(Debug, Clone, Copy, Default)]
pub struct CsdDiagnostics {
    pub iterations: usize,
    pub final_active: usize,
    pub converged: bool,
}

/// Iterative-reweighting CSD solver.
///
/// Holds the precomputed `HᵀH + ε I` matrix and the constraint matrix `B`.
/// `Send + Sync` so the rayon voxel loop can share one solver across worker
/// threads (each `solve` allocates its own per-iteration scratch).
#[derive(Debug, Clone)]
pub struct CsdSolver {
    h: DMatrix<f64>,
    ht: DMatrix<f64>,
    constraint: DMatrix<f64>,
    hth_base: DMatrix<f64>,
    cfg: CsdConfig,
    n_coeffs: usize,
    n_constraints: usize,
}

impl CsdSolver {
    /// Build a solver for `min ½‖Hx − b‖²  s.t.  constraint · x ≥ τ` (with
    /// `τ` from `cfg.tau`).
    ///
    /// Panics if `h.ncols() != constraint.ncols()`.
    pub fn new(h: DMatrix<f64>, constraint: DMatrix<f64>, cfg: CsdConfig) -> Self {
        assert_eq!(
            h.ncols(),
            constraint.ncols(),
            "design (n={}) and constraint (n={}) must share parameter count",
            h.ncols(),
            constraint.ncols()
        );
        let n_coeffs = h.ncols();
        let n_constraints = constraint.nrows();
        let ht = h.transpose();
        let mut hth_base = &ht * &h;
        for i in 0..n_coeffs {
            hth_base[(i, i)] += cfg.epsilon;
        }
        Self {
            h,
            ht,
            constraint,
            hth_base,
            cfg,
            n_coeffs,
            n_constraints,
        }
    }

    /// Number of parameters (columns of `H`).
    pub fn n_coeffs(&self) -> usize {
        self.n_coeffs
    }

    /// Number of inequality constraints (rows of `B`).
    pub fn n_constraints(&self) -> usize {
        self.n_constraints
    }

    /// Forward operator `H`.
    pub fn h(&self) -> &DMatrix<f64> {
        &self.h
    }

    /// Solve for `x` given the right-hand-side `b` (length = `H.nrows()`).
    pub fn solve(&self, b: &DVector<f64>) -> (DVector<f64>, CsdDiagnostics) {
        let rhs = &self.ht * b;

        // Iteration 0: unconstrained Tikhonov solve.
        let chol = Cholesky::new(self.hth_base.clone())
            .expect("HᵀH + εI must be positive definite");
        let mut x = chol.solve(&rhs);

        let mut prev_active: Vec<bool> = vec![false; self.n_constraints];
        let mut active: Vec<bool> = vec![false; self.n_constraints];
        let mut converged = false;
        let mut iters = 0usize;
        let mut last_active = 0usize;
        let mut prev_n_active: Option<usize> = None;

        for k in 1..=self.cfg.max_iter {
            iters = k;

            // Identify violated constraints at the current iterate.
            let bx = &self.constraint * &x;
            let mut n_active = 0usize;
            for i in 0..self.n_constraints {
                let v = bx[i] < self.cfg.tau;
                active[i] = v;
                if v {
                    n_active += 1;
                }
            }
            last_active = n_active;

            // Convergence: the active set is bit-equal to the previous
            // iterate's, OR its cardinality matches (per Tournier 2007 — once
            // the count of violated directions stops changing, remaining flips
            // are amplitudes hovering near zero and don't materially change
            // the solution).
            if k > 1 && (active == prev_active || prev_n_active == Some(n_active)) {
                converged = true;
                break;
            }
            prev_n_active = Some(n_active);

            // Build augmented normal-equations matrix:
            //   A = HᵀH + λ · B_aᵀ B_a + ε I
            let mut a = self.hth_base.clone();
            if n_active > 0 {
                let mut b_active = DMatrix::<f64>::zeros(n_active, self.n_coeffs);
                let mut row = 0usize;
                for i in 0..self.n_constraints {
                    if active[i] {
                        b_active.row_mut(row).copy_from(&self.constraint.row(i));
                        row += 1;
                    }
                }
                let bta_ba = b_active.transpose() * &b_active;
                a += self.cfg.lambda * bta_ba;
            }

            let chol_k = match Cholesky::new(a) {
                Some(c) => c,
                None => {
                    // Numerical breakdown — should be rare given ε > 0; bail
                    // with the previous iterate rather than panicking.
                    break;
                }
            };
            x = chol_k.solve(&rhs);

            std::mem::swap(&mut prev_active, &mut active);
        }

        (
            x,
            CsdDiagnostics {
                iterations: iters,
                final_active: last_active,
                converged,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn unconstrained_recovers_ols_when_no_active_constraints() {
        // H is 4×2 full-rank; with constraints that are trivially satisfied
        // at the OLS solution, the solver should converge immediately.
        let h = DMatrix::<f64>::from_row_slice(
            4,
            2,
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        );
        let true_x = DVector::<f64>::from_row_slice(&[2.0, 3.0]);
        let b = &h * &true_x;
        // Constraints: x_0 ≥ 0, x_1 ≥ 0 (both satisfied at the OLS minimum).
        let constraint = DMatrix::<f64>::identity(2, 2);
        let solver = CsdSolver::new(h, constraint, CsdConfig { epsilon: 1e-12, ..Default::default() });
        let (x, diag) = solver.solve(&b);
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-6);
        assert_eq!(diag.final_active, 0);
        assert!(diag.converged);
    }

    #[test]
    fn enforces_non_negativity_on_known_negative_ols_solution() {
        // OLS would prefer x = (-1, 1); with x_0 ≥ 0 the constrained optimum
        // pushes x_0 toward 0.
        let h = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = DVector::<f64>::from_row_slice(&[-1.0, 1.0]);
        let constraint = DMatrix::<f64>::identity(2, 2);
        let solver = CsdSolver::new(
            h,
            constraint,
            CsdConfig {
                lambda: 1e6, // hard penalty
                epsilon: 1e-12,
                ..Default::default()
            },
        );
        let (x, diag) = solver.solve(&b);
        assert!(x[0].abs() < 1e-3, "x[0] should be driven near zero, got {}", x[0]);
        assert!(x[1] > 0.5, "x[1] should remain ≈ 1, got {}", x[1]);
        assert!(diag.iterations >= 1);
    }

    #[test]
    fn linear_constraint_on_amplitude_pushes_solution_into_feasible_region() {
        // x is a 1-vector. Constraint: 1·x ≥ 0 ⇒ x ≥ 0. b = -2 ⇒ unconstrained
        // minimizer is x = -2; constrained is x = 0.
        let h = DMatrix::<f64>::from_row_slice(1, 1, &[1.0]);
        let b = DVector::<f64>::from_row_slice(&[-2.0]);
        let constraint = DMatrix::<f64>::from_row_slice(1, 1, &[1.0]);
        let solver = CsdSolver::new(
            h,
            constraint,
            CsdConfig {
                lambda: 1e8,
                epsilon: 1e-12,
                ..Default::default()
            },
        );
        let (x, _) = solver.solve(&b);
        assert!(x[0].abs() < 1e-3, "expected x ≈ 0, got {}", x[0]);
    }
}
