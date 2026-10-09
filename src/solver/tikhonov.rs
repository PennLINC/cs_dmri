// SPDX-License-Identifier: MIT OR Apache-2.0
//! Closed-form L2-regularized least squares with cached Cholesky factor.
//!
//! Solves: argmin_c ‖Mc − S‖² + λ_N ⟨c, N c⟩ + λ_L ⟨c, L c⟩
//! where N and L are diagonal, given by the basis. The normal-equations
//! matrix `(MᵀM + λ_N N + λ_L L)` does not depend on the signal, so it is
//! factored once at construction and reused for every voxel.

use nalgebra::{Cholesky, DMatrix, DVector};

use super::{FitDiagnostics, Problem, Solver};
use crate::basis::RegularizationDiagonals;

/// L2-regularized least-squares solver with a precomputed factorization.
pub struct TikhonovSolver {
    /// Lower-triangular Cholesky factor of (MᵀM + λ_N N + λ_L L).
    chol: Cholesky<f64, nalgebra::Dyn>,
    /// Mᵀ, kept around so we can compute the right-hand side cheaply.
    mt: DMatrix<f64>,
    design: DMatrix<f64>,
    pub lambda_primary: f64,
    pub lambda_secondary: f64,
}

impl TikhonovSolver {
    pub fn new(
        design: DMatrix<f64>,
        regularization: &RegularizationDiagonals,
        lambda_primary: f64,
        lambda_secondary: f64,
    ) -> Self {
        let mt = design.transpose();
        let mut a = &mt * &design;
        // primary diagonal
        for (i, &d) in regularization.primary.iter().enumerate() {
            a[(i, i)] += lambda_primary * d;
        }
        // optional secondary diagonal
        if let Some(secondary) = &regularization.secondary {
            for (i, &d) in secondary.iter().enumerate() {
                a[(i, i)] += lambda_secondary * d;
            }
        }
        let chol = Cholesky::new(a).expect("normal-equations matrix should be PD");
        Self {
            chol,
            mt,
            design,
            lambda_primary,
            lambda_secondary,
        }
    }

    pub fn design(&self) -> &DMatrix<f64> {
        &self.design
    }
}

impl Solver for TikhonovSolver {
    fn fit(&self, problem: &Problem<'_>) -> (DVector<f64>, FitDiagnostics) {
        let rhs = &self.mt * problem.signal;
        let coef = self.chol.solve(&rhs);
        let residual = problem.signal - &self.design * &coef;
        let resid_l2 = residual.norm();
        (
            coef,
            FitDiagnostics {
                iterations: 1,
                residual_l2: resid_l2,
                converged: true,
                alpha: 0.0,
                regularization_kind: 2,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::DMatrix;

    #[test]
    fn unregularized_recovery_of_known_solution() {
        // No regularization → ordinary least squares.
        let m = DMatrix::<f64>::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0]);
        let true_c = DVector::<f64>::from_row_slice(&[2.0, -3.0]);
        let s = &m * &true_c;
        let reg = RegularizationDiagonals {
            primary: DVector::from_element(2, 0.0),
            secondary: None,
        };
        let solver = TikhonovSolver::new(m.clone(), &reg, 1e-12, 0.0);
        let problem = Problem {
            design: &m,
            signal: &s,
        };
        let (coef, diag) = solver.fit(&problem);
        for i in 0..2 {
            assert_abs_diff_eq!(coef[i], true_c[i], epsilon = 1e-6);
        }
        assert!(diag.converged);
        assert_eq!(diag.regularization_kind, 2);
    }
}
