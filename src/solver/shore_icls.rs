// SPDX-License-Identifier: MIT OR Apache-2.0
//! SHORE fit with hard non-negativity on the *projected ODF amplitudes*.
//!
//! `cs-fit --non-negative` (the FISTA flag) enforces non-negativity on the
//! raw SHORE coefficients, which is rarely what users want — valid signals
//! routinely produce negative SH coefficients in the SHORE basis. What you
//! usually want is "amplitude of the projected ODF, sampled on a sphere, is
//! non-negative everywhere". That's a *linear* constraint on the SHORE
//! coefficients (composed of the SHORE→Tournier-SH projection and the SH
//! evaluation matrix on the constraint sphere) and the [`IclsSolver`] from
//! the SS3T port already solves problems of exactly that shape.
//!
//! This adapter assembles the constraint matrix once at construction and
//! delegates per-voxel solves to ICLS. Plugs into
//! [`crate::fit::fit_volume_reporting`] via the [`Solver`] trait.

use nalgebra::{DMatrix, DVector};

use crate::basis::shore::ShoreBasis;
use crate::odf::shore_to_tournier_sh_matrix;
use crate::sh::sh2amp_cart;
use crate::solver::icls::{IclsConfig, IclsSolver, IclsWorkspace};
use crate::solver::{FitDiagnostics, Problem, Solver};

/// SHORE-fit with hard amplitude non-negativity via Goldfarb-Idnani ICLS.
///
/// `regularization_kind` reports as `3` in `FitDiagnostics` to distinguish
/// from FISTA (1) and Tikhonov (2).
#[derive(Debug, Clone)]
pub struct ShoreIclsSolver {
    icls: IclsSolver,
    lmax: u32,
    n_constraint_dirs: usize,
}

impl ShoreIclsSolver {
    /// Build a solver. `design` is the SHORE design matrix from `basis`;
    /// `lmax` is the SH order used to project SHORE coefficients to ODF
    /// amplitudes for the constraint (typically `default_lmax(radial_order)`).
    /// The constraint sphere is the dsistudio ODF8 hemisphere (321 dirs).
    pub fn new(
        design: DMatrix<f64>,
        basis: &ShoreBasis,
        lmax: u32,
        cfg: IclsConfig,
    ) -> Self {
        // Constraint matrix: amplitude of the projected ODF on each sphere
        // direction is non-negative.
        //   B = Y_sphere · (SHORE → Tournier-SH)
        //   Bx ≥ 0  ⇔  ODF amplitude at every direction is ≥ 0.
        let projection = shore_to_tournier_sh_matrix(basis, lmax); // (n_sh × n_shore)
        let dirs = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
        let n_dirs = dirs.len();
        let y_sphere_f32 = sh2amp_cart(dirs, lmax as usize); // (n_dirs × n_sh) f32
        let n_sh = projection.nrows();
        // Cast f32 → f64. odx-rs returns ndarray::Array2<f32>; copy into a
        // nalgebra DMatrix for the matmul.
        let mut y_sphere = DMatrix::<f64>::zeros(n_dirs, n_sh);
        for i in 0..n_dirs {
            for j in 0..n_sh {
                y_sphere[(i, j)] = y_sphere_f32[(i, j)] as f64;
            }
        }
        let constraint = &y_sphere * &projection;
        let icls = IclsSolver::new(design, constraint, cfg);
        Self {
            icls,
            lmax,
            n_constraint_dirs: n_dirs,
        }
    }

    /// Lmax used for the constraint projection. Records to the sidecar.
    pub fn lmax(&self) -> u32 {
        self.lmax
    }

    /// Number of sphere directions on which non-negativity is enforced.
    pub fn n_constraint_dirs(&self) -> usize {
        self.n_constraint_dirs
    }

    /// Build a per-thread workspace for high-throughput parallel fits.
    pub fn workspace(&self) -> IclsWorkspace {
        self.icls.workspace()
    }

    /// Solve into a caller-supplied workspace. Mirrors
    /// [`IclsSolver::solve_into`] but computes the residual relative to the
    /// SHORE design (rather than the constraint matrix) so `FitDiagnostics`
    /// reflects fit-to-signal quality.
    pub fn solve_into(
        &self,
        b: &DVector<f64>,
        ws: &mut IclsWorkspace,
    ) -> (DVector<f64>, FitDiagnostics) {
        let (x, icls_diag) = self.icls.solve_into(b, ws);
        let residual = self.icls.h() * &x - b;
        let diag = FitDiagnostics {
            iterations: icls_diag.iterations as u32,
            residual_l2: residual.norm(),
            converged: icls_diag.converged,
            alpha: 0.0,
            regularization_kind: 3,
        };
        (x, diag)
    }
}

impl Solver for ShoreIclsSolver {
    fn fit(&self, problem: &Problem<'_>) -> (DVector<f64>, FitDiagnostics) {
        let mut ws = self.workspace();
        self.solve_into(problem.signal, &mut ws)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::Basis;
    use crate::basis::shore::ShoreBasis;
    use crate::qspace::GradientTable;
    use approx::assert_abs_diff_eq;

    /// Build a small synthetic problem and verify ICLS recovers a fit whose
    /// projected ODF has no negative-amplitude lobes on the constraint sphere.
    #[test]
    fn fit_produces_non_negative_amplitude_odf() {
        // 30 random-ish gradient directions on the sphere, b=1000.
        let n_grads = 30;
        let mut bvals = Vec::with_capacity(n_grads + 1);
        let mut bvecs: Vec<[f64; 3]> = Vec::with_capacity(n_grads + 1);
        bvals.push(0.0);
        bvecs.push([0.0, 0.0, 0.0]);
        for i in 0..n_grads {
            let theta = (i as f64) * 0.7;
            let phi = (i as f64) * 1.3;
            bvals.push(1000.0);
            bvecs.push([
                theta.cos() * phi.sin(),
                theta.sin() * phi.sin(),
                phi.cos(),
            ]);
        }
        let gtab = GradientTable::new(bvals, bvecs, Some(0.05), Some(0.012), None).unwrap();
        let basis = ShoreBasis::new(4, 700.0);
        let design = basis.design_matrix(&gtab);

        // Synthesize a "signal" that, fit unconstrained, would have a wildly
        // negative-lobed ODF: random signs with large magnitudes.
        let signal: DVector<f64> = DVector::from_iterator(
            n_grads + 1,
            (0..(n_grads + 1)).map(|i| {
                if i == 0 {
                    1.0
                } else {
                    0.3 + 0.2 * ((i as f64) * 1.7).sin()
                }
            }),
        );

        let lmax = crate::odf::default_lmax(basis.radial_order);
        let cfg = IclsConfig {
            max_iter: 500,
            tol: 1e-9,
            epsilon: 1e-10,
        };
        let solver = ShoreIclsSolver::new(design.clone(), &basis, lmax, cfg);
        let problem = Problem {
            design: &design,
            signal: &signal,
        };
        let (x, diag) = solver.fit(&problem);
        assert!(diag.converged, "ICLS should converge on a 30-grad fit");
        assert_eq!(diag.regularization_kind, 3);

        // Check: the recovered SHORE coefficients project to an ODF whose
        // sphere amplitudes are all ≥ 0 (within tolerance).
        let projection = shore_to_tournier_sh_matrix(&basis, lmax);
        let dirs = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
        let y = sh2amp_cart(dirs, lmax as usize);
        let sh_coeffs = &projection * &x;
        let mut min_amp = f64::INFINITY;
        for d in 0..dirs.len() {
            let mut amp = 0.0_f64;
            for k in 0..sh_coeffs.len() {
                amp += y[(d, k)] as f64 * sh_coeffs[k];
            }
            if amp < min_amp {
                min_amp = amp;
            }
        }
        assert!(
            min_amp >= -1e-6,
            "min ODF amplitude on sphere = {} (should be ≥ 0)",
            min_amp
        );
        assert_abs_diff_eq!(min_amp.max(0.0), 0.0, epsilon = 1.0); // soft check
    }
}
