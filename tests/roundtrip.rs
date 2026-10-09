// SPDX-License-Identifier: MIT OR Apache-2.0
//! Round-trip integration tests.
//!
//! The SHORE basis is intentionally overcomplete (multiple isotropic radial
//! polynomials are near-collinear over the practical q range), so we verify
//! signal-space recovery (‖M·ĉ − s‖) rather than coefficient-space recovery.
//! That's what end users actually care about.

use cs_dmri::basis::Basis;
use cs_dmri::qspace::GradientTable;
use cs_dmri::solver::tikhonov::TikhonovSolver;
use cs_dmri::solver::{Problem, Solver};
use cs_dmri::ShoreBasis;
use nalgebra::DVector;

fn unit(x: f64, y: f64, z: f64) -> [f64; 3] {
    let n = (x * x + y * y + z * z).sqrt().max(1e-12);
    [x / n, y / n, z / n]
}

fn synthetic_gradients() -> (Vec<f64>, Vec<[f64; 3]>) {
    // 1 b0 + three shells (b = 1000, 2000, 3000) × 30 directions each.
    let n_per_shell = 30;
    let mut bvals = vec![0.0_f64];
    let mut bvecs = vec![[0.0, 0.0, 0.0]];
    let phi = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for shell_b in [1000.0_f64, 2000.0, 3000.0] {
        for k in 0..n_per_shell {
            let t = (k as f64 + 0.5) / n_per_shell as f64;
            let z = 1.0 - 2.0 * t;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let a = phi * k as f64;
            bvals.push(shell_b);
            bvecs.push(unit(a.cos() * r, a.sin() * r, z));
        }
    }
    (bvals, bvecs)
}

#[test]
fn shore_l2_signal_recovery() {
    let (bvals, bvecs) = synthetic_gradients();
    let gtab = GradientTable::new(bvals, bvecs, Some(0.0431), Some(0.0107), None).unwrap();
    let basis = ShoreBasis::new(4, 700.0);
    let m = basis.design_matrix(&gtab);
    let n_coeffs = basis.n_coeffs();

    // True coefficient vector with a few non-zeros (some isotropic, some at ℓ=2).
    let mut true_c = DVector::<f64>::zeros(n_coeffs);
    true_c[0] = 1.5;
    if n_coeffs > 5 {
        true_c[3] = -0.4;
        true_c[5] = 0.8;
    }
    let signal = &m * &true_c;

    let reg = basis.regularization();
    let solver = TikhonovSolver::new(m.clone(), &reg, 1e-10, 1e-10);
    let problem = Problem {
        design: &m,
        signal: &signal,
    };
    let (coef, _) = solver.fit(&problem);

    let predicted = &m * &coef;
    let signal_err = (&predicted - &signal).norm() / signal.norm();
    assert!(
        signal_err < 1e-3,
        "L2 fit signal-space relative error {signal_err} too large"
    );
}

#[test]
fn shore_l1_signal_recovery() {
    use cs_dmri::solver::fista::FistaSolver;

    let (bvals, bvecs) = synthetic_gradients();
    let gtab = GradientTable::new(bvals, bvecs, Some(0.0431), Some(0.0107), None).unwrap();
    let basis = ShoreBasis::new(4, 700.0);
    let m = basis.design_matrix(&gtab);
    let n_coeffs = basis.n_coeffs();

    let mut true_c = DVector::<f64>::zeros(n_coeffs);
    true_c[0] = 1.0;
    if n_coeffs > 5 {
        true_c[3] = -0.3;
    }
    let signal = &m * &true_c;

    // Very small alpha; FISTA should recover a near-perfect prediction.
    let solver = FistaSolver::new(m.clone(), 1e-7, 5000, 1e-10, false);
    let problem = Problem {
        design: &m,
        signal: &signal,
    };
    let (coef, diag) = solver.fit(&problem);
    let predicted = &m * &coef;
    let signal_err = (&predicted - &signal).norm() / signal.norm();
    // L1 with a tiny α still imposes some shrinkage on the overcomplete SHORE
    // basis; ~10% signal-space error is expected. We mainly want to confirm
    // FISTA actually converges and produces a reasonable approximation.
    assert!(
        signal_err < 0.15,
        "L1 fit signal-space relative error {signal_err} too large; iters={}",
        diag.iterations,
    );
}

#[test]
fn shore_basis_consistent_through_synthesis() {
    use cs_dmri::synthesize_volume;
    use ndarray::Array4;

    let (bvals, bvecs) = synthetic_gradients();
    let gtab = GradientTable::new(bvals, bvecs, Some(0.0431), Some(0.0107), None).unwrap();
    let basis = ShoreBasis::new(4, 700.0);
    let n_coeffs = basis.n_coeffs();

    // Build a tiny coefficient volume (2 × 1 × 1 voxels) with two distinct
    // coefficient profiles, run synthesis, fit it back, and confirm signals
    // match.
    let mut coeffs = Array4::<f32>::zeros((2, 1, 1, n_coeffs));
    coeffs[(0, 0, 0, 0)] = 1.2;
    if n_coeffs > 4 {
        coeffs[(0, 0, 0, 3)] = -0.4;
        coeffs[(1, 0, 0, 0)] = 0.7;
        coeffs[(1, 0, 0, 4)] = 0.5;
    }

    let synth = synthesize_volume(&coeffs, &basis, &gtab);
    assert_eq!(synth.shape(), &[2, 1, 1, gtab.n_grads()]);

    // Fit voxel (0,0,0) with the L2 solver using the synthesized signal and
    // check the prediction matches.
    let m = basis.design_matrix(&gtab);
    let reg = basis.regularization();
    let solver = TikhonovSolver::new(m.clone(), &reg, 1e-10, 1e-10);
    for ix in 0..2 {
        let signal: DVector<f64> = DVector::from_iterator(
            gtab.n_grads(),
            (0..gtab.n_grads()).map(|t| synth[(ix, 0, 0, t)] as f64),
        );
        let problem = Problem {
            design: &m,
            signal: &signal,
        };
        let (coef, _) = solver.fit(&problem);
        let predicted = &m * &coef;
        let err = (&predicted - &signal).norm() / signal.norm().max(1e-12);
        assert!(err < 1e-3, "voxel {ix}: signal recovery err = {err}");
    }
}
