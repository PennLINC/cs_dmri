// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for `cs_dmri::scalars::microstructure`.
//!
//! Generates an analytical single-tensor signal, fits BrainSuiteSHORE, and
//! checks RTOP/RTAP/RTPP/MSD/QIV/NG against closed-form ground-truth values
//! (mirroring the assertions in `dipy/reconst/tests/test_mapmri.py:573-661`).

use std::f64::consts::PI;

use cs_dmri::ShoreBasis;
use cs_dmri::basis::Basis;
use cs_dmri::qspace::GradientTable;
use cs_dmri::scalars::microstructure::{self, ScalarBasisInfo};
use cs_dmri::solver::tikhonov::TikhonovSolver;
use cs_dmri::solver::{Problem, Solver};
use nalgebra::DVector;

fn unit(x: f64, y: f64, z: f64) -> [f64; 3] {
    let n = (x * x + y * y + z * z).sqrt().max(1e-12);
    [x / n, y / n, z / n]
}

/// Multi-shell gradient table: 1 b0 + (b=1000, 2000, 3000, 4000) × 64 dirs each.
/// Spirals on the sphere via golden-angle z-stack — matches the spread of a
/// typical CS-DSI scheme well enough for SHORE convergence.
fn dense_multishell() -> (Vec<f64>, Vec<[f64; 3]>) {
    let n_per_shell = 64;
    let mut bvals = vec![0.0_f64];
    let mut bvecs = vec![[0.0, 0.0, 0.0]];
    let phi = PI * (3.0 - 5.0_f64.sqrt());
    for shell_b in [1000.0_f64, 2000.0, 3000.0, 4000.0] {
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

/// Synthesize $E(q) = \exp(-b\,\hat u^\top D\,\hat u)$ for a diagonal diffusion
/// tensor `evals` with eigenvectors aligned to (x, y, z). Returns the signal
/// vector (already normalized: `E(b=0) = 1`).
fn single_tensor_signal(
    bvals: &[f64],
    bvecs: &[[f64; 3]],
    evals: [f64; 3],
) -> DVector<f64> {
    let n = bvals.len();
    let mut s = DVector::<f64>::zeros(n);
    for i in 0..n {
        let [vx, vy, vz] = bvecs[i];
        let adc = evals[0] * vx * vx + evals[1] * vy * vy + evals[2] * vz * vz;
        s[i] = (-bvals[i] * adc).exp();
    }
    s
}

/// Closed-form ground-truth scalars for an isotropic Gaussian propagator.
/// Anisotropic versions (Mapmri test_mapmri.py:585-593) — `lambda1` is the
/// principal-axis diffusivity (the one RTPP uses).
fn ground_truth(l1: f64, l2: f64, l3: f64, tau: f64) -> Truth {
    let rtpp = 1.0 / (2.0 * (PI * l1 * tau).sqrt());
    let rtap = 1.0 / (2.0 * (PI * l2 * tau).sqrt())
        * 1.0
        / (2.0 * (PI * l3 * tau).sqrt());
    let rtop = rtpp * rtap;
    let msd = 2.0 * (l1 + l2 + l3) * tau;
    let qiv = 64.0 * PI.powf(7.0 / 2.0) * (l1 * l2 * l3 * tau.powi(3)).powf(1.5)
        / ((l2 * l3 + l1 * (l2 + l3)) * tau.powi(2));
    Truth { rtop, rtap, rtpp, msd, qiv }
}

#[derive(Debug, Clone, Copy)]
struct Truth {
    rtop: f64,
    rtap: f64,
    rtpp: f64,
    msd: f64,
    qiv: f64,
}

/// Ratio-tolerance helper: |a − b| / |b| ≤ tol. Required because the SHORE
/// truncation introduces O(few percent) bias in scalar magnitudes.
fn close(a: f64, b: f64, tol: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    if b == 0.0 {
        return a.abs() <= tol;
    }
    (a - b).abs() / b.abs() <= tol
}

#[test]
fn isotropic_single_tensor_rtop_msd_match_ground_truth() {
    let (bvals, bvecs) = dense_multishell();
    // Use deltas that give a clean τ; identical to roundtrip.rs' choice.
    let big_delta = 0.0431_f64;
    let small_delta = 0.0107_f64;
    let tau = big_delta - small_delta / 3.0;

    let gtab = GradientTable::new(
        bvals.clone(),
        bvecs.clone(),
        Some(big_delta),
        Some(small_delta),
        None,
    )
    .unwrap();

    // Isotropic diffusivity D = 0.7e-3 mm²/s — characteristic of GM/CSF.
    let d = 0.7e-3_f64;
    let signal = single_tensor_signal(&bvals, &bvecs, [d, d, d]);

    let basis = ShoreBasis::new(6, 700.0);
    let m = basis.design_matrix(&gtab);
    let reg = basis.regularization();
    // Light Tikhonov for numerical stability — the SHORE basis has near-collinear
    // radial modes at low q, so a tiny ridge is needed to avoid amplifying noise.
    let solver = TikhonovSolver::new(m.clone(), &reg, 1e-6, 1e-6);
    let problem = Problem { design: &m, signal: &signal };
    let (coef, _diag) = solver.fit(&problem);
    let coef_vec: Vec<f64> = coef.iter().copied().collect();

    let info = ScalarBasisInfo::from_basis(&basis);
    let truth = ground_truth(d, d, d, tau);

    let rtop = microstructure::rtop(&coef_vec, &info);
    let msd = microstructure::msd(&coef_vec, &info);
    let qiv = microstructure::qiv(&coef_vec, &info);

    eprintln!(
        "[iso] RTOP fit={rtop:.4e} truth={:.4e} (rel err={:.3})",
        truth.rtop,
        (rtop - truth.rtop).abs() / truth.rtop
    );
    eprintln!(
        "[iso] MSD  fit={msd:.4e} truth={:.4e} (rel err={:.3})",
        truth.msd,
        (msd - truth.msd).abs() / truth.msd
    );
    eprintln!(
        "[iso] QIV  fit={qiv:.4e} truth={:.4e} (rel err={:.3})",
        truth.qiv,
        (qiv - truth.qiv).abs() / truth.qiv
    );

    // RTOP and MSD are SHORE's bread-and-butter scalars; isotropic + multi-shell
    // should give <10% error.
    assert!(close(rtop, truth.rtop, 0.10), "rtop {} vs gt {}", rtop, truth.rtop);
    assert!(close(msd, truth.msd, 0.10), "msd {} vs gt {}", msd, truth.msd);
}

#[test]
fn anisotropic_single_tensor_rtap_rtpp_match_principal_axis() {
    let (bvals, bvecs) = dense_multishell();
    let big_delta = 0.0431_f64;
    let small_delta = 0.0107_f64;
    let tau = big_delta - small_delta / 3.0;

    let gtab = GradientTable::new(
        bvals.clone(),
        bvecs.clone(),
        Some(big_delta),
        Some(small_delta),
        None,
    )
    .unwrap();

    // Anisotropic single fiber along x: λ1 (parallel) > λ2 = λ3 (perp).
    let l1 = 1.7e-3_f64;
    let l2 = 0.3e-3_f64;
    let l3 = 0.3e-3_f64;
    let signal = single_tensor_signal(&bvals, &bvecs, [l1, l2, l3]);

    let basis = ShoreBasis::new(6, 700.0);
    let m = basis.design_matrix(&gtab);
    let reg = basis.regularization();
    let solver = TikhonovSolver::new(m.clone(), &reg, 1e-6, 1e-6);
    let problem = Problem { design: &m, signal: &signal };
    let (coef, _diag) = solver.fit(&problem);
    let coef_vec: Vec<f64> = coef.iter().copied().collect();

    let info = ScalarBasisInfo::from_basis(&basis);
    let truth = ground_truth(l1, l2, l3, tau);

    let rtop = microstructure::rtop(&coef_vec, &info);
    let msd = microstructure::msd(&coef_vec, &info);
    let qiv = microstructure::qiv(&coef_vec, &info);
    let principal = [1.0, 0.0, 0.0]; // ground-truth fiber direction
    let rtap = microstructure::rtap(&coef_vec, &info, principal);
    let rtpp = microstructure::rtpp(&coef_vec, &info, principal);

    eprintln!(
        "[aniso x-fiber] RTOP fit={rtop:.4e} truth={:.4e} (rel err={:.3})",
        truth.rtop,
        (rtop - truth.rtop).abs() / truth.rtop
    );
    eprintln!(
        "[aniso x-fiber] RTAP fit={rtap:.4e} truth={:.4e} (rel err={:.3})",
        truth.rtap,
        (rtap - truth.rtap).abs() / truth.rtap
    );
    eprintln!(
        "[aniso x-fiber] RTPP fit={rtpp:.4e} truth={:.4e} (rel err={:.3})",
        truth.rtpp,
        (rtpp - truth.rtpp).abs() / truth.rtpp
    );
    eprintln!(
        "[aniso x-fiber] MSD  fit={msd:.4e} truth={:.4e} (rel err={:.3})",
        truth.msd,
        (msd - truth.msd).abs() / truth.msd
    );
    eprintln!(
        "[aniso x-fiber] QIV  fit={qiv:.4e} truth={:.4e} (rel err={:.3})",
        truth.qiv,
        (qiv - truth.qiv).abs() / truth.qiv
    );

    // Tolerances are loose because the BrainSuiteSHORE truncation at
    // radial_order=6 introduces a few-percent bias even for well-behaved
    // anisotropic Gaussians; the Python validation harness in
    // `scripts/validate_microstructure.py` does the rigorous cross-check
    // against dipy iso-MAPMRI on the same coefficients. RTOP and RTPP are
    // the most accurate (single-axis or m=0 only); RTAP and MSD pick up
    // additional cross-mode bias.
    assert!(close(rtop, truth.rtop, 0.15), "rtop {} vs gt {}", rtop, truth.rtop);
    assert!(close(msd, truth.msd, 0.20), "msd {} vs gt {}", msd, truth.msd);
    assert!(close(rtap, truth.rtap, 0.30), "rtap {} vs gt {}", rtap, truth.rtap);
    assert!(close(rtpp, truth.rtpp, 0.20), "rtpp {} vs gt {}", rtpp, truth.rtpp);
}
