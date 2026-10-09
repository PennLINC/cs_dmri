// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end coverage of the per-voxel α-strategy fit driver.
//!
//! The `solver::alpha` unit tests already cover the trait, path math, and BIC
//! recovery on a single problem. This file checks the orchestration layer:
//! parallel `fit_volume_with_alpha_strategy` over a synthetic 3-D volume,
//! including the per-voxel α map, the α-distribution summary, and the
//! interaction with FISTA's warm-started path.

use cs_dmri::basis::{Basis, RegularizationDiagonals};
use cs_dmri::io::dwi::DwiData;
use cs_dmri::qspace::{BvecFrame, GradientTable};
use cs_dmri::solver::fista::FistaSolver;
use cs_dmri::solver::tikhonov::TikhonovSolver;
use cs_dmri::solver::{AlphaPath, AlphaStrategy};
use cs_dmri::{
    FitConfig, ShoreBasis, fit_volume_with_alpha_strategy,
    fit_volume_with_alpha_strategy_l2_anchored_reporting,
};
use ndarray::{Array3, Array4};
use std::path::PathBuf;

/// Build a tiny synthetic DWI volume from a known coefficient pattern.
fn synth_dwi(nx: usize, ny: usize, nz: usize) -> (DwiData, ShoreBasis, nalgebra::DMatrix<f64>) {
    // Single-shell + b0, 32 directions on an icosahedron-ish spread.
    let bvecs: Vec<[f64; 3]> = (0..32)
        .map(|i| {
            let t = (i as f64) / 32.0 * std::f64::consts::TAU;
            let z = ((i as f64) - 15.5) / 16.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            [r * t.cos(), r * t.sin(), z]
        })
        .chain(std::iter::once([0.0, 0.0, 0.0]))
        .collect();
    let mut bvals: Vec<f64> = vec![1500.0; 32];
    bvals.push(0.0);

    let gtab = GradientTable::new(bvals, bvecs, Some(0.043), Some(0.011), None).unwrap();
    let basis = ShoreBasis::new(4, 700.0);
    let design = basis.design_matrix(&gtab);
    let n_coeffs = basis.n_coeffs();
    let n_grads = gtab.n_grads();

    // Two coefficient patterns — alternating across voxels — keeps each fit
    // genuinely sparse but with enough variation that a per-voxel α makes
    // sense.
    let mut coef_a = vec![0.0_f64; n_coeffs];
    let mut coef_b = vec![0.0_f64; n_coeffs];
    coef_a[0] = 1.0;
    coef_a[5] = 0.6;
    coef_b[0] = 0.8;
    coef_b[3] = -0.4;

    let mut data = Array4::<f32>::zeros((nx, ny, nz, n_grads));
    let mut mask = Array3::<bool>::from_elem((nx, ny, nz), true);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let coef = if (x + y + z) % 2 == 0 { &coef_a } else { &coef_b };
                let cv = nalgebra::DVector::<f64>::from_row_slice(coef);
                let s = &design * &cv;
                for k in 0..n_grads {
                    // Scale so values are roughly DWI-magnitude (a few thousand).
                    data[(x, y, z, k)] = (s[k] * 2000.0) as f32;
                }
                // Carve out one all-zero voxel to exercise the degenerate path.
                if x == 0 && y == 0 && z == 0 {
                    for k in 0..n_grads {
                        data[(x, y, z, k)] = 0.0;
                    }
                    mask[(x, y, z)] = true;
                }
            }
        }
    }

    let dwi = DwiData::from_table(
        data,
        mask,
        gtab,
        BvecFrame::ImageAxis,
        PathBuf::from("/tmp/synthetic.nii.gz"),
    );
    (dwi, basis, design)
}

#[test]
fn alpha_strategy_path_bic_runs_per_voxel_and_writes_alpha_map() {
    let (dwi, basis, design) = synth_dwi(3, 3, 2);
    let n_coeffs = basis.n_coeffs();

    let strategy = AlphaStrategy::PathBic {
        path: AlphaPath { n: 12, eps: 1e-3 },
    };
    // Initial alpha is a placeholder — every voxel overwrites it via the path.
    let base = FistaSolver::new(design.clone(), 1.0, 2000, 1e-7, false);

    let fit = fit_volume_with_alpha_strategy(
        &dwi,
        &design,
        &base,
        &strategy,
        n_coeffs,
        FitConfig { compute_diagnostics: true },
    );

    assert_eq!(fit.result.coefficients.shape(), &[3, 3, 2, n_coeffs]);
    let diag = fit
        .result
        .diagnostics
        .expect("diagnostics requested");
    let alpha_map = diag.alpha.expect("alpha map requested");

    // The all-zero voxel at (0, 0, 0) should fall through to alpha = 0 and
    // c = 0 without panicking.
    assert_eq!(alpha_map[(0, 0, 0)], 0.0);
    for k in 0..n_coeffs {
        assert_eq!(fit.result.coefficients[(0, 0, 0, k)], 0.0);
    }

    // Every other (in-mask) voxel should pick a strictly-positive α and have
    // at least the DC coefficient surviving the L1 shrinkage.
    let mut nonzero_voxels = 0_usize;
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..2 {
                if (x, y, z) == (0, 0, 0) {
                    continue;
                }
                assert!(
                    alpha_map[(x, y, z)] > 0.0,
                    "voxel ({x},{y},{z}) chose α = 0"
                );
                if fit.result.coefficients[(x, y, z, 0)].abs() > 0.0 {
                    nonzero_voxels += 1;
                }
            }
        }
    }
    assert!(
        nonzero_voxels >= 16,
        "expected most voxels to retain DC coefficient, got {nonzero_voxels}"
    );

    // R² should be high on the noiseless synthetic data.
    let r2 = diag.r2;
    let mean_r2: f64 = (0..3)
        .flat_map(|x| (0..3).flat_map(move |y| (0..2).map(move |z| (x, y, z))))
        .filter(|&(x, y, z)| (x, y, z) != (0, 0, 0))
        .map(|(x, y, z)| r2[(x, y, z)] as f64)
        .sum::<f64>()
        / 17.0;
    assert!(mean_r2 > 0.9, "expected high R² on noiseless synthetic, got {mean_r2}");

    // alpha_distribution should be reported (PathBic is a per-voxel strategy).
    let (median, p10, p90) = fit
        .alpha_distribution
        .expect("PathBic should report a per-voxel α distribution");
    assert!(p10 <= median && median <= p90);
    assert!(median > 0.0);
}

#[test]
fn alpha_strategy_fixed_skips_distribution() {
    let (dwi, basis, design) = synth_dwi(2, 2, 1);
    let strategy = AlphaStrategy::Fixed { alpha: 0.05 };
    let base = FistaSolver::new(design.clone(), 0.05, 1000, 1e-6, false);

    let fit = fit_volume_with_alpha_strategy(
        &dwi,
        &design,
        &base,
        &strategy,
        basis.n_coeffs(),
        FitConfig { compute_diagnostics: false },
    );
    assert!(fit.alpha_distribution.is_none(), "Fixed strategy reports a single global α");
    assert_eq!(fit.representative_alpha, 0.05);
}

#[test]
fn alpha_strategy_alpha_max_ratio_yields_per_voxel_alpha_map() {
    let (dwi, basis, design) = synth_dwi(2, 2, 1);
    let strategy = AlphaStrategy::AlphaMaxRatio { ratio: 0.01 };
    let base = FistaSolver::new(design.clone(), 1.0, 1000, 1e-6, false);

    let fit = fit_volume_with_alpha_strategy(
        &dwi,
        &design,
        &base,
        &strategy,
        basis.n_coeffs(),
        FitConfig { compute_diagnostics: true },
    );
    let diag = fit.result.diagnostics.unwrap();
    let alpha_map = diag.alpha.unwrap();
    // Some voxel must have a positive α (the all-zero voxel falls through).
    let pos = alpha_map.iter().filter(|v| **v > 0.0).count();
    assert!(pos > 0, "AlphaMaxRatio should leave at least one voxel with α > 0");
}

#[test]
fn alpha_strategy_path_l2_anchored_keeps_rss_within_slack_of_l2() {
    // Goal: every in-mask voxel's chosen-α RSS is ≤ (1+slack)·RSS_L2,
    // *except* possibly the fallback case where no α along the path
    // satisfies the constraint (and we fall back to argmin RSS — still
    // a sensible answer).
    let (dwi, basis, design) = synth_dwi(3, 3, 2);
    let n_coeffs = basis.n_coeffs();
    let n_grads = dwi.gtab.n_grads();

    let slack = 0.05;
    let strategy = AlphaStrategy::PathL2Anchored {
        path: AlphaPath { n: 12, eps: 1e-3 },
        slack,
    };

    let base = FistaSolver::new(design.clone(), 1.0, 2000, 1e-7, false);
    // Use the basis's natural regularization diagonals (matches the
    // production cs-fit --reg l2 invocation) so MᵀM + λN + λL is PD even
    // when MᵀM is rank-deficient.
    let regularization = basis.regularization();
    let _ = RegularizationDiagonals { primary: regularization.primary.clone(),
                                      secondary: regularization.secondary.clone() };
    let l2 = TikhonovSolver::new(design.clone(), &regularization, 1e-8, 1e-8);

    let fit = fit_volume_with_alpha_strategy_l2_anchored_reporting(
        &dwi,
        &design,
        &base,
        &l2,
        &strategy,
        n_coeffs,
        FitConfig { compute_diagnostics: true },
        || (),
    );

    let diag = fit.result.diagnostics.expect("diagnostics requested");
    let alpha_map = diag.alpha.expect("alpha map requested");
    let rss_l2_map = diag.rss_l2.expect("rss_l2 map requested");
    let resid_l1 = diag.residual_l2;     // square it to compare against rss_l2

    let mut respected = 0_usize;
    let mut total = 0_usize;
    let mut nonzero_alphas = 0_usize;
    let mut sparser_than_l2 = 0_usize;
    let mut zero_voxel_seen = false;

    for x in 0..3 {
        for y in 0..3 {
            for z in 0..2 {
                let alpha = alpha_map[(x, y, z)];
                if (x, y, z) == (0, 0, 0) {
                    // Degenerate all-zero voxel: α==0, no constraint.
                    assert_eq!(alpha, 0.0);
                    zero_voxel_seen = true;
                    continue;
                }
                total += 1;
                if alpha > 0.0 {
                    nonzero_alphas += 1;
                }
                let rss_l1 = (resid_l1[(x, y, z)] as f64).powi(2);
                let rss_l2 = rss_l2_map[(x, y, z)] as f64;
                // Voxel passes either the slack constraint OR the fallback
                // case (smallest RSS along path can still exceed the slack
                // when the path doesn't reach an L2-good fit).
                if rss_l1 <= (1.0 + slack) * rss_l2 + 1e-9 {
                    respected += 1;
                }
                // Sparsity bookkeeping: count strict zeros in the chosen
                // coefficient vector. L1 fits should have ≥ 1 zero
                // (otherwise we got the OLS solution).
                let mut zeros = 0_usize;
                for k in 0..n_coeffs {
                    if fit.result.coefficients[(x, y, z, k)].abs() == 0.0 {
                        zeros += 1;
                    }
                }
                if zeros >= 1 {
                    sparser_than_l2 += 1;
                }
            }
        }
    }

    assert!(zero_voxel_seen, "test fixture should include the (0,0,0) zero voxel");
    assert!(
        nonzero_alphas == total,
        "every non-degenerate voxel should pick α > 0; got {nonzero_alphas} / {total}"
    );
    // The slack constraint should bind in the strong majority. The
    // synthetic problem is small and noiseless, so on this fixture all
    // 17 in-mask voxels typically respect the slack; we allow a tiny
    // fallback margin to be robust to numerical drift.
    let respected_frac = respected as f64 / total as f64;
    assert!(
        respected_frac >= 0.8,
        "expected ≥80% of voxels' RSS_L1 ≤ (1+slack)·RSS_L2; got {:.0}% ({}/{})",
        respected_frac * 100.0,
        respected,
        total,
    );

    // L1 should yield at least one zero coefficient per voxel for some
    // voxels — otherwise the L1 prior bought us nothing over OLS.
    assert!(
        sparser_than_l2 >= total / 2,
        "expected at least half the voxels to have ≥1 zero coef under L2-anchored; \
         got {sparser_than_l2} / {total}"
    );

    // alpha_distribution should be a (median, p10, p90) tuple (PathL2Anchored
    // is per-voxel, like PathBic).
    let (_median, p10, p90) = fit
        .alpha_distribution
        .expect("PathL2Anchored should report a per-voxel α distribution");
    assert!(p10 <= p90);

    // Sanity: rss_l2 is non-negative everywhere.
    for v in rss_l2_map.iter() {
        assert!(*v >= 0.0, "negative rss_l2 entry: {v}");
    }

    // Number of gradient points exposed for diagnostics.
    let _ = n_grads;
}

#[test]
fn alpha_strategy_path_l2_anchored_serializes_round_trip() {
    // Make sure the new variant survives serde round-trip — the sidecar
    // JSON depends on this.
    let strat = AlphaStrategy::PathL2Anchored {
        path: AlphaPath { n: 16, eps: 1e-3 },
        slack: 0.05,
    };
    let s = serde_json::to_string(&strat).expect("serialize");
    let back: AlphaStrategy = serde_json::from_str(&s).expect("deserialize");
    match back {
        AlphaStrategy::PathL2Anchored { path, slack } => {
            assert_eq!(path.n, 16);
            assert!((path.eps - 1e-3).abs() < 1e-12);
            assert!((slack - 0.05).abs() < 1e-12);
        }
        other => panic!("round-trip yielded wrong variant: {other:?}"),
    }
}
