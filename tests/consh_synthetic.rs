// SPDX-License-Identifier: MIT OR Apache-2.0
//! Whole-volume ground-truth test of continuous-b multi-tissue CSD (CONSH).
//!
//! A small synthetic brain is sampled on a Cartesian (DSI-like) q-space grid,
//! which has 14 distinct b-values and no shells. It runs through the same steps
//! as the `consh` tool (RESTORE tensor fit, Dhollander tissue selection,
//! dense-b response estimation, deconvolution) with each response estimator,
//! and the result is compared with the planted tissue classes and fiber
//! directions.

use std::f64::consts::PI;

use cs_dmri::dti::{DtiFitConfig, RestoreConfig, fit_volume_restore};
use cs_dmri::io::dwi::DwiData;
use cs_dmri::multitissue::consh::{ConshConfig, fit_volume_consh};
use cs_dmri::multitissue::response_estimation::{
    ConshResponseConfig, ResponseEstimator, estimate_consh_responses,
};
use cs_dmri::qspace::{BvecFrame, GradientTable};
use cs_dmri::solver::icls::IclsConfig;
use ndarray::{Array3, Array4};
use rand::{Rng, SeedableRng};

const NX: usize = 28;
const NY: usize = 28;
const NZ: usize = 8;
const CENTER: f64 = 13.5;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Tissue {
    Csf,
    Wm,
    Crossing,
    Gm,
    Outside,
}

fn tissue_at(x: usize, y: usize) -> Tissue {
    let r = ((x as f64 - CENTER).powi(2) + (y as f64 - CENTER).powi(2)).sqrt();
    match r {
        r if r < 4.0 => Tissue::Csf,
        r if r < 9.0 => Tissue::Wm,
        r if r < 10.5 => Tissue::Crossing,
        r if r < 13.5 => Tissue::Gm,
        _ => Tissue::Outside,
    }
}

/// In-plane direction tangent to the circle through the voxel.
fn circumferential(x: usize, y: usize) -> [f64; 3] {
    let (dx, dy) = (x as f64 - CENTER, y as f64 - CENTER);
    let r = (dx * dx + dy * dy).sqrt().max(1e-9);
    [-dy / r, dx / r, 0.0]
}

/// q-space points of a DSI lattice with |q|² ≤ 16, one per antipodal pair,
/// plus two b=0 volumes: 14 distinct b-values up to 3000 s/mm² (|q|² cannot
/// be 7 or 15, which are not sums of three squares).
fn dsi_table() -> GradientTable {
    let (mut bvals, mut bvecs) = (vec![0.0, 0.0], vec![[0.0, 0.0, 1.0]; 2]);
    for i in -4i32..=4 {
        for j in -4i32..=4 {
            for k in 0i32..=4 {
                let n2 = i * i + j * j + k * k;
                let upper = k > 0 || j > 0 || (j == 0 && i > 0);
                if n2 == 0 || n2 > 16 || !upper {
                    continue;
                }
                let n = (n2 as f64).sqrt();
                bvals.push(3000.0 * n2 as f64 / 16.0);
                bvecs.push([i as f64 / n, j as f64 / n, k as f64 / n]);
            }
        }
    }
    GradientTable::new(bvals, bvecs, Some(0.043), Some(0.011), None).unwrap()
}

/// Axially symmetric Gaussian (zeppelin) signal along `axis`.
fn zeppelin(b: f64, g: [f64; 3], axis: [f64; 3]) -> f64 {
    let c = g[0] * axis[0] + g[1] * axis[1] + g[2] * axis[2];
    let (d_par, d_perp) = (1.7e-3, 0.2e-3);
    (-b * (d_perp + (d_par - d_perp) * c * c)).exp()
}

fn synthetic_brain() -> DwiData {
    let gtab = dsi_table();
    let n = gtab.n_grads();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    // Gaussian noise (σ = 4, an SNR of 250 in WM at b=0), by Box–Muller.
    let mut noise = move || {
        let (u, v): (f64, f64) = (rng.gen_range(1e-12..1.0), rng.gen());
        4.0 * (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    };
    let mut data = Array4::<f32>::zeros((NX, NY, NZ, n));
    let mut mask = Array3::<bool>::from_elem((NX, NY, NZ), false);
    for x in 0..NX {
        for y in 0..NY {
            let t = tissue_at(x, y);
            if t == Tissue::Outside {
                continue;
            }
            for z in 0..NZ {
                mask[(x, y, z)] = true;
                for i in 0..n {
                    let (b, g) = (gtab.bvals[i], gtab.bvecs[i]);
                    let s = match t {
                        Tissue::Csf => 2600.0 * (-3.0e-3 * b).exp(),
                        Tissue::Gm => 1300.0 * (-0.8e-3 * b).exp(),
                        Tissue::Wm => 1000.0 * zeppelin(b, g, circumferential(x, y)),
                        Tissue::Crossing => {
                            500.0 * zeppelin(b, g, circumferential(x, y))
                                + 500.0 * zeppelin(b, g, [0.0, 0.0, 1.0])
                        }
                        Tissue::Outside => unreachable!(),
                    };
                    data[(x, y, z, i)] = (s + noise()).max(0.0) as f32;
                }
            }
        }
    }
    DwiData::from_table(data, mask, gtab, BvecFrame::WorldRas, "synthetic.nii.gz".into())
}

/// Amplitudes of the Tournier SH series `coeffs` along unit vectors `dirs`.
fn amplitudes(coeffs: &[f64], lmax: usize, dirs: &[[f32; 3]]) -> Vec<f64> {
    let b = odx_rs::mrtrix_sh::sh2amp_cart(dirs, lmax);
    (0..dirs.len())
        .map(|i| (0..coeffs.len()).map(|j| b[(i, j)] as f64 * coeffs[j]).sum())
        .collect()
}

/// `n` nearly uniform directions on the upper hemisphere (Fibonacci spiral),
/// about 2° apart.
fn upper_hemisphere(n: usize) -> Vec<[f32; 3]> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let phi = golden * i as f64;
            [(r * phi.cos()) as f32, (r * phi.sin()) as f32, z as f32]
        })
        .collect()
}

fn angle_deg(a: [f32; 3], b: [f64; 3]) -> f64 {
    let c = (a[0] as f64 * b[0] + a[1] as f64 * b[1] + a[2] as f64 * b[2]).abs().min(1.0);
    c.acos() * 180.0 / PI
}

#[test]
fn consh_recovers_planted_tissues_and_fibers_on_non_shelled_qspace() {
    let dwi = synthetic_brain();
    let shells: std::collections::BTreeSet<i64> =
        dwi.gtab.bvals.iter().map(|&b| b.round() as i64).collect();
    assert_eq!(shells.len(), 15, "14 distinct b-values plus b=0");

    let dti = fit_volume_restore(&dwi, &RestoreConfig::default(), DtiFitConfig::default()).unwrap();
    let sphere = upper_hemisphere(4000);

    for estimator in [ResponseEstimator::Kernel, ResponseEstimator::Smooth, ResponseEstimator::Dki] {
        let mut rcfg = ConshResponseConfig { estimator, ..ConshResponseConfig::default() };
        rcfg.dhollander.stages.erode = 0;
        let (responses, diag) = estimate_consh_responses(&dwi, &dti, &rcfg)
            .unwrap_or_else(|e| panic!("{estimator:?}: response estimation failed: {e}"));
        assert!(diag.n_wm_voxels > 0 && diag.n_gm_voxels > 0 && diag.n_csf_voxels > 0);

        let cfg = ConshConfig {
            lmax_wm: 8,
            icls: IclsConfig { max_iter: 2000, ..Default::default() },
            ..ConshConfig::default()
        };
        let fit = fit_volume_consh(&dwi, &responses, &cfg, false, || {}).unwrap();
        let n_wm = fit.wm.shape()[3];

        let (mut class_ok, mut class_n) = (0usize, 0usize);
        let mut wm_errors = Vec::new();
        let mut crossing_ok = 0usize;
        let mut crossing_n = 0usize;
        for x in 0..NX {
            for y in 0..NY {
                let t = tissue_at(x, y);
                if t == Tissue::Outside {
                    continue;
                }
                let z = NZ / 2;
                let wm: Vec<f64> = (0..n_wm).map(|k| fit.wm[(x, y, z, k)] as f64).collect();
                let (gm, csf) = (fit.gm[(x, y, z, 0)] as f64, fit.csf[(x, y, z, 0)] as f64);
                let total = wm[0] + gm + csf;
                let share = |v: f64| v / total.max(1e-12);
                let dominant = match t {
                    Tissue::Csf => share(csf),
                    Tissue::Gm => share(gm),
                    Tissue::Wm | Tissue::Crossing => share(wm[0]),
                    Tissue::Outside => unreachable!(),
                };
                class_n += 1;
                if dominant > 0.8 {
                    class_ok += 1;
                }
                let amp = amplitudes(&wm, 8, &sphere);
                let peak = amp
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                match t {
                    Tissue::Wm => wm_errors.push(angle_deg(sphere[peak], circumferential(x, y))),
                    Tissue::Crossing => {
                        // Both planted fibers carry a strong lobe; the in-plane
                        // radial direction, perpendicular to both, does not.
                        let max = amp[peak];
                        let along = |d: [f64; 3]| {
                            amplitudes(&wm, 8, &[[d[0] as f32, d[1] as f32, d[2] as f32]])[0]
                        };
                        let (dx, dy) = (x as f64 - CENTER, y as f64 - CENTER);
                        let r = (dx * dx + dy * dy).sqrt();
                        let radial = [dx / r, dy / r, 0.0];
                        crossing_n += 1;
                        if along(circumferential(x, y)) > 0.5 * max
                            && along([0.0, 0.0, 1.0]) > 0.5 * max
                            && along(radial) < 0.2 * max
                        {
                            crossing_ok += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
        wm_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_error = wm_errors[wm_errors.len() / 2];
        let p90_error = wm_errors[wm_errors.len() * 9 / 10];
        eprintln!(
            "{estimator:?}: tissue classes {class_ok}/{class_n}, WM peak error median {median_error:.1}° \
             p90 {p90_error:.1}°, crossings resolved {crossing_ok}/{crossing_n}"
        );
        assert!(class_ok as f64 >= 0.95 * class_n as f64, "{estimator:?}: {class_ok}/{class_n}");
        assert!(median_error < 2.0 && p90_error < 4.0, "{estimator:?}: {median_error} {p90_error}");
        assert!(crossing_ok as f64 >= 0.9 * crossing_n as f64, "{estimator:?}: {crossing_ok}/{crossing_n}");
    }
}

#[test]
fn consh_refuses_single_shell_data() {
    use cs_dmri::multitissue::consh_pipeline::check_consh_sampling;
    let full = synthetic_brain();
    // b=0 plus the single shell |q|² = 8 (b = 1500).
    let keep: Vec<usize> = (0..full.gtab.n_grads())
        .filter(|&i| full.gtab.bvals[i] == 0.0 || (full.gtab.bvals[i] - 1500.0).abs() < 1.0)
        .collect();
    let gtab = GradientTable::new(
        keep.iter().map(|&i| full.gtab.bvals[i]).collect(),
        keep.iter().map(|&i| full.gtab.bvecs[i]).collect(),
        Some(0.043),
        Some(0.011),
        None,
    )
    .unwrap();
    let data = full.data.select(ndarray::Axis(3), &keep);
    let single = DwiData::from_table(data, full.mask.clone(), gtab, BvecFrame::WorldRas, "s.nii.gz".into());
    let err = check_consh_sampling(&single).unwrap_err().to_string();
    assert!(err.contains("3 or more distinct b-values") && err.contains("b = 1500"), "{err}");
    assert!(check_consh_sampling(&full).is_ok());
}
