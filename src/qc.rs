// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//! Input-quality metrics for raw DWI series.
//!
//! - [`neighboring_dwi_correlation`] (NDC): the mean correlation between each
//!   DWI volume and its nearest neighbour in q-space. Yeh et al. (2019) flag
//!   NDC below 0.4 as a low-quality image.
//! - [`dwi_contrast`]: mean neighbour correlation divided by the mean
//!   correlation with a "contrast" volume whose q-vector is closest to
//!   perpendicular. DSI Studio's conventional reading is < 1.1 poor,
//!   1.1–1.3 fair, > 1.3 good: a series whose volumes correlate as strongly
//!   with perpendicular directions as with neighbouring ones carries little
//!   angular contrast.
//!
//! [`assess_input`] runs both on a loaded [`DwiData`] inside its mask, which
//! every DWI-consuming binary does before fitting.
//!
//! ## Provenance
//!
//! Ported from dipy's `dipy/stats/qc.py`: `find_qspace_neighbors` and
//! `neighboring_dwi_correlation` (dipy master, blob aa5bb46), and
//! `find_qspace_contrast` and `dwi_contrast` from dipy PR #4224
//! (https://github.com/dipy/dipy/pull/4224), which was still open and
//! unmerged when this was ported (2026-10). Behaviour matches dipy, including
//! its edge cases: neighbours are found with antipodal symmetry, the contrast
//! volume is not, and a volume with zero variance inside the mask yields a NaN
//! correlation that propagates into the mean.
//!
//! The ported functions are Copyright (c) 2008-2026, dipy developers, under
//! the BSD 3-Clause licence reproduced in LICENSE-DIPY; the rest of this file
//! is MIT OR Apache-2.0 like the rest of cs-dmri.
//!
//! References:
//! - Yeh, Liu, Hsu, Lee, Ge, Lin, Lin, Chen, Jhang & Tseng (2019),
//!   *"Differential tractography as a track-based biomarker for neuronal
//!   injury"*, NeuroImage 202:116131.
//! - DSI Studio quality control documentation:
//!   https://dsi-studio.labsolver.org/doc/gui_t1.html#step-t1a-quality-control-optional

use ndarray::{Array3, Array4};
use rayon::prelude::*;

use crate::io::dwi::DwiData;
use crate::qspace::GradientTable;

/// NDC below this flags a low-quality image (Yeh et al. 2019).
pub const NDC_LOW_THRESHOLD: f64 = 0.4;

/// DWI contrast below this is conventionally "poor".
pub const DWI_CONTRAST_POOR_THRESHOLD: f64 = 1.1;

/// DWI contrast above this is conventionally "good" (1.1–1.3 is "fair").
pub const DWI_CONTRAST_GOOD_THRESHOLD: f64 = 1.3;

/// Approximate q-space coordinates `√b · bvec` (diffusion times ignored), as
/// dipy computes them.
fn pseudo_qvecs(gtab: &GradientTable) -> Vec<[f64; 3]> {
    gtab.bvals
        .iter()
        .zip(&gtab.bvecs)
        .map(|(&b, v)| {
            let q = b.sqrt();
            [q * v[0], q * v[1], q * v[2]]
        })
        .collect()
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Map each b>0 volume to its nearest b>0 neighbour in approximate q-space,
/// counting `q` and `-q` as the same direction. Returns `(volume, neighbour)`
/// pairs in volume order; the mapping need not be symmetric. Ties go to the
/// lowest index, as with numpy's `argmin`.
///
/// Port of dipy's `find_qspace_neighbors`. Empty if there are fewer than two
/// b>0 volumes (dipy would pair the lone volume with a b=0 instead).
pub fn find_qspace_neighbors(gtab: &GradientTable) -> Vec<(usize, usize)> {
    let b0 = gtab.b0_mask();
    let qvecs = pseudo_qvecs(gtab);
    let dwi: Vec<usize> = (0..qvecs.len()).filter(|&i| !b0[i]).collect();
    if dwi.len() < 2 {
        return Vec::new();
    }
    dwi.iter()
        .map(|&i| {
            let q = qvecs[i];
            let mut best = (f64::INFINITY, usize::MAX);
            for (j, c) in qvecs.iter().enumerate() {
                if j == i || b0[j] {
                    continue;
                }
                let neg = [-c[0], -c[1], -c[2]];
                let d = dist(&q, c).min(dist(&q, &neg));
                if d < best.0 {
                    best = (d, j);
                }
            }
            (i, best.1)
        })
        .collect()
}

/// Map each b>0 volume to its contrast volume: the other b>0 volume closest to
/// the reference q-vector's perpendicular direction, as in Yeh's DSI Studio
/// implementation. For each candidate, its component perpendicular to the
/// reference is rescaled to the reference's magnitude, and the candidate
/// nearest that rescaled vector wins. Candidates parallel to the reference
/// (perpendicular component within 1e-8 of zero) are skipped.
///
/// Port of `find_qspace_contrast` from dipy PR #4224. Returns `None` if some
/// volume has no non-parallel candidate (where dipy raises `ValueError`), or
/// if there are fewer than two b>0 volumes.
pub fn find_qspace_contrast(gtab: &GradientTable) -> Option<Vec<(usize, usize)>> {
    let b0 = gtab.b0_mask();
    let qvecs = pseudo_qvecs(gtab);
    let dwi: Vec<usize> = (0..qvecs.len()).filter(|&i| !b0[i]).collect();
    if dwi.len() < 2 {
        return None;
    }
    dwi.iter()
        .map(|&i| {
            let q = qvecs[i];
            let q_norm_sq = dot(&q, &q);
            let q_norm = q_norm_sq.sqrt();
            let mut best = (f64::INFINITY, None);
            for &j in &dwi {
                if j == i {
                    continue;
                }
                let c = qvecs[j];
                let s = dot(&q, &c) / q_norm_sq;
                let mut perp = [c[0] - q[0] * s, c[1] - q[1] * s, c[2] - q[2] * s];
                // numpy.allclose(perp, 0.0): |x| <= atol (1e-8) element-wise.
                if perp.iter().all(|x| x.abs() <= 1e-8) {
                    continue;
                }
                let scale = q_norm / dot(&perp, &perp).sqrt();
                for x in &mut perp {
                    *x *= scale;
                }
                let d = dist(&c, &perp);
                if d < best.0 {
                    best = (d, Some(j));
                }
            }
            best.1.map(|j| (i, j))
        })
        .collect()
}

/// Voxel indices where `mask` is set, in C order (dipy's boolean indexing),
/// or every voxel when there is no mask.
fn voxel_list(shape: (usize, usize, usize), mask: Option<&Array3<bool>>) -> Vec<(usize, usize, usize)> {
    let (nx, ny, nz) = shape;
    (0..nx)
        .flat_map(|x| (0..ny).flat_map(move |y| (0..nz).map(move |z| (x, y, z))))
        .filter(|&(x, y, z)| mask.is_none_or(|m| m[(x, y, z)]))
        .collect()
}

/// Pearson correlation of two volumes over `voxels`, in f64 like
/// `numpy.corrcoef`. NaN if either volume is constant over `voxels`.
fn volume_correlation(data: &Array4<f32>, voxels: &[(usize, usize, usize)], a: usize, b: usize) -> f64 {
    let n = voxels.len() as f64;
    let (mut sa, mut sb) = (0.0, 0.0);
    for &(x, y, z) in voxels {
        sa += data[(x, y, z, a)] as f64;
        sb += data[(x, y, z, b)] as f64;
    }
    let (ma, mb) = (sa / n, sb / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for &(x, y, z) in voxels {
        let da = data[(x, y, z, a)] as f64 - ma;
        let db = data[(x, y, z, b)] as f64 - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    sab / (saa * sbb).sqrt()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Neighboring DWI Correlation: the mean, over b>0 volumes, of each volume's
/// correlation with its nearest q-space neighbour ([`find_qspace_neighbors`]).
/// Restrict to a brain `mask` where possible; without one the background
/// dominates the correlation. `None` with fewer than two b>0 volumes.
///
/// Port of dipy's `neighboring_dwi_correlation`.
pub fn neighboring_dwi_correlation(
    data: &Array4<f32>,
    gtab: &GradientTable,
    mask: Option<&Array3<bool>>,
) -> Option<f64> {
    let pairs = find_qspace_neighbors(gtab);
    if pairs.is_empty() {
        return None;
    }
    let s = data.shape();
    let voxels = voxel_list((s[0], s[1], s[2]), mask);
    let r: Vec<f64> = pairs
        .par_iter()
        .map(|&(a, b)| volume_correlation(data, &voxels, a, b))
        .collect();
    Some(mean(&r))
}

/// DWI contrast: mean neighbour correlation over mean contrast-volume
/// correlation ([`find_qspace_contrast`]). See the module docs for the
/// conventional thresholds. `None` when no contrast mapping exists.
///
/// Port of `dwi_contrast` from dipy PR #4224.
pub fn dwi_contrast(data: &Array4<f32>, gtab: &GradientTable, mask: Option<&Array3<bool>>) -> Option<f64> {
    let neighbors = find_qspace_neighbors(gtab);
    let contrast = find_qspace_contrast(gtab)?;
    debug_assert!(neighbors.iter().zip(&contrast).all(|(n, c)| n.0 == c.0));
    let s = data.shape();
    let voxels = voxel_list((s[0], s[1], s[2]), mask);
    let (rn, rc): (Vec<f64>, Vec<f64>) = neighbors
        .par_iter()
        .zip(contrast.par_iter())
        .map(|(&(a, n), &(_, c))| {
            (volume_correlation(data, &voxels, a, n), volume_correlation(data, &voxels, a, c))
        })
        .unzip();
    Some(mean(&rn) / mean(&rc))
}

/// Input-quality summary for a DWI series, computed inside its mask.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputQc {
    /// Neighboring DWI Correlation; `None` with fewer than two b>0 volumes.
    pub ndc: Option<f64>,
    /// DWI contrast; `None` if no contrast mapping exists (e.g. every b>0
    /// volume shares one direction).
    pub dwi_contrast: Option<f64>,
}

impl InputQc {
    /// Warnings for values in the low-quality range. NaN (a constant volume
    /// inside the mask) is reported too, since it hides the real value.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(v) = self.ndc {
            if v.is_nan() || v < NDC_LOW_THRESHOLD {
                out.push(format!(
                    "neighboring DWI correlation {v:.3} is below {NDC_LOW_THRESHOLD} \
                     (Yeh et al. 2019 low-quality threshold)"
                ));
            }
        }
        if let Some(v) = self.dwi_contrast {
            if v.is_nan() || v < DWI_CONTRAST_POOR_THRESHOLD {
                out.push(format!(
                    "DWI contrast {v:.3} is below {DWI_CONTRAST_POOR_THRESHOLD} (conventionally poor)"
                ));
            }
        }
        out
    }

    /// One-line summary, e.g. `NDC 0.712, DWI contrast 1.42 (good)`.
    pub fn summary(&self) -> String {
        let ndc = self.ndc.map_or("n/a".to_string(), |v| format!("{v:.3}"));
        let contrast = self.dwi_contrast.map_or("n/a".to_string(), |v| {
            let grade = if v < DWI_CONTRAST_POOR_THRESHOLD {
                "poor"
            } else if v <= DWI_CONTRAST_GOOD_THRESHOLD {
                "fair"
            } else {
                "good"
            };
            format!("{v:.3} ({grade})")
        });
        format!("NDC {ndc}, DWI contrast {contrast}")
    }
}

/// Compute [`InputQc`] for a loaded DWI series inside its mask.
pub fn assess_input(dwi: &DwiData) -> InputQc {
    InputQc {
        ndc: neighboring_dwi_correlation(&dwi.data, &dwi.gtab, Some(&dwi.mask)),
        dwi_contrast: dwi_contrast(&dwi.data, &dwi.gtab, Some(&dwi.mask)),
    }
}

/// Run [`assess_input`] and report it on stderr as `[tool] input QC: ...`.
/// The summary line is suppressed by `quiet`; low-quality warnings are not.
pub fn report_input_qc(tool: &str, dwi: &DwiData, quiet: bool) -> InputQc {
    let qc = assess_input(dwi);
    if !quiet {
        eprintln!("[{tool}] input QC: {}", qc.summary());
    }
    for w in qc.warnings() {
        eprintln!("[{tool}] WARNING: {w}");
    }
    qc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qspace::GradientTable;

    fn gtab(bvals: &[f64], bvecs: &[[f64; 3]]) -> GradientTable {
        GradientTable::new(bvals.to_vec(), bvecs.to_vec(), Some(0.03), Some(0.01), None).unwrap()
    }

    /// dipy's doctest for `find_qspace_neighbors`.
    #[test]
    fn neighbors_match_dipy_doctest() {
        let g = gtab(
            &[0.0, 1000.0, 1000.0, 2000.0],
            &[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.99, 0.0001, 0.0001], [1.0, 0.0, 0.0]],
        );
        assert_eq!(find_qspace_neighbors(&g), vec![(1, 2), (2, 1), (3, 1)]);
    }

    /// `-q` counts as the same direction as `q`.
    #[test]
    fn neighbors_are_antipodally_symmetric() {
        let g = gtab(
            &[0.0, 1000.0, 1000.0, 1000.0],
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.99, 0.1, 0.0]],
        );
        assert_eq!(find_qspace_neighbors(&g)[0], (1, 3));
    }

    /// The contrast volume is the one closest to perpendicular, and parallel
    /// candidates are skipped.
    #[test]
    fn contrast_picks_the_perpendicular_direction() {
        let s = 0.5_f64.sqrt();
        let g = gtab(
            &[0.0, 1000.0, 1000.0, 1000.0, 2000.0],
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [s, s, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        );
        let c = find_qspace_contrast(&g).unwrap();
        assert_eq!(c[0], (1, 3)); // x → y, not the 45° or the parallel b=2000 volume
        assert_eq!(c[2], (3, 1)); // y → x
    }

    #[test]
    fn contrast_is_none_when_every_direction_is_parallel() {
        let g = gtab(&[0.0, 1000.0, 2000.0], &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        assert!(find_qspace_contrast(&g).is_none());
    }

    #[test]
    fn correlation_matches_closed_form() {
        // Volume 1 = 2·volume 0 + 3 → r = 1; volume 2 = −volume 0 → r = −1.
        let mut data = Array4::<f32>::zeros((2, 2, 2, 3));
        let vox = voxel_list((2, 2, 2), None);
        for (k, &(x, y, z)) in vox.iter().enumerate() {
            let v = (k as f32).powi(2);
            data[(x, y, z, 0)] = v;
            data[(x, y, z, 1)] = 2.0 * v + 3.0;
            data[(x, y, z, 2)] = -v;
        }
        assert!((volume_correlation(&data, &vox, 0, 1) - 1.0).abs() < 1e-12);
        assert!((volume_correlation(&data, &vox, 0, 2) + 1.0).abs() < 1e-12);
    }

    /// Synthetic single-fibre-like series: each volume is a direction-dependent
    /// pattern, so neighbours correlate strongly and perpendicular volumes
    /// weakly — NDC high, contrast > 1. Pure noise drives both down.
    #[test]
    fn structured_series_scores_better_than_noise() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let n = 30;
        let mut bvals = vec![0.0];
        let mut bvecs = vec![[0.0, 0.0, 0.0]];
        for i in 0..n {
            let t = std::f64::consts::PI * i as f64 / n as f64;
            bvals.push(1000.0);
            bvecs.push([t.cos(), t.sin(), 0.0]);
        }
        let g = gtab(&bvals, &bvecs);
        let (nx, ny, nz) = (8, 8, 4);
        // Each voxel has its own in-plane fibre angle and a baseline (the
        // proton-density / T2 structure every real volume shares, which keeps
        // all inter-volume correlations positive); signal falls off with
        // alignment between gradient and fibre.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let angles: Vec<f64> = (0..nx * ny * nz).map(|_| rng.gen::<f64>() * std::f64::consts::PI).collect();
        let baseline: Vec<f64> = (0..nx * ny * nz).map(|_| 0.5 + rng.gen::<f64>()).collect();
        let mut structured = Array4::<f32>::zeros((nx, ny, nz, n + 1));
        let mut noise = Array4::<f32>::zeros((nx, ny, nz, n + 1));
        for (k, (x, y, z)) in voxel_list((nx, ny, nz), None).into_iter().enumerate() {
            let (fx, fy) = (angles[k].cos(), angles[k].sin());
            structured[(x, y, z, 0)] = baseline[k] as f32;
            noise[(x, y, z, 0)] = 1.0;
            for v in 1..=n {
                let c = bvecs[v][0] * fx + bvecs[v][1] * fy;
                structured[(x, y, z, v)] = (baseline[k] * (0.6 + 0.4 * (-3.0 * c * c).exp())) as f32;
                noise[(x, y, z, v)] = rng.gen::<f32>();
            }
        }
        let ndc_s = neighboring_dwi_correlation(&structured, &g, None).unwrap();
        let ndc_n = neighboring_dwi_correlation(&noise, &g, None).unwrap();
        let con_s = dwi_contrast(&structured, &g, None).unwrap();
        assert!(ndc_s > 0.9, "structured NDC {ndc_s}");
        assert!(ndc_n.abs() < 0.3, "noise NDC {ndc_n}");
        assert!(con_s > DWI_CONTRAST_GOOD_THRESHOLD, "structured contrast {con_s}");
        let qc = InputQc { ndc: Some(ndc_n), dwi_contrast: Some(1.0) };
        assert_eq!(qc.warnings().len(), 2);
        let qc = InputQc { ndc: Some(ndc_s), dwi_contrast: Some(con_s) };
        assert!(qc.warnings().is_empty(), "{:?}", qc.warnings());
    }
}
