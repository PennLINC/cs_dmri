// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//! Image-quality metrics for raw and preprocessed DWI series.
//!
//! | Metric | What it measures |
//! |---|---|
//! | [`neighboring_dwi_correlation`] (NDC) | Mean correlation between each b>0 volume and its nearest q-space neighbour. Motion, eddy currents and signal dropout lower it. Below 0.4 flags a low-quality image (Yeh et al. 2019). |
//! | [`dwi_contrast_ratio`] | Mean neighbour correlation ÷ mean correlation with each volume's most nearly perpendicular "contrast" volume. Near 1 the series carries little angular contrast; conventionally < 1.1 poor, 1.1–1.3 fair, > 1.3 good. |
//! | [`outlier_slices`] | Slices that don't lie between their two adjacent slices in the same volume (signal dropout, corrupted slices). No other volume is consulted. |
//! | [`fixel_coherence`] | Share (by FA weight) of above-threshold voxels whose principal diffusion direction continues coherently into a neighbouring voxel (odx-rs primary coherence). |
//!
//! [`assess`] computes the model-free metrics in one pass and returns a
//! serialisable [`QcReport`]; `fixel_coherence` needs a tensor fit and is
//! computed separately.
//!
//! ## Conventions
//!
//! - **b=0 volumes** are those with `b ≤ b0_threshold` (a parameter; qsiprep
//!   passes its own `b0_threshold`).
//! - **Neighbours** are found in approximate q-space `√b · bvec`, treating `q`
//!   and `−q` as the same direction, exactly as dipy does. Repeated
//!   acquisitions of a q-space point may pair with each other. In merged AP+PA
//!   series, where every volume's nearest neighbour is its twin from the other
//!   run, that pairing measured slightly *lower* NDC than excluding twins
//!   (0.786 vs 0.803 on a HASC55 AP+PA series): the twin carries cross-run
//!   distortion and motion differences, so pairing with it is a consistency
//!   check rather than an inflation.
//! - **NDC averages over every b>0 volume.** DSI Studio instead keeps a pair
//!   only when the volume's index exceeds its neighbour's, which makes NDC
//!   depend on acquisition order (0.970–0.977 under random reorderings of one
//!   HASC92 series, against a reorder-invariant 0.9733 here).
//! - **Contrast volume:** for each candidate, its component perpendicular to the
//!   reference q-vector is rescaled to the reference's length; the candidate
//!   nearest that vector wins.
//! - **Masks:** every metric takes an optional mask. Without one the background
//!   dominates the correlations, so masked variants are what to compare across
//!   scans. The contrast ratio is especially mask-sensitive.
//!
//! ## Provenance
//!
//! The neighbour search and NDC are adapted from dipy's
//! `dipy/stats/qc.py::find_qspace_neighbors` / `neighboring_dwi_correlation`,
//! and the contrast-volume search from dipy PR #4224, which ports Fang-Cheng
//! Yeh's DSI Studio definition with his permission. The adapted portions are
//! Copyright (c) 2008-2026, dipy developers, under the BSD 3-Clause licence in
//! LICENSE-DIPY; the rest of this file is MIT OR Apache-2.0.
//!
//! References:
//! - Yeh, Liu, Hsu, Lee, Ge, Lin, Lin, Chen, Jhang & Tseng (2019),
//!   *"Differential tractography as a track-based biomarker for neuronal
//!   injury"*, NeuroImage 202:116131.

use ndarray::{Array2, ArrayView3, ArrayView4};
use rayon::prelude::*;
use serde::Serialize;

use crate::io::dwi::DwiData;
use crate::qspace::GradientTable;
use crate::{CsDmriError, Result};

/// NDC below this flags a low-quality image (Yeh et al. 2019).
pub const NDC_LOW_THRESHOLD: f64 = 0.4;
/// DWI contrast ratio below this is conventionally "poor".
pub const DWI_CONTRAST_POOR_THRESHOLD: f64 = 1.1;
/// DWI contrast ratio above this is conventionally "good" (1.1–1.3 is "fair").
pub const DWI_CONTRAST_GOOD_THRESHOLD: f64 = 1.3;

// ---------------------------------------------------------------- q-space

fn qvecs(bvals: &[f64], bvecs: &[[f64; 3]]) -> Vec<[f64; 3]> {
    bvals
        .iter()
        .zip(bvecs)
        .map(|(&b, v)| {
            let q = b.max(0.0).sqrt();
            [q * v[0], q * v[1], q * v[2]]
        })
        .collect()
}

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Indices of the b>0 volumes.
fn dwi_indices(bvals: &[f64], b0_threshold: f64) -> Vec<usize> {
    (0..bvals.len()).filter(|&i| bvals[i] > b0_threshold).collect()
}

/// For each b>0 volume, its nearest other b>0 volume in approximate q-space
/// (antipodally symmetric), as `(volume, neighbour)` pairs in volume order.
/// Ties go to the lowest index. Matches dipy's `find_qspace_neighbors`, except
/// that a lone b>0 volume has no neighbour here (dipy pairs it with a b=0).
pub fn find_qspace_neighbors(gtab: &GradientTable, b0_threshold: f64) -> Vec<(usize, usize)> {
    let q = qvecs(&gtab.bvals, &gtab.bvecs);
    let dwi = dwi_indices(&gtab.bvals, b0_threshold);
    dwi.iter()
        .filter_map(|&i| {
            let mut best: Option<(f64, usize)> = None;
            for &j in &dwi {
                if j == i {
                    continue;
                }
                let d = norm(&sub(&q[i], &q[j])).min(norm(&add(&q[i], &q[j])));
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, j));
                }
            }
            best.map(|(_, j)| (i, j))
        })
        .collect()
}

/// For each b>0 volume, its contrast volume: the b>0 candidate closest to the
/// reference's perpendicular direction. A candidate's component perpendicular
/// to the reference q-vector is rescaled to the reference's length, and the
/// candidate nearest that vector wins. Parallel candidates (no perpendicular
/// component, which includes repeats of the reference's direction) are
/// skipped, as in dipy PR #4224.
pub fn find_qspace_contrast(gtab: &GradientTable, b0_threshold: f64) -> Vec<(usize, usize)> {
    let q = qvecs(&gtab.bvals, &gtab.bvecs);
    let dwi = dwi_indices(&gtab.bvals, b0_threshold);
    dwi.iter()
        .filter_map(|&i| {
            let qi = q[i];
            let len2 = dot(&qi, &qi);
            let len = len2.sqrt();
            let mut best: Option<(f64, usize)> = None;
            for &j in &dwi {
                if j == i {
                    continue;
                }
                let c = q[j];
                let s = dot(&qi, &c) / len2;
                let perp = [c[0] - qi[0] * s, c[1] - qi[1] * s, c[2] - qi[2] * s];
                let pn = norm(&perp);
                if pn <= 1e-8 * len.max(1.0) {
                    continue;
                }
                let scaled = [perp[0] * len / pn, perp[1] * len / pn, perp[2] * len / pn];
                let d = norm(&sub(&c, &scaled));
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, j));
                }
            }
            best.map(|(_, j)| (i, j))
        })
        .collect()
}

// ----------------------------------------------------------- correlations

/// Voxel indices where `mask` is set (C order), or every voxel.
fn voxel_list(shape: (usize, usize, usize), mask: Option<ArrayView3<bool>>) -> Vec<(usize, usize, usize)> {
    let (nx, ny, nz) = shape;
    (0..nx)
        .flat_map(|x| (0..ny).flat_map(move |y| (0..nz).map(move |z| (x, y, z))))
        .filter(|&(x, y, z)| mask.is_none_or(|m| m[(x, y, z)]))
        .collect()
}

/// Pearson correlation and means of volumes `a` and `b` over `voxels`
/// (accumulated in f64). Correlation is NaN if either is constant.
fn corr_and_means(
    data: &ArrayView4<f32>,
    voxels: &[(usize, usize, usize)],
    a: usize,
    b: usize,
) -> (f64, f64, f64) {
    let n = voxels.len() as f64;
    if voxels.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
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
    (sab / (saa * sbb).sqrt(), ma, mb)
}

fn mean(xs: &[f64]) -> Option<f64> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
}

fn check_shapes(data: &ArrayView4<f32>, gtab: &GradientTable, mask: Option<ArrayView3<bool>>) -> Result<()> {
    let s = data.shape();
    if s[3] != gtab.n_grads() {
        return Err(CsDmriError::Dimension(format!(
            "data has {} volumes but the gradient table has {}",
            s[3],
            gtab.n_grads()
        )));
    }
    if let Some(m) = mask {
        if m.shape() != &s[..3] {
            return Err(CsDmriError::Dimension(format!(
                "mask shape {:?} does not match data spatial shape {:?}",
                m.shape(),
                &s[..3]
            )));
        }
    }
    Ok(())
}

/// Neighboring DWI Correlation: mean over b>0 volumes of each volume's
/// correlation with its q-space neighbour ([`find_qspace_neighbors`]), over
/// `mask` (or every voxel). `None` when no volume has a neighbour.
pub fn neighboring_dwi_correlation(
    data: ArrayView4<f32>,
    gtab: &GradientTable,
    mask: Option<ArrayView3<bool>>,
    b0_threshold: f64,
) -> Result<Option<f64>> {
    check_shapes(&data, gtab, mask)?;
    let pairs = find_qspace_neighbors(gtab, b0_threshold);
    let s = data.shape();
    let voxels = voxel_list((s[0], s[1], s[2]), mask);
    let r: Vec<f64> = pairs
        .par_iter()
        .map(|&(a, b)| corr_and_means(&data, &voxels, a, b).0)
        .collect();
    Ok(mean(&r))
}

/// DWI contrast ratio: mean neighbour correlation ÷ mean contrast-volume
/// correlation, over `mask` (or every voxel). Both means run over the volumes
/// that have both a neighbour and a contrast volume. `None` if there are none.
pub fn dwi_contrast_ratio(
    data: ArrayView4<f32>,
    gtab: &GradientTable,
    mask: Option<ArrayView3<bool>>,
    b0_threshold: f64,
) -> Result<Option<f64>> {
    check_shapes(&data, gtab, mask)?;
    let neighbors: std::collections::HashMap<usize, usize> =
        find_qspace_neighbors(gtab, b0_threshold).into_iter().collect();
    let triples: Vec<(usize, usize, usize)> = find_qspace_contrast(gtab, b0_threshold)
        .into_iter()
        .filter_map(|(i, c)| neighbors.get(&i).map(|&n| (i, n, c)))
        .collect();
    let s = data.shape();
    let voxels = voxel_list((s[0], s[1], s[2]), mask);
    let (rn, rc): (Vec<f64>, Vec<f64>) = triples
        .par_iter()
        .map(|&(i, n, c)| {
            (
                corr_and_means(&data, &voxels, i, n).0,
                corr_and_means(&data, &voxels, i, c).0,
            )
        })
        .unzip();
    Ok(match (mean(&rn), mean(&rc)) {
        (Some(a), Some(b)) => Some(a / b),
        _ => None,
    })
}

// --------------------------------------------------------- outlier slices

/// Settings for [`outlier_slices`].
#[derive(Debug, Clone, Copy, Serialize)]
pub struct OutlierSliceOptions {
    /// Spatial axis the slices are taken along (0, 1 or 2). Default 2.
    pub axis: usize,
    /// Slices with fewer in-mask voxels than this are not scored. Default 100.
    pub min_voxels: usize,
    /// In-plane Gaussian smoothing (σ, voxels) applied before comparing slices,
    /// so noise doesn't swamp the comparison in low-SNR volumes. Default 2.
    pub smoothing_sigma: f64,
    /// Flag a slice whose inconsistency ratio exceeds this. Default 2.5.
    pub threshold: f64,
}

impl Default for OutlierSliceOptions {
    fn default() -> Self {
        Self {
            axis: 2,
            min_voxels: 100,
            smoothing_sigma: 2.0,
            threshold: 2.5,
        }
    }
}

/// Per-volume, per-slice outlier detection.
#[derive(Debug, Clone, Serialize)]
pub struct OutlierSlices {
    /// `(n_volumes, n_slices)`: true where the slice was flagged. Edge and
    /// unscored slices are false.
    #[serde(skip)]
    pub flags: Array2<bool>,
    /// The inconsistency ratio of each slice ([`outlier_slices`]); NaN where
    /// not scored.
    #[serde(skip)]
    pub ratio: Array2<f64>,
    /// Flagged `(volume, slice)` pairs, in volume then slice order.
    pub flagged: Vec<(usize, usize)>,
    pub options: OutlierSliceOptions,
}

impl OutlierSlices {
    pub fn count(&self) -> usize {
        self.flagged.len()
    }
}

/// Normalised 1-D Gaussian kernel, radius `round(4σ)` (scipy's default).
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let r = (4.0 * sigma + 0.5) as isize;
    let w: Vec<f64> = (-r..=r).map(|x| (-0.5 * (x * x) as f64 / (sigma * sigma)).exp()).collect();
    let t: f64 = w.iter().sum();
    w.into_iter().map(|x| x / t).collect()
}

/// Mirror index into `0..n` with half-sample symmetric boundaries
/// (`d c b a | a b c d`, scipy's `reflect`).
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let period = 2 * n;
    let mut j = i.rem_euclid(period);
    if j >= n {
        j = period - 1 - j;
    }
    j as usize
}

/// One volume, smoothed in the two axes other than `slice_axis`.
fn smooth_in_plane(data: &ArrayView4<f32>, v: usize, slice_axis: usize, kernel: &[f64]) -> ndarray::Array3<f32> {
    let s = data.shape();
    let dims = [s[0], s[1], s[2]];
    let r = (kernel.len() / 2) as isize;
    let mut cur = ndarray::Array3::<f32>::from_shape_fn((dims[0], dims[1], dims[2]), |(x, y, z)| data[(x, y, z, v)]);
    if kernel.len() == 1 {
        return cur;
    }
    for axis in (0..3).filter(|&a| a != slice_axis) {
        let n = dims[axis];
        let mut next = ndarray::Array3::<f32>::zeros((dims[0], dims[1], dims[2]));
        for ((x, y, z), out) in next.indexed_iter_mut() {
            let idx = [x, y, z];
            let mut acc = 0.0f64;
            for (t, w) in kernel.iter().enumerate() {
                let mut j = idx;
                j[axis] = reflect(idx[axis] as isize + t as isize - r, n);
                acc += w * cur[(j[0], j[1], j[2])] as f64;
            }
            *out = acc as f32;
        }
        cur = next;
    }
    cur
}

/// Detect corrupted slices within each volume, without reference to any other
/// volume. After in-plane smoothing, every interior slice `k` along `axis` is
/// compared with the voxelwise average of slices `k−1` and `k+1`, over slice
/// `k`'s in-mask voxels:
///
/// ```text
///   ratio_k = mean|I_k − (I_{k−1} + I_{k+1})/2|  /  (½ · mean|I_{k+1} − I_{k−1}|)
/// ```
///
/// A slice consistent with its neighbours lies between them and scores below
/// about 2 even in raw, low-SNR data (maximum 2.39 over ~64,000 slices of eight
/// qsiprep test series, raw and preprocessed). Signal dropout or a corrupted
/// slice pushes it away from both neighbours. A slice is flagged when the ratio
/// exceeds `threshold`; on those series a dropout to 50% signal was caught 90%
/// of the time and to 70% about two times in three.
pub fn outlier_slices(
    data: ArrayView4<f32>,
    mask: Option<ArrayView3<bool>>,
    opts: &OutlierSliceOptions,
) -> Result<OutlierSlices> {
    let s = data.shape();
    if let Some(m) = mask {
        if m.shape() != &s[..3] {
            return Err(CsDmriError::Dimension(format!(
                "mask shape {:?} does not match data spatial shape {:?}",
                m.shape(),
                &s[..3]
            )));
        }
    }
    if opts.axis > 2 {
        return Err(CsDmriError::Other(format!("slice axis must be 0, 1 or 2, got {}", opts.axis)));
    }
    if !(opts.smoothing_sigma >= 0.0) {
        return Err(CsDmriError::Other(format!("smoothing sigma must be ≥ 0, got {}", opts.smoothing_sigma)));
    }
    let (n_vol, n_slice) = (s[3], s[opts.axis]);
    let mut per_slice: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n_slice];
    for v in voxel_list((s[0], s[1], s[2]), mask) {
        per_slice[[v.0, v.1, v.2][opts.axis]].push(v);
    }
    let scored: Vec<bool> = (0..n_slice)
        .map(|k| k > 0 && k + 1 < n_slice && per_slice[k].len() >= opts.min_voxels)
        .collect();
    let kernel = if opts.smoothing_sigma > 0.0 { gaussian_kernel(opts.smoothing_sigma) } else { vec![1.0] };
    let shift = |(x, y, z): (usize, usize, usize), d: isize| {
        let mut c = [x, y, z];
        c[opts.axis] = (c[opts.axis] as isize + d) as usize;
        (c[0], c[1], c[2])
    };

    let rows: Vec<Vec<f64>> = (0..n_vol)
        .into_par_iter()
        .map(|v| {
            let sm = smooth_in_plane(&data, v, opts.axis, &kernel);
            (0..n_slice)
                .map(|k| {
                    if !scored[k] {
                        return f64::NAN;
                    }
                    let (mut dev, mut spread) = (0.0f64, 0.0f64);
                    for &p in &per_slice[k] {
                        let (a, b) = (shift(p, -1), shift(p, 1));
                        let (o, lo, hi) = (sm[p] as f64, sm[a] as f64, sm[b] as f64);
                        dev += (o - 0.5 * (lo + hi)).abs();
                        spread += 0.5 * (hi - lo).abs();
                    }
                    if spread > 0.0 { dev / spread } else { f64::NAN }
                })
                .collect()
        })
        .collect();

    let mut flags = Array2::<bool>::from_elem((n_vol, n_slice), false);
    let mut ratio = Array2::<f64>::from_elem((n_vol, n_slice), f64::NAN);
    for (v, row) in rows.into_iter().enumerate() {
        for (k, r) in row.into_iter().enumerate() {
            ratio[(v, k)] = r;
            flags[(v, k)] = r > opts.threshold;
        }
    }
    let flagged = flags.indexed_iter().filter(|(_, f)| **f).map(|((v, k), _)| (v, k)).collect();
    Ok(OutlierSlices {
        flags,
        ratio,
        flagged,
        options: *opts,
    })
}

// --------------------------------------------------------- fixel coherence

/// Settings for [`fixel_coherence`].
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CoherenceOptions {
    /// Drop the lowest `quantile` of FA (over voxels with a direction) before
    /// scoring. Default 0.1 (odx-rs `DEFAULT_QC_QUANTILE`).
    pub quantile: f32,
    /// Maximum angle (degrees) between neighbouring directions for them to
    /// count as connected. Default 15.
    pub angle_degrees: f32,
}

impl Default for CoherenceOptions {
    fn default() -> Self {
        Self {
            quantile: odx_rs::DEFAULT_QC_QUANTILE,
            angle_degrees: 15.0,
        }
    }
}

/// Coherence of the principal-direction field.
#[derive(Debug, Clone, Serialize)]
pub struct CoherenceReport {
    /// FA-weighted share of evaluated voxels connected to a coherent neighbour,
    /// in [0, 1]. `None` if nothing passed the threshold.
    pub coherence: Option<f64>,
    /// d ln(coherence) / d ln(threshold); near 0 means the index does not
    /// depend on where the FA cut fell.
    pub threshold_elasticity: Option<f64>,
    pub evaluated_voxels: usize,
    pub connected_voxels: usize,
    pub fa_threshold: Option<f32>,
    pub options: CoherenceOptions,
}

/// Fixel coherence of a single-direction-per-voxel field: odx-rs's primary
/// coherence, with `fa` as the weighting and thresholding metric.
///
/// `principal_dir` is `(X, Y, Z, 3)` in world RAS when `world_directions`,
/// else in the image's voxel-axis frame (rotated to world with `affine`).
/// Voxels outside `mask`, or with zero FA or a zero direction, carry no fixel.
pub fn fixel_coherence(
    principal_dir: ArrayView4<f32>,
    fa: ArrayView3<f32>,
    mask: ArrayView3<bool>,
    affine: [[f64; 4]; 4],
    world_directions: bool,
    opts: &CoherenceOptions,
) -> Result<CoherenceReport> {
    use crate::io::odx_out::{canonicalize_array3_f32, canonicalize_array4, canonicalize_bool_mask};
    use odx_rs::{
        CanonTransform, CoherenceMode, FixelQcOptions, OdxBuilder, ThresholdMode,
        coherence_threshold_elasticity, compute_primary_coherence, dtype::DType,
    };
    fn err<E: std::fmt::Display>(e: E) -> CsDmriError {
        CsDmriError::Other(format!("fixel coherence: {e}"))
    }
    let s = fa.shape();
    if principal_dir.shape() != [s[0], s[1], s[2], 3] || mask.shape() != s {
        return Err(CsDmriError::Dimension(format!(
            "principal_dir {:?}, fa {:?} and mask {:?} must share a grid (principal_dir with 3 channels)",
            principal_dir.shape(),
            s,
            mask.shape()
        )));
    }
    let rot = if world_directions {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    } else {
        crate::qspace::affine_rotation(&affine)
    };

    // Canonical RAS+ grid, as every ODX this crate writes. Directions are
    // world vectors, so reindexing voxels leaves them unchanged.
    let canon = CanonTransform::from_affine(affine);
    let (pd, c_affine) = canonicalize_array4(&principal_dir.to_owned(), affine, &canon).map_err(err)?;
    let fa_c = canonicalize_array3_f32(&fa.to_owned(), affine, &canon).map_err(err)?;
    let mask_c = canonicalize_bool_mask(&mask.to_owned(), affine, &canon).map_err(err)?;
    let cs = fa_c.shape();
    let (nx, ny, nz) = (cs[0], cs[1], cs[2]);

    let mut mask_bytes = Vec::with_capacity(nx * ny * nz);
    let mut voxels = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                mask_bytes.push(mask_c[(x, y, z)] as u8);
                if mask_c[(x, y, z)] {
                    voxels.push((x, y, z));
                }
            }
        }
    }
    let mut builder = OdxBuilder::new(c_affine, [nx as u64, ny as u64, nz as u64], mask_bytes);
    let mut amplitude: Vec<u8> = Vec::new();
    for &(x, y, z) in &voxels {
        let d = [pd[(x, y, z, 0)] as f64, pd[(x, y, z, 1)] as f64, pd[(x, y, z, 2)] as f64];
        let w = [
            rot[0][0] * d[0] + rot[0][1] * d[1] + rot[0][2] * d[2],
            rot[1][0] * d[0] + rot[1][1] * d[1] + rot[1][2] * d[2],
            rot[2][0] * d[0] + rot[2][1] * d[1] + rot[2][2] * d[2],
        ];
        let n = norm(&w);
        let f = fa_c[(x, y, z)];
        if n > 0.0 && f.is_finite() && f > 0.0 {
            builder.push_voxel_peaks(&[[(w[0] / n) as f32, (w[1] / n) as f32, (w[2] / n) as f32]]);
            amplitude.extend_from_slice(&f.to_le_bytes());
        } else {
            builder.push_voxel_peaks(&[]);
        }
    }
    builder.set_dpf_data("amplitude", amplitude, 1, DType::Float32);
    let odx = builder.finalize().map_err(|e| CsDmriError::Other(format!("fixel coherence ODX: {e}")))?;

    let qc_opts = FixelQcOptions {
        primary_metric: Some("amplitude".to_string()),
        threshold: ThresholdMode::Quantile(opts.quantile),
        angle_degrees: opts.angle_degrees,
    };
    let report = compute_primary_coherence(&odx, &qc_opts).map_err(err)?;
    let elasticity = coherence_threshold_elasticity(&odx, &qc_opts, CoherenceMode::Primary).map_err(err)?;
    Ok(CoherenceReport {
        coherence: report.coherence_index,
        threshold_elasticity: elasticity,
        evaluated_voxels: report.evaluated_voxels,
        connected_voxels: report.connected_voxels,
        fa_threshold: report.threshold_value,
        options: *opts,
    })
}

// ------------------------------------------------------------------ report

/// Settings for [`assess`].
#[derive(Debug, Clone, Copy, Serialize)]
pub struct QcOptions {
    /// Volumes with `b ≤ b0_threshold` are b=0. Default 50.
    pub b0_threshold: f64,
    pub outlier_slices: OutlierSliceOptions,
}

impl Default for QcOptions {
    fn default() -> Self {
        Self {
            b0_threshold: 50.0,
            outlier_slices: OutlierSliceOptions::default(),
        }
    }
}

/// Model-free QC metrics for one series. Unmasked variants use every voxel;
/// masked variants (and outlier slices) use the mask passed to [`assess`].
#[derive(Debug, Clone, Serialize)]
pub struct QcReport {
    pub dimensions: [usize; 3],
    pub n_volumes: usize,
    pub n_dwi_volumes: usize,
    pub n_b0_volumes: usize,
    pub max_b: f64,
    pub mask_voxels: Option<usize>,
    pub ndc: Option<f64>,
    pub ndc_masked: Option<f64>,
    pub dwi_contrast_ratio: Option<f64>,
    pub dwi_contrast_ratio_masked: Option<f64>,
    pub outlier_slices: OutlierSlices,
    pub options: QcOptions,
}

impl QcReport {
    pub fn n_outlier_slices(&self) -> usize {
        self.outlier_slices.count()
    }

    /// Values in the conventional low-quality range, as messages. Masked values
    /// are preferred where available.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(v) = self.ndc_masked.or(self.ndc) {
            if v.is_nan() || v < NDC_LOW_THRESHOLD {
                out.push(format!(
                    "neighboring DWI correlation {v:.3} is below {NDC_LOW_THRESHOLD} \
                     (Yeh et al. 2019 low-quality threshold)"
                ));
            }
        }
        if let Some(v) = self.dwi_contrast_ratio_masked.or(self.dwi_contrast_ratio) {
            if v.is_nan() || v < DWI_CONTRAST_POOR_THRESHOLD {
                out.push(format!(
                    "DWI contrast ratio {v:.3} is below {DWI_CONTRAST_POOR_THRESHOLD} (conventionally poor)"
                ));
            }
        }
        out
    }
}

/// Grade a contrast ratio: "poor", "fair" or "good".
pub fn contrast_grade(v: f64) -> &'static str {
    if v < DWI_CONTRAST_POOR_THRESHOLD {
        "poor"
    } else if v <= DWI_CONTRAST_GOOD_THRESHOLD {
        "fair"
    } else {
        "good"
    }
}

/// Model-free QC of `data` (X, Y, Z, N). Outlier slices are scored inside
/// `mask` when given, else over every voxel.
pub fn assess(
    data: ArrayView4<f32>,
    gtab: &GradientTable,
    mask: Option<ArrayView3<bool>>,
    opts: &QcOptions,
) -> Result<QcReport> {
    check_shapes(&data, gtab, mask)?;
    let s = data.shape();
    let b0 = opts.b0_threshold;
    let n_dwi = gtab.bvals.iter().filter(|&&b| b > b0).count();
    let masked = |f: fn(ArrayView4<f32>, &GradientTable, Option<ArrayView3<bool>>, f64) -> Result<Option<f64>>| {
        mask.map(|m| f(data, gtab, Some(m), b0)).transpose().map(Option::flatten)
    };
    Ok(QcReport {
        dimensions: [s[0], s[1], s[2]],
        n_volumes: s[3],
        n_dwi_volumes: n_dwi,
        n_b0_volumes: s[3] - n_dwi,
        max_b: gtab.bvals.iter().cloned().fold(0.0, f64::max),
        mask_voxels: mask.map(|m| m.iter().filter(|&&v| v).count()),
        ndc: neighboring_dwi_correlation(data, gtab, None, b0)?,
        ndc_masked: masked(neighboring_dwi_correlation)?,
        dwi_contrast_ratio: dwi_contrast_ratio(data, gtab, None, b0)?,
        dwi_contrast_ratio_masked: masked(dwi_contrast_ratio)?,
        outlier_slices: outlier_slices(data, mask, &opts.outlier_slices)?,
        options: *opts,
    })
}

// ------------------------------------------------------- tabular output

/// One column of the flat QC table, with a BIDS-style description.
#[derive(Debug, Clone, Serialize)]
pub struct QcColumn {
    pub name: &'static str,
    #[serde(rename = "LongName")]
    pub long_name: &'static str,
    #[serde(rename = "Description")]
    pub description: &'static str,
    #[serde(rename = "Units", skip_serializing_if = "Option::is_none")]
    pub units: Option<&'static str>,
    /// The DSI Studio column (as named in qsiprep) this one replaces, when the
    /// computation changed enough to warrant a new name.
    #[serde(rename = "Replaces", skip_serializing_if = "Option::is_none")]
    pub replaces: Option<&'static str>,
}

/// The QC table's columns, in output order.
pub fn qc_columns() -> Vec<QcColumn> {
    let c = |name, long_name, description, units, replaces| QcColumn {
        name,
        long_name,
        description,
        units,
        replaces,
    };
    vec![
        c("dimension_x", "Image dimension (x)", "Number of voxels along the first image axis.", None, None),
        c("dimension_y", "Image dimension (y)", "Number of voxels along the second image axis.", None, None),
        c("dimension_z", "Image dimension (z)", "Number of voxels along the third image axis.", None, None),
        c("voxel_size_x", "Voxel size (x)", "Voxel spacing along the first image axis.", Some("mm"), None),
        c("voxel_size_y", "Voxel size (y)", "Voxel spacing along the second image axis.", Some("mm"), None),
        c("voxel_size_z", "Voxel size (z)", "Voxel spacing along the third image axis.", Some("mm"), None),
        c("max_b", "Maximum b-value", "Largest b-value in the gradient table.", Some("s/mm^2"), None),
        c(
            "n_dwi_volumes",
            "Diffusion-weighted volumes",
            "Number of volumes with b above the b=0 threshold.",
            None,
            Some("num_directions (which counted volumes, not unique directions)"),
        ),
        c("n_b0_volumes", "b=0 volumes", "Number of volumes with b at or below the b=0 threshold.", None, None),
        c(
            "ndc",
            "Neighboring DWI correlation",
            "Mean, over every b>0 volume, of the correlation between the volume and its nearest other b>0 \
             volume in q-space (antipodally symmetric), over all voxels; dipy's definition. Unlike DSI \
             Studio's version it does not depend on volume order. Below 0.4 flags a low-quality image \
             (Yeh et al. 2019).",
            None,
            Some("neighbor_corr"),
        ),
        c(
            "ndc_masked",
            "Neighboring DWI correlation (masked)",
            "As ndc, computed inside the brain mask.",
            None,
            Some("masked_neighbor_corr (which used DSI Studio's internal mask)"),
        ),
        c(
            "dwi_contrast_ratio",
            "DWI contrast ratio",
            "Mean neighbour correlation divided by the mean correlation with each volume's most nearly \
             perpendicular q-space volume, over all voxels. Highly mask-dependent; prefer the masked value.",
            None,
            Some("dwi_contrast"),
        ),
        c(
            "dwi_contrast_ratio_masked",
            "DWI contrast ratio (masked)",
            "As dwi_contrast_ratio, inside the brain mask. Conventionally below 1.1 poor, 1.1-1.3 fair, \
             above 1.3 good.",
            None,
            Some("dwi_contrast (which used DSI Studio's internal mask)"),
        ),
        c(
            "n_outlier_slices",
            "Outlier slices",
            "Number of (volume, slice) pairs that do not lie between their two adjacent slices in the same \
             volume: after in-plane smoothing (sigma 2 voxels), the mean absolute deviation from the adjacent \
             slices' average exceeds 2.5 times half their mean absolute difference, inside the brain mask. \
             No other volume is consulted.",
            None,
            Some("num_bad_slices"),
        ),
        c(
            "fixel_coherence",
            "Fixel coherence",
            "FA-weighted fraction (0-1) of voxels above the 10th FA percentile whose RESTORE principal \
             direction continues within 15 degrees into a neighbouring voxel (odx-rs primary coherence).",
            None,
            Some("coherence_index (an unbounded index from a GQI fib)"),
        ),
    ]
}

impl QcReport {
    /// The flat table row, keyed by [`qc_columns`] names. `voxel_size` and
    /// `fixel_coherence` come from outside the model-free report; missing
    /// values are `None`.
    pub fn row(&self, voxel_size: Option<[f64; 3]>, fixel_coherence: Option<f64>) -> Vec<(&'static str, Option<f64>)> {
        let vs = |i: usize| voxel_size.map(|v| v[i]);
        vec![
            ("dimension_x", Some(self.dimensions[0] as f64)),
            ("dimension_y", Some(self.dimensions[1] as f64)),
            ("dimension_z", Some(self.dimensions[2] as f64)),
            ("voxel_size_x", vs(0)),
            ("voxel_size_y", vs(1)),
            ("voxel_size_z", vs(2)),
            ("max_b", Some(self.max_b)),
            ("n_dwi_volumes", Some(self.n_dwi_volumes as f64)),
            ("n_b0_volumes", Some(self.n_b0_volumes as f64)),
            ("ndc", self.ndc),
            ("ndc_masked", self.ndc_masked),
            ("dwi_contrast_ratio", self.dwi_contrast_ratio),
            ("dwi_contrast_ratio_masked", self.dwi_contrast_ratio_masked),
            ("n_outlier_slices", Some(self.n_outlier_slices() as f64)),
            ("fixel_coherence", fixel_coherence),
        ]
    }
}

/// Voxel spacing (column norms of the affine's 3×3 block).
pub fn voxel_size(affine: &[[f64; 4]; 4]) -> [f64; 3] {
    let col = |c: usize| (affine[0][c].powi(2) + affine[1][c].powi(2) + affine[2][c].powi(2)).sqrt();
    [col(0), col(1), col(2)]
}

// ------------------------------------------------- binaries' input check

/// Input-quality summary the fitting binaries print before they start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputQc {
    pub ndc: Option<f64>,
    pub dwi_contrast: Option<f64>,
}

impl InputQc {
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
        let contrast = self
            .dwi_contrast
            .map_or("n/a".to_string(), |v| format!("{v:.3} ({})", contrast_grade(v)));
        format!("NDC {ndc}, DWI contrast {contrast}")
    }
}

/// NDC and contrast ratio of a loaded series inside its mask.
pub fn assess_input(dwi: &DwiData) -> InputQc {
    let b0 = dwi.gtab.b0_threshold;
    let (data, mask) = (dwi.data.view(), Some(dwi.mask.view()));
    InputQc {
        ndc: neighboring_dwi_correlation(data, &dwi.gtab, mask, b0).ok().flatten(),
        dwi_contrast: dwi_contrast_ratio(data, &dwi.gtab, mask, b0).ok().flatten(),
    }
}

/// Run [`assess_input`] and report on stderr as `[tool] input QC: ...`. The
/// summary is suppressed by `quiet`; low-quality warnings are not.
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
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use ndarray::{Array3, Array4};

    fn gtab(bvals: &[f64], bvecs: &[[f64; 3]]) -> GradientTable {
        GradientTable::new(bvals.to_vec(), bvecs.to_vec(), Some(0.03), Some(0.01), None).unwrap()
    }

    /// dipy's doctest for `find_qspace_neighbors`, plus our repeat rule:
    /// volume 3 repeats volume 1's direction at a different b, so it is a
    /// legitimate neighbour, unlike an exact repeat.
    #[test]
    fn neighbors_match_dipy_doctest() {
        let g = gtab(
            &[0.0, 1000.0, 1000.0, 2000.0],
            &[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.99, 0.0001, 0.0001], [1.0, 0.0, 0.0]],
        );
        assert_eq!(find_qspace_neighbors(&g, 50.0), vec![(1, 2), (2, 1), (3, 1)]);
    }

    #[test]
    fn neighbors_are_antipodally_symmetric() {
        let g = gtab(
            &[0.0, 1000.0, 1000.0, 1000.0],
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.99, 0.1, 0.0]],
        );
        assert_eq!(find_qspace_neighbors(&g, 50.0)[0], (1, 3));
    }

    /// A repeated acquisition of the same q-space point is a valid neighbour.
    #[test]
    fn repeats_pair_with_each_other() {
        let g = gtab(
            &[1000.0, 1000.0, 1000.0],
            &[[1.0, 0.0, 0.0], [0.9, 0.43589, 0.0], [-1.0, 0.0, 0.0]],
        );
        let n = find_qspace_neighbors(&g, 50.0);
        assert_eq!(n[0], (0, 2));
        assert_eq!(n[2], (2, 0));
    }

    /// NDC is the same whatever order the volumes are stored in.
    #[test]
    fn ndc_is_invariant_to_volume_order() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let n = 20;
        let mut bvals = vec![0.0];
        let mut bvecs = vec![[0.0, 0.0, 0.0]];
        for _ in 0..n {
            let v: [f64; 3] = [rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5];
            let l = norm(&v);
            bvals.push(1000.0);
            bvecs.push([v[0] / l, v[1] / l, v[2] / l]);
        }
        let data = Array4::<f32>::from_shape_fn((6, 6, 4, n + 1), |_| rng.gen());
        let base = neighboring_dwi_correlation(data.view(), &gtab(&bvals, &bvecs), None, 50.0).unwrap().unwrap();
        let mut order: Vec<usize> = (0..=n).collect();
        order.reverse();
        let (b2, v2): (Vec<f64>, Vec<[f64; 3]>) = order.iter().map(|&i| (bvals[i], bvecs[i])).unzip();
        let d2 = Array4::from_shape_fn(data.dim(), |(x, y, z, t)| data[(x, y, z, order[t])]);
        let rev = neighboring_dwi_correlation(d2.view(), &gtab(&b2, &v2), None, 50.0).unwrap().unwrap();
        assert!((base - rev).abs() < 1e-12, "{base} vs {rev}");
    }

    #[test]
    fn contrast_picks_the_perpendicular_direction() {
        let s = 0.5_f64.sqrt();
        let g = gtab(
            &[0.0, 1000.0, 1000.0, 1000.0, 2000.0],
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [s, s, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        );
        let c = find_qspace_contrast(&g, 50.0);
        assert_eq!(c[0], (1, 3));
        assert_eq!(c[2], (3, 1));
    }

    #[test]
    fn contrast_is_empty_when_every_direction_is_parallel() {
        let g = gtab(&[0.0, 1000.0, 2000.0], &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        assert!(find_qspace_contrast(&g, 50.0).is_empty());
        let d = Array4::<f32>::from_elem((2, 2, 2, 3), 1.0);
        assert!(dwi_contrast_ratio(d.view(), &g, None, 50.0).unwrap().is_none());
    }

    #[test]
    fn shape_mismatches_are_errors() {
        let g = gtab(&[0.0, 1000.0], &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let d = Array4::<f32>::zeros((2, 2, 2, 3));
        assert!(neighboring_dwi_correlation(d.view(), &g, None, 50.0).is_err());
    }

    /// Synthetic fibre-like series with a shared anatomical baseline.
    fn synthetic(n: usize, seed: u64) -> (GradientTable, Array4<f32>) {
        let mut bvals = vec![0.0];
        let mut bvecs = vec![[0.0, 0.0, 0.0]];
        for i in 0..n {
            let t = std::f64::consts::PI * i as f64 / n as f64;
            bvals.push(1000.0);
            bvecs.push([t.cos(), t.sin(), 0.0]);
        }
        let (nx, ny, nz) = (12, 12, 8);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut data = Array4::<f32>::zeros((nx, ny, nz, n + 1));
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let angle = rng.gen::<f64>() * std::f64::consts::PI;
                    let base = 0.5 + rng.gen::<f64>();
                    data[(x, y, z, 0)] = base as f32;
                    for v in 1..=n {
                        let c = bvecs[v][0] * angle.cos() + bvecs[v][1] * angle.sin();
                        let noise = 0.01 * (rng.gen::<f64>() - 0.5);
                        data[(x, y, z, v)] = (base * (0.6 + 0.4 * (-3.0 * c * c).exp()) + noise) as f32;
                    }
                }
            }
        }
        (gtab(&bvals, &bvecs), data)
    }

    #[test]
    fn structured_series_scores_well() {
        let (g, d) = synthetic(30, 7);
        let r = assess(d.view(), &g, None, &QcOptions::default()).unwrap();
        assert!(r.ndc.unwrap() > 0.9, "{:?}", r.ndc);
        assert!(r.dwi_contrast_ratio.unwrap() > DWI_CONTRAST_GOOD_THRESHOLD, "{:?}", r.dwi_contrast_ratio);
        assert_eq!(r.n_dwi_volumes, 30);
        assert_eq!(r.n_b0_volumes, 1);
        assert!(r.warnings().is_empty(), "{:?}", r.warnings());
    }

    /// Smooth anatomy along the slice axis, so adjacent slices predict each other.
    fn smooth_volume_series(n_vol: usize, seed: u64) -> Array4<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (nx, ny, nz) = (20, 20, 16);
        let phase: Vec<f64> = (0..n_vol).map(|_| rng.gen::<f64>() * 6.0).collect();
        Array4::from_shape_fn((nx, ny, nz, n_vol), |(x, y, z, v)| {
            let (fx, fy, fz) = (x as f64 / 3.0, y as f64 / 4.0, z as f64 / 5.0);
            let anat = 2.0 + (fx + phase[v]).sin() * (fy).cos() + 0.5 * (fz + 0.3 * fx).sin();
            (anat * (1.0 + 0.3 * v as f64 / n_vol as f64) + 0.01 * (rng.gen::<f64>() - 0.5)) as f32
        })
    }

    /// In-plane smoothing equals scipy.ndimage.gaussian_filter(a, 2.0)
    /// (reflect boundaries, radius 4σ) on a 9×7 plane.
    #[test]
    fn smoothing_matches_scipy() {
        let a = [
            [0.6369616873, 0.2697867138, 0.0409735239, 0.0165276355, 0.8132702392, 0.9127555773, 0.6066357758],
            [0.729496561, 0.5436249915, 0.9350724238, 0.8158535541, 0.0027385002, 0.8574042766, 0.0335855753],
            [0.7296554464, 0.1756556206, 0.8631789223, 0.5414612202, 0.2997118905, 0.4226872212, 0.0283196711],
            [0.1242832765, 0.6706244147, 0.6471895116, 0.6153851115, 0.3836775543, 0.9972099358, 0.9808353388],
            [0.6855419845, 0.6504592763, 0.6884467306, 0.388921424, 0.135096505, 0.7214883402, 0.5253543225],
            [0.3102418756, 0.4858353588, 0.8894878343, 0.934043516, 0.3577951967, 0.5715298307, 0.3218693911],
            [0.5943000302, 0.3379112255, 0.3916190005, 0.890274352, 0.2271575935, 0.6231871447, 0.0840153436],
            [0.8326441477, 0.7870983075, 0.239369443, 0.8764842308, 0.0585680348, 0.3361170605, 0.1502794669],
            [0.4503393666, 0.7963242703, 0.230642209, 0.0520213011, 0.4045518398, 0.1985130445, 0.0907530456],
        ];
        let d = Array4::<f32>::from_shape_fn((9, 7, 1, 1), |(x, y, _, _)| a[x][y] as f32);
        let sm = smooth_in_plane(&d.view(), 0, 2, &gaussian_kernel(2.0));
        for ((x, y), want) in [((0, 0), 0.5185786786), ((4, 3), 0.5291140102), ((8, 6), 0.2904624581)] {
            assert!((sm[(x, y, 0)] as f64 - want).abs() < 1e-6, "({x},{y}): {} vs {want}", sm[(x, y, 0)]);
        }
    }

    /// Injected dropout and scrambled slices are found, and nothing on clean data.
    #[test]
    fn outlier_slices_finds_injected_corruption() {
        let mut d = smooth_volume_series(12, 11);
        let opts = OutlierSliceOptions { min_voxels: 50, ..OutlierSliceOptions::default() };
        let clean = outlier_slices(d.view(), None, &opts).unwrap();
        assert_eq!(clean.count(), 0, "false positives on clean data: {:?}", clean.flagged);

        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for x in 0..20 {
            for y in 0..20 {
                d[(x, y, 6, 4)] *= 0.3; // dropout
                d[(x, y, 9, 7)] = 1.0 + 2.0 * rng.gen::<f32>(); // scrambled
            }
        }
        let bad = outlier_slices(d.view(), None, &opts).unwrap();
        assert!(bad.flags[(4, 6)], "dropout slice missed");
        assert!(bad.flags[(7, 9)], "scrambled slice missed");
        // A corrupted slice also spoils the prediction of its two adjacent
        // slices, which may be flagged too; nothing else may be.
        for &(v, k) in &bad.flagged {
            assert!(
                (v == 4 && (5..=7).contains(&k)) || (v == 7 && (8..=10).contains(&k)),
                "unexpected flag at volume {v}, slice {k}"
            );
        }
    }

    #[test]
    fn row_matches_column_dictionary() {
        let (g, d) = synthetic(10, 1);
        let r = assess(d.view(), &g, None, &QcOptions::default()).unwrap();
        let names: Vec<&str> = r.row(None, None).into_iter().map(|(n, _)| n).collect();
        let cols: Vec<&str> = qc_columns().into_iter().map(|c| c.name).collect();
        assert_eq!(names, cols);
    }

    /// A smooth field is fully coherent; random directions are not.
    #[test]
    fn coherence_separates_smooth_from_random_fields() {
        let (nx, ny, nz) = (16, 16, 8);
        let affine = [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let fa = Array3::<f32>::from_elem((nx, ny, nz), 0.6);
        let mask = Array3::<bool>::from_elem((nx, ny, nz), true);
        let smooth = Array4::<f32>::from_shape_fn((nx, ny, nz, 3), |(_, _, _, c)| [1.0, 0.0, 0.0][c]);
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let random = Array4::<f32>::from_shape_fn((nx, ny, nz, 3), |_| rng.gen::<f32>() - 0.5);
        let opts = CoherenceOptions::default();
        let a = fixel_coherence(smooth.view(), fa.view(), mask.view(), affine, true, &opts).unwrap();
        let b = fixel_coherence(random.view(), fa.view(), mask.view(), affine, true, &opts).unwrap();
        assert!(a.coherence.unwrap() > 0.95, "smooth {:?}", a.coherence);
        assert!(b.coherence.unwrap() < 0.5, "random {:?}", b.coherence);
    }
}
