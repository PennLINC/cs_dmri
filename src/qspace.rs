// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gradient-table abstraction: bvals, bvecs, big/small delta, derived q and tau.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::{CsDmriError, Result};

/// Default b0 detection threshold (s/mm^2).
pub const DEFAULT_B0_THRESHOLD: f64 = 50.0;

/// TORTOISE assumed maximum gradient amplitude (T/m).
/// Mirrors `estimate_mapmri_main.cxx` — hardcoded as `40 mT/m * 2`.
pub const TORTOISE_DEFAULT_GMAX: f64 = 80e-3;

/// Gyromagnetic ratio for protons (rad / (s · T)).
pub const GYROMAGNETIC_RATIO: f64 = 267.51532e6;

/// How `(big_delta, small_delta)` were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaSource {
    /// Both supplied by caller.
    UserSupplied,
    /// Estimated via TORTOISE's max-bval heuristic.
    EstimatedTortoise,
}

/// Coordinate frame the bvecs (and therefore the fitted SH coefficients) live in.
///
/// FSL ships bvecs in `ImageAxis` (the columns of the NIfTI affine define the
/// world-RAS axis directions but FSL bvecs ignore that and live in the raw voxel
/// axis frame). Most ODF viewers — including TRXViz — assume `WorldRas`, so by
/// default cs-fit rotates bvecs into world-RAS using the column-normalized 3×3
/// of the NIfTI affine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BvecFrame {
    /// Image-axis frame, i.e. raw FSL convention. Matches dipy/qsirecon.
    ImageAxis,
    /// World-RAS frame, after applying the NIfTI affine's rotation.
    WorldRas,
}

/// Column-normalized 3×3 rotation extracted from a NIfTI affine. Strips voxel
/// sizes so the result is a pure rotation/reflection mapping image-axis unit
/// vectors to world-RAS unit vectors.
pub fn affine_rotation(affine: &[[f64; 4]; 4]) -> [[f64; 3]; 3] {
    let mut r = [[0.0_f64; 3]; 3];
    for j in 0..3 {
        let col = [affine[0][j], affine[1][j], affine[2][j]];
        let norm = (col[0] * col[0] + col[1] * col[1] + col[2] * col[2]).sqrt();
        let s = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for i in 0..3 {
            r[i][j] = col[i] * s;
        }
    }
    r
}

/// Apply a 3×3 rotation in-place to each bvec.
pub fn rotate_bvecs(bvecs: &mut [[f64; 3]], rotation: &[[f64; 3]; 3]) {
    for b in bvecs.iter_mut() {
        let v = *b;
        b[0] = rotation[0][0] * v[0] + rotation[0][1] * v[1] + rotation[0][2] * v[2];
        b[1] = rotation[1][0] * v[0] + rotation[1][1] * v[1] + rotation[1][2] * v[2];
        b[2] = rotation[2][0] * v[0] + rotation[2][1] * v[1] + rotation[2][2] * v[2];
    }
}

/// Diffusion gradient table.
#[derive(Debug, Clone)]
pub struct GradientTable {
    pub bvals: Vec<f64>,
    pub bvecs: Vec<[f64; 3]>,
    /// big delta (Δ) in seconds.
    pub big_delta: f64,
    /// small delta (δ) in seconds.
    pub small_delta: f64,
    pub delta_source: DeltaSource,
    pub b0_threshold: f64,
}

impl GradientTable {
    /// Build a gradient table from raw values, estimating deltas if either is missing.
    ///
    /// `gmax` (T/m) is only consulted when at least one of `big_delta` / `small_delta`
    /// is `None`. Pass `Some(TORTOISE_DEFAULT_GMAX)` to match TORTOISE.
    pub fn new(
        bvals: Vec<f64>,
        bvecs: Vec<[f64; 3]>,
        big_delta: Option<f64>,
        small_delta: Option<f64>,
        gmax: Option<f64>,
    ) -> Result<Self> {
        if bvals.len() != bvecs.len() {
            return Err(CsDmriError::Dimension(format!(
                "bvals ({}) and bvecs ({}) must have the same length",
                bvals.len(),
                bvecs.len()
            )));
        }
        let (big_delta, small_delta, source) = match (big_delta, small_delta) {
            (Some(big), Some(small)) => (big, small, DeltaSource::UserSupplied),
            _ => {
                let gmax = gmax.unwrap_or(TORTOISE_DEFAULT_GMAX);
                let (big_est, small_est) = estimate_deltas_tortoise(&bvals, gmax);
                (big_est, small_est, DeltaSource::EstimatedTortoise)
            }
        };
        Ok(Self {
            bvals,
            bvecs,
            big_delta,
            small_delta,
            delta_source: source,
            b0_threshold: DEFAULT_B0_THRESHOLD,
        })
    }

    /// Number of gradient measurements (rows of the design matrix).
    #[inline]
    pub fn n_grads(&self) -> usize {
        self.bvals.len()
    }

    /// Effective diffusion time τ = Δ − δ/3, in seconds.
    #[inline]
    pub fn tau(&self) -> f64 {
        self.big_delta - self.small_delta / 3.0
    }

    /// q-magnitude per measurement: q = sqrt(b / (4π² τ)).
    /// b0 directions get q = 0.
    pub fn qvals(&self) -> Vec<f64> {
        let tau = self.tau();
        let denom = 4.0 * PI * PI * tau;
        self.bvals
            .iter()
            .map(|&b| {
                if b <= self.b0_threshold {
                    0.0
                } else {
                    (b / denom).sqrt()
                }
            })
            .collect()
    }

    /// q-vectors: qval * bvec, length 3.
    pub fn qvecs(&self) -> Vec<[f64; 3]> {
        let q = self.qvals();
        self.bvecs
            .iter()
            .zip(q.iter())
            .map(|(v, &q)| [v[0] * q, v[1] * q, v[2] * q])
            .collect()
    }

    /// Boolean mask: true where the measurement is a b0.
    pub fn b0_mask(&self) -> Vec<bool> {
        self.bvals.iter().map(|&b| b <= self.b0_threshold).collect()
    }

    /// Partition gradients into shells. The b=0 shell (bval ≤ `b0_threshold`)
    /// always comes first with `b == 0.0`; subsequent shells are sorted by
    /// ascending mean b-value. Two non-zero gradients land in the same shell if
    /// their bvals differ by less than `tolerance` (default 50 s/mm² — the
    /// same scale `qsirecon`/`dipy` use for shell detection).
    pub fn shells(&self, tolerance: f64) -> Vec<Shell> {
        let mut shells: Vec<Shell> = Vec::new();
        let mut b0 = Shell { b: 0.0, indices: Vec::new() };
        for (idx, &b) in self.bvals.iter().enumerate() {
            if b <= self.b0_threshold {
                b0.indices.push(idx);
                continue;
            }
            let slot = shells
                .iter_mut()
                .find(|s| (s.b - b).abs() <= tolerance);
            if let Some(s) = slot {
                let n = s.indices.len() as f64;
                s.b = (s.b * n + b) / (n + 1.0);
                s.indices.push(idx);
            } else {
                shells.push(Shell { b, indices: vec![idx] });
            }
        }
        shells.sort_by(|a, b| a.b.partial_cmp(&b.b).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = Vec::with_capacity(1 + shells.len());
        if !b0.indices.is_empty() {
            out.push(b0);
        }
        out.extend(shells);
        out
    }
}

/// One shell from a gradient table partition.
#[derive(Debug, Clone, PartialEq)]
pub struct Shell {
    /// Mean b-value of the shell (`0.0` for the b=0 shell).
    pub b: f64,
    /// Indices into the gradient table belonging to this shell.
    pub indices: Vec<usize>,
}

/// TORTOISE's heuristic for estimating (Δ, δ) when not provided by the user.
///
/// Mirrors `tortoisev4/src/tools/EstimateMAPMRI/estimate_mapmri_main.cxx:202-220`.
/// Returns `(big_delta, small_delta)` in seconds.
pub fn estimate_deltas_tortoise(bvals: &[f64], gmax: f64) -> (f64, f64) {
    let max_b = bvals.iter().cloned().fold(0.0_f64, f64::max);
    if max_b <= 0.0 {
        // Degenerate: no diffusion weighting. Return dipy's traditional fallback.
        let small_ms = (1.0_f64).powf(1.0 / 3.0);
        return (3.0 * small_ms * 1e-3, small_ms * 1e-3);
    }
    let gyro = GYROMAGNETIC_RATIO;
    // temp = max_bval / gyro^2 / G^2 / 2 * 1e6  (TORTOISE's formula)
    let temp = max_b / gyro / gyro / gmax / gmax / 2.0 * 1e6;
    // small_delta and big_delta in milliseconds in TORTOISE; we convert to seconds.
    let small_delta_ms = temp.powf(1.0 / 3.0) * 1000.0;
    let big_delta_ms = small_delta_ms * 3.0;
    (big_delta_ms * 1e-3, small_delta_ms * 1e-3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn user_supplied_deltas_pass_through() {
        let gt = GradientTable::new(
            vec![0.0, 1000.0, 1000.0],
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Some(0.05),
            Some(0.012),
            None,
        )
        .unwrap();
        assert_eq!(gt.delta_source, DeltaSource::UserSupplied);
        assert_abs_diff_eq!(gt.big_delta, 0.05, epsilon = 1e-12);
        assert_abs_diff_eq!(gt.small_delta, 0.012, epsilon = 1e-12);
        assert_abs_diff_eq!(gt.tau(), 0.05 - 0.012 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn missing_delta_triggers_tortoise_estimate() {
        // bvals in FSL units (s/mm^2). TORTOISE's formula folds the unit
        // conversion via the trailing 1e6 factor.
        let gt = GradientTable::new(
            vec![0.0, 3000.0],
            vec![[0.0; 3], [1.0, 0.0, 0.0]],
            None,
            None,
            Some(TORTOISE_DEFAULT_GMAX),
        )
        .unwrap();
        assert_eq!(gt.delta_source, DeltaSource::EstimatedTortoise);
        assert_abs_diff_eq!(gt.big_delta / gt.small_delta, 3.0, epsilon = 1e-9);
        // Order-of-magnitude sanity: for b=3000, Gmax=80mT/m, expected δ ≈ 13–15 ms.
        assert!(gt.small_delta > 5e-3 && gt.small_delta < 30e-3);
    }

    #[test]
    fn qvals_are_zero_for_b0() {
        let gt = GradientTable::new(
            vec![0.0, 1000.0],
            vec![[0.0; 3], [1.0, 0.0, 0.0]],
            Some(0.043),
            Some(0.011),
            None,
        )
        .unwrap();
        let q = gt.qvals();
        assert_eq!(q[0], 0.0);
        assert!(q[1] > 0.0);
    }

    #[test]
    fn affine_rotation_strips_voxel_sizes() {
        // LPS image with 1.7 mm isotropic voxels (matches the test fixture).
        let aff = [
            [-1.7, 0.0, 0.0, 79.9],
            [0.0, -1.7, 0.0, 80.75],
            [0.0, 0.0, 1.7, -80.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let r = affine_rotation(&aff);
        for j in 0..3 {
            let n = (r[0][j] * r[0][j] + r[1][j] * r[1][j] + r[2][j] * r[2][j]).sqrt();
            assert_abs_diff_eq!(n, 1.0, epsilon = 1e-12);
        }
        assert_abs_diff_eq!(r[0][0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[1][1], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[2][2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rotate_bvecs_lps_to_ras_flips_x_and_y() {
        let aff = [
            [-1.7, 0.0, 0.0, 0.0],
            [0.0, -1.7, 0.0, 0.0],
            [0.0, 0.0, 1.7, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let r = affine_rotation(&aff);
        let mut bvecs = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        rotate_bvecs(&mut bvecs, &r);
        assert_eq!(bvecs[0], [-1.0, 0.0, 0.0]);
        assert_eq!(bvecs[1], [0.0, -1.0, 0.0]);
        assert_eq!(bvecs[2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn shells_partitions_b0_and_dwi() {
        let gt = GradientTable::new(
            vec![0.0, 1000.0, 1010.0, 990.0, 0.0, 3000.0, 3000.0],
            vec![[0.0; 3]; 7],
            Some(0.05),
            Some(0.012),
            None,
        )
        .unwrap();
        let shells = gt.shells(50.0);
        assert_eq!(shells.len(), 3);
        assert_eq!(shells[0].b, 0.0);
        assert_eq!(shells[0].indices, vec![0, 4]);
        assert!((shells[1].b - 1000.0).abs() < 10.0);
        assert_eq!(shells[1].indices, vec![1, 2, 3]);
        assert!((shells[2].b - 3000.0).abs() < 1e-9);
        assert_eq!(shells[2].indices, vec![5, 6]);
    }

    #[test]
    fn shells_omits_b0_when_absent() {
        let gt = GradientTable::new(
            vec![1000.0, 1000.0, 3000.0],
            vec![[0.0; 3]; 3],
            Some(0.05),
            Some(0.012),
            None,
        )
        .unwrap();
        let shells = gt.shells(50.0);
        assert_eq!(shells.len(), 2);
        assert!((shells[0].b - 1000.0).abs() < 1e-9);
        assert!((shells[1].b - 3000.0).abs() < 1e-9);
    }

    #[test]
    fn tortoise_heuristic_matches_reference_formula() {
        // Reproduce TORTOISE's exact computation in Rust.
        let bvals = vec![0.0_f64, 3000.0];
        let (big, small) = estimate_deltas_tortoise(&bvals, TORTOISE_DEFAULT_GMAX);
        let temp = 3000.0_f64 / GYROMAGNETIC_RATIO / GYROMAGNETIC_RATIO
            / TORTOISE_DEFAULT_GMAX
            / TORTOISE_DEFAULT_GMAX
            / 2.0
            * 1e6;
        let small_expected_ms = temp.powf(1.0 / 3.0) * 1000.0;
        let big_expected_ms = small_expected_ms * 3.0;
        assert_abs_diff_eq!(small, small_expected_ms * 1e-3, epsilon = 1e-15);
        assert_abs_diff_eq!(big, big_expected_ms * 1e-3, epsilon = 1e-15);
    }
}
