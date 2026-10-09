// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-voxel effective gradient tables from a gradient-deviation ("graddev")
//! field, for fitting with the gradients the scanner actually applied.
//!
//! The field (HCP/FSL `grad_dev` layout, or qsiprep/TORTOISE's
//! `*_space-ACPC_graddev.nii.gz`) holds, per voxel, a row-major 3×3 `T` with
//! `g_eff = Tᵀ · g`, components in the field image's voxel axes — the frame of
//! [`DwiData::bvecs_vox`]. Loading and the `T` vs `T − I` detection are
//! odx-rs's ([`odx_rs::GradDevField`]); this module only requires the field to
//! share the DWI's grid and turns it into effective tables:
//!
//! ```text
//!   q = Tᵀ · ĝ_vox      b_eff = b · |q|²      dir = normalize(R · q)
//! ```
//!
//! with `R` = [`DwiData::frame_rotation`], so `dir` lands in the same frame as
//! `gtab.bvecs` (and the fitted SH). Volumes at or below the b=0 threshold keep
//! their nominal entry: a zero gradient is not deviated.
//!
//! Correcting at fit time handles both halves of the deviation — direction and
//! per-direction b — which a post-hoc rotation of a fitted ODF (`odx graddev`)
//! cannot: the stretch part of `T` makes b direction-dependent and changes the
//! apparent ODF shape, not just its orientation.

use std::path::Path;

use nalgebra::{Matrix3, Vector3};
use ndarray::Array3;
use odx_rs::{GradDevField, IdentityPolicy};

use crate::io::dwi::DwiData;
use crate::{CsDmriError, Result};

/// Max |T − I| entry below which a voxel is treated as undeviated and fit with
/// the nominal table (exactly the no-graddev result).
pub const IDENTITY_TOL: f64 = 1e-6;

/// Affine entries (mm) may differ by this much between the DWI and the field.
const AFFINE_TOL: f64 = 1e-3;

/// A gradient-deviation field checked against a DWI's grid.
#[derive(Debug, Clone)]
pub struct VoxelGradDev {
    field: GradDevField,
}

impl VoxelGradDev {
    /// Load `path` and require it to share `dwi`'s voxel grid (dims and affine).
    pub fn load(path: &Path, policy: IdentityPolicy, dwi: &DwiData) -> Result<Self> {
        let field = GradDevField::load_nifti(path, policy)
            .map_err(|e| CsDmriError::Other(format!("graddev {path:?}: {e}")))?;
        let dwi_affine = odx_rs::reference_affine::read_reference_affine(&dwi.source_nifti)
            .map_err(|e| CsDmriError::Other(format!("read affine from {:?}: {e}", dwi.source_nifti)))?;
        Self::new(field, dwi, &dwi_affine)
    }

    /// Wrap an already-loaded field, checking it against `dwi`'s grid.
    pub fn new(field: GradDevField, dwi: &DwiData, dwi_affine: &[[f64; 4]; 4]) -> Result<Self> {
        let s = dwi.data.shape();
        if field.dims() != [s[0], s[1], s[2]] {
            return Err(CsDmriError::Dimension(format!(
                "graddev grid {:?} does not match the DWI grid {:?}; resample it to the DWI first",
                field.dims(),
                [s[0], s[1], s[2]]
            )));
        }
        let fa = field.affine();
        let worst = (0..3)
            .flat_map(|r| (0..4).map(move |c| (r, c)))
            .map(|(r, c)| (fa[r][c] - dwi_affine[r][c]).abs())
            .fold(0.0_f64, f64::max);
        if worst > AFFINE_TOL {
            return Err(CsDmriError::Dimension(format!(
                "graddev affine differs from the DWI affine by up to {worst:.4} — its components \
                 are in its own voxel frame, so it must be on the DWI's exact grid"
            )));
        }
        Ok(Self { field })
    }

    pub fn identity_added(&self) -> bool {
        self.field.identity_added()
    }

    /// Row-major `T` at voxel `(x, y, z)` (identity included).
    pub fn matrix(&self, x: usize, y: usize, z: usize) -> Matrix3<f64> {
        self.field.matrix_ijk_at(x, y, z)
    }

    /// True when `T` is the identity to within [`IDENTITY_TOL`].
    pub fn is_identity(&self, x: usize, y: usize, z: usize) -> bool {
        (self.matrix(x, y, z) - Matrix3::identity()).abs().max() < IDENTITY_TOL
    }

    /// Effective table at `(x, y, z)`, written into the caller's buffers (in
    /// `dwi`'s gradient order, directions in `dwi.bvec_frame`).
    pub fn effective_table_into(
        &self,
        dwi: &DwiData,
        x: usize,
        y: usize,
        z: usize,
        bvals: &mut Vec<f64>,
        dirs: &mut Vec<[f64; 3]>,
    ) {
        effective_table_into(&self.matrix(x, y, z), dwi, bvals, dirs);
    }

    /// Largest effective b over the masked voxels (at least the nominal bmax),
    /// so a response defined over b can cover every voxel.
    pub fn max_effective_b(&self, dwi: &DwiData, mask: &Array3<bool>) -> f64 {
        let nominal = dwi.gtab.bvals.iter().cloned().fold(0.0_f64, f64::max);
        let mut best = nominal;
        for ((x, y, z), &m) in mask.indexed_iter() {
            if !m {
                continue;
            }
            let tt = self.matrix(x, y, z).transpose();
            for (i, &b) in dwi.gtab.bvals.iter().enumerate() {
                if let Some(g) = unit(dwi.bvecs_vox[i]) {
                    best = best.max(b * (tt * g).norm_squared());
                }
            }
        }
        best
    }

    /// Quantile `q` (0–1) of `|b_eff − b|` over the masked voxels' diffusion-
    /// weighted volumes: how far the field moves b off the acquired values. On
    /// a Prisma-class whole-body coil the 95th percentile is a few tens of
    /// s/mm²; on a strong-gradient system, hundreds. Large voxel counts are
    /// subsampled (deterministically) to keep this cheap.
    pub fn b_shift_quantile(&self, dwi: &DwiData, mask: &Array3<bool>, q: f64) -> f64 {
        let b0_thr = dwi.gtab.b0_threshold;
        let voxels: Vec<(usize, usize, usize)> =
            mask.indexed_iter().filter(|(_, &m)| m).map(|(ix, _)| ix).collect();
        let stride = (voxels.len() / 20_000).max(1);
        let mut shifts = Vec::new();
        for &(x, y, z) in voxels.iter().step_by(stride) {
            let tt = self.matrix(x, y, z).transpose();
            for (i, &b) in dwi.gtab.bvals.iter().enumerate() {
                if b <= b0_thr {
                    continue;
                }
                if let Some(g) = unit(dwi.bvecs_vox[i]) {
                    shifts.push((b * (tt * g).norm_squared() - b).abs());
                }
            }
        }
        if shifts.is_empty() {
            return 0.0;
        }
        let k = ((shifts.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
        shifts.select_nth_unstable_by(k, f64::total_cmp);
        shifts[k]
    }
}

/// Effective table for one deviation matrix `t` (row-major, `g_eff = tᵀ g`).
pub fn effective_table_into(
    t: &Matrix3<f64>,
    dwi: &DwiData,
    bvals: &mut Vec<f64>,
    dirs: &mut Vec<[f64; 3]>,
) {
    let tt = t.transpose();
    let r = Matrix3::from_fn(|i, j| dwi.frame_rotation[i][j]);
    let b0_thr = dwi.gtab.b0_threshold;
    bvals.clear();
    dirs.clear();
    for (i, &b) in dwi.gtab.bvals.iter().enumerate() {
        match unit(dwi.bvecs_vox[i]).filter(|_| b > b0_thr) {
            Some(g) => {
                let q = tt * g;
                let d = (r * q).normalize();
                bvals.push(b * q.norm_squared());
                dirs.push([d[0], d[1], d[2]]);
            }
            None => {
                bvals.push(b);
                dirs.push(dwi.gtab.bvecs[i]);
            }
        }
    }
}

fn unit(v: [f64; 3]) -> Option<Vector3<f64>> {
    let v = Vector3::new(v[0], v[1], v[2]);
    let n = v.norm();
    (n > 1e-12).then(|| v / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qspace::{BvecFrame, GradientTable};
    use ndarray::Array4;

    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn dwi_with(bvals: Vec<f64>, bvecs_vox: Vec<[f64; 3]>, rot: [[f64; 3]; 3]) -> DwiData {
        let mut bvecs = bvecs_vox.clone();
        crate::qspace::rotate_bvecs(&mut bvecs, &rot);
        let n = bvals.len();
        let gtab = GradientTable::new(bvals, bvecs, Some(0.04), Some(0.01), None).unwrap();
        let mut d = DwiData::from_table(
            Array4::zeros((1, 1, 1, n)),
            Array3::from_elem((1, 1, 1), true),
            gtab,
            BvecFrame::WorldRas,
            "synthetic".into(),
        );
        d.bvecs_vox = bvecs_vox;
        d.frame_rotation = rot;
        d
    }

    #[test]
    fn identity_reproduces_nominal_table() {
        let lps = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        let dwi = dwi_with(vec![0.0, 1000.0, 2000.0], vec![[0.0; 3], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], lps);
        let (mut b, mut d) = (Vec::new(), Vec::new());
        effective_table_into(&Matrix3::identity(), &dwi, &mut b, &mut d);
        assert_eq!(b, dwi.gtab.bvals);
        for (got, want) in d.iter().zip(&dwi.gtab.bvecs) {
            for k in 0..3 {
                assert!((got[k] - want[k]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn deviation_acts_in_the_voxel_frame_then_rotates() {
        // Stretch voxel axis x by 1.1 and rotate into an LPS world frame: the
        // b-value scales by 1.21 along x, and the direction comes out as R·x̂.
        let lps = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        let dwi = dwi_with(vec![0.0, 1000.0, 1000.0], vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], lps);
        let t = Matrix3::new(1.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let (mut b, mut d) = (Vec::new(), Vec::new());
        effective_table_into(&t, &dwi, &mut b, &mut d);
        assert_eq!(b[0], 0.0);
        assert!((b[1] - 1210.0).abs() < 1e-9 && (b[2] - 1000.0).abs() < 1e-9);
        assert!((d[1][0] + 1.0).abs() < 1e-12 && (d[2][1] + 1.0).abs() < 1e-12);
        // A shear row of T moves y-gradients toward x: g_eff = Tᵀ g picks up T[1][0].
        let t = Matrix3::new(1.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0);
        effective_table_into(&t, &dwi, &mut b, &mut d);
        let q: Vector3<f64> = Vector3::new(0.1, 1.0, 0.0);
        assert!((b[2] - 1000.0 * q.norm_squared()).abs() < 1e-9);
        let want = Matrix3::from_fn(|i, j| lps[i][j]) * q.normalize();
        for k in 0..3 {
            assert!((d[2][k] - want[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn b_shift_quantile_measures_how_far_the_field_moves_b() {
        let dwi = dwi_with(vec![0.0, 1000.0, 1000.0], vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], IDENTITY);
        let field = |t: Matrix3<f64>| {
            let data: Vec<f32> = (0..9).map(|k| t[(k / 3, k % 3)] as f32).collect();
            let affine = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
            let f = GradDevField::from_parts([1, 1, 1], affine, data, IdentityPolicy::Included).unwrap();
            VoxelGradDev { field: f }
        };
        assert_eq!(field(Matrix3::identity()).b_shift_quantile(&dwi, &dwi.mask, 0.95), 0.0);
        // x stretched by 1.1: the x volume moves 1000 → 1210, the y volume not at all.
        let t = Matrix3::new(1.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let gd = field(t);
        assert!((gd.b_shift_quantile(&dwi, &dwi.mask, 1.0) - 210.0).abs() < 1e-3); // f32 field
        assert!(gd.b_shift_quantile(&dwi, &dwi.mask, 0.0).abs() < 1e-9);
    }
}
