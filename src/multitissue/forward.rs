// SPDX-License-Identifier: MIT OR Apache-2.0
//! Forward operators for multi-tissue spherical-deconvolution fits.
//!
//! Given tissue responses (zonal SH per shell) and a per-shell list of unit
//! gradient directions, builds the dense `H` matrix that maps a stack of
//! per-tissue SH coefficient vectors to predicted DWI samples. Each shell's
//! contribution is:
//!
//! ```text
//!   H[shell, tissue] = Y_lmax(shell_dirs) · diag(zonal_factors(response[shell], lmax))
//! ```
//!
//! For an isotropic tissue (lmax = 0) the contribution simplifies to a single
//! column of `response[shell, 0]`, repeated for each direction in the shell.

use nalgebra::DMatrix;

use crate::multitissue::response::TissueResponse;
use crate::sh::{ncoeffs_for_lmax, sh2amp_cart, zonal_factors_per_coeff};

/// Spec for one tissue compartment within a multi-tissue forward block.
#[derive(Debug, Clone)]
pub struct TissueSlot<'a> {
    /// Tissue response (zonal SH per shell).
    pub response: &'a TissueResponse,
    /// Maximum SH order for this tissue's FOD. `0` for isotropic compartments.
    pub lmax: usize,
}

impl<'a> TissueSlot<'a> {
    /// Number of SH coefficients this slot contributes (1 for isotropic, more
    /// for anisotropic).
    pub fn n_coeffs(&self) -> usize {
        ncoeffs_for_lmax(self.lmax)
    }
}

/// Per-shell directions plus the row indices they occupy in the assembled H.
#[derive(Debug, Clone)]
pub struct ShellPlan {
    /// b-value of this shell (0.0 for the b=0 shell; informational only —
    /// not consumed by H assembly itself).
    pub b: f64,
    /// Unit vectors (RAS) of every measurement on this shell.
    pub dirs_ras: Vec<[f32; 3]>,
    /// Augmented-signal row indices these directions occupy in `H`.
    pub aug_indices: Vec<usize>,
}

impl ShellPlan {
    pub fn n_dirs(&self) -> usize {
        self.dirs_ras.len()
    }
}

/// Built forward operator: a dense `H` plus per-tissue column ranges.
#[derive(Debug, Clone)]
pub struct ForwardOperator {
    /// (n_aug × n_coeffs_total) augmented design matrix.
    pub h: DMatrix<f64>,
    /// (start, len) into `h`'s columns for each tissue slot, in slot order.
    pub tissue_columns: Vec<(usize, usize)>,
}

impl ForwardOperator {
    /// Sub-view of `h`'s columns for tissue slot `i`.
    pub fn tissue_design(&self, i: usize) -> nalgebra::DMatrixView<'_, f64> {
        let (c0, w) = self.tissue_columns[i];
        self.h.columns(c0, w)
    }
}

/// Build the `H` matrix mapping `[c_t0, c_t1, …]` → predicted augmented signal.
///
/// `n_aug` is the total number of augmented signal rows; `shells` and
/// `tissues` together describe the per-shell, per-tissue contributions.
/// All `aug_indices` across shells must lie in `0..n_aug` and be disjoint.
pub fn build_h(n_aug: usize, shells: &[ShellPlan], tissues: &[TissueSlot<'_>]) -> ForwardOperator {
    // Column layout: tissues laid out in order, each occupying `n_coeffs` cols.
    let mut tissue_columns = Vec::with_capacity(tissues.len());
    let mut total_cols = 0usize;
    for t in tissues {
        let n = t.n_coeffs();
        tissue_columns.push((total_cols, n));
        total_cols += n;
    }

    let mut h = DMatrix::<f64>::zeros(n_aug, total_cols);
    for (shell_idx, shell) in shells.iter().enumerate() {
        for (slot_idx, slot) in tissues.iter().enumerate() {
            let factors =
                zonal_factors_per_coeff(slot.response.coeffs.get(shell_idx).map_or(&[][..], |v| &v[..]), slot.lmax);
            // Y is (n_dirs × n_sh) in f32; cast on the fly.
            let y = sh2amp_cart(&shell.dirs_ras, slot.lmax);
            let n_sh = factors.len();
            debug_assert_eq!(y.ncols(), n_sh);
            let (c0, _) = tissue_columns[slot_idx];
            for (row_in_shell, &aug_row) in shell.aug_indices.iter().enumerate() {
                for j in 0..n_sh {
                    h[(aug_row, c0 + j)] = y[(row_in_shell, j)] as f64 * factors[j];
                }
            }
        }
    }

    ForwardOperator { h, tissue_columns }
}

/// Per-shell predictor: maps a single tissue's SH coefficients to predicted
/// DWI samples on the augmented signal vector. Cheap to evaluate (just a
/// dense matvec per shell, scattered into the destination).
#[derive(Debug, Clone)]
pub struct PerShellPredictor {
    per_shell: Vec<DMatrix<f64>>, // shell -> (n_dirs × n_sh_tissue)
    aug_indices: Vec<Vec<usize>>, // shell -> aug rows
    n_aug: usize,
    n_sh: usize,
}

impl PerShellPredictor {
    /// Build a predictor for a single tissue across all shells.
    pub fn new(n_aug: usize, shells: &[ShellPlan], tissue: &TissueSlot<'_>) -> Self {
        let n_sh = tissue.n_coeffs();
        let mut per_shell = Vec::with_capacity(shells.len());
        let mut aug_indices = Vec::with_capacity(shells.len());
        for (shell_idx, shell) in shells.iter().enumerate() {
            let factors = zonal_factors_per_coeff(
                tissue
                    .response
                    .coeffs
                    .get(shell_idx)
                    .map_or(&[][..], |v| &v[..]),
                tissue.lmax,
            );
            let y = sh2amp_cart(&shell.dirs_ras, tissue.lmax);
            let mut block = DMatrix::<f64>::zeros(shell.n_dirs(), n_sh);
            for i in 0..shell.n_dirs() {
                for j in 0..n_sh {
                    block[(i, j)] = y[(i, j)] as f64 * factors[j];
                }
            }
            per_shell.push(block);
            aug_indices.push(shell.aug_indices.clone());
        }
        Self {
            per_shell,
            aug_indices,
            n_aug,
            n_sh,
        }
    }

    pub fn n_sh(&self) -> usize {
        self.n_sh
    }

    /// Compute `dest += scale · H_tissue · coefs`, where `H_tissue` is the
    /// per-shell forward block scattered into the augmented signal layout.
    ///
    /// `coefs` length must equal `self.n_sh()`; `dest` length must equal
    /// `n_aug`. Used for both forward prediction (`scale=+1`) and residual
    /// subtraction (`scale=−1`).
    pub fn add_scaled(&self, coefs: &[f64], scale: f64, dest: &mut nalgebra::DVector<f64>) {
        assert_eq!(coefs.len(), self.n_sh);
        assert_eq!(dest.len(), self.n_aug);
        let coef_vec = nalgebra::DVector::<f64>::from_column_slice(coefs);
        for (block, indices) in self.per_shell.iter().zip(self.aug_indices.iter()) {
            let pred = block * &coef_vec;
            for (i, &row) in indices.iter().enumerate() {
                dest[row] += scale * pred[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multitissue::response::TissueResponse;
    use approx::assert_abs_diff_eq;

    fn iso_response(rows: Vec<f64>) -> TissueResponse {
        TissueResponse {
            coeffs: rows.into_iter().map(|r| vec![r]).collect(),
            lmax: 0,
        }
    }

    #[test]
    fn isotropic_two_shell_h_has_constant_columns_per_shell() {
        // Two shells: 3 b=0 dirs and 4 b=1000 dirs. One isotropic tissue with
        // r_0 = 2.0 at b=0 and r_0 = 0.5 at b=1000.
        let dirs_b0 = vec![[1.0_f32, 0.0, 0.0]; 3];
        let dirs_b1 = vec![
            [1.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ];
        let n_aug = 7;
        let shells = vec![
            ShellPlan { b: 0.0, dirs_ras: dirs_b0, aug_indices: vec![0, 1, 2] },
            ShellPlan { b: 1000.0, dirs_ras: dirs_b1, aug_indices: vec![3, 4, 5, 6] },
        ];
        let resp = iso_response(vec![2.0, 0.5]);
        let tissues = vec![TissueSlot { response: &resp, lmax: 0 }];
        let op = build_h(n_aug, &shells, &tissues);
        assert_eq!(op.h.shape(), (7, 1));
        // Per-shell column entries should equal r_0[shell]. Tolerance allows
        // for f32→f64 rounding from `sh2amp_cart` (which returns f32).
        for i in 0..3 {
            assert_abs_diff_eq!(op.h[(i, 0)], 2.0, epsilon = 1e-5);
        }
        for i in 3..7 {
            assert_abs_diff_eq!(op.h[(i, 0)], 0.5, epsilon = 1e-5);
        }
    }

    #[test]
    fn predictor_subtracts_isotropic_signal_per_shell() {
        let dirs_b0 = vec![[1.0_f32, 0.0, 0.0]; 2];
        let dirs_b1 = vec![[0.0_f32, 1.0, 0.0]; 3];
        let n_aug = 5;
        let shells = vec![
            ShellPlan { b: 0.0, dirs_ras: dirs_b0, aug_indices: vec![0, 1] },
            ShellPlan { b: 1000.0, dirs_ras: dirs_b1, aug_indices: vec![2, 3, 4] },
        ];
        let resp = iso_response(vec![3.0, 0.7]);
        let tissue = TissueSlot { response: &resp, lmax: 0 };
        let pred = PerShellPredictor::new(n_aug, &shells, &tissue);
        // Coef = 4.0 → predicted = 4.0 · r_0[shell]. Loose epsilon for f32→f64
        // rounding inside `sh2amp_cart`.
        let mut dest = nalgebra::DVector::<f64>::from_element(5, 0.0);
        pred.add_scaled(&[4.0], 1.0, &mut dest);
        for i in 0..2 {
            assert_abs_diff_eq!(dest[i], 4.0 * 3.0, epsilon = 1e-5);
        }
        for i in 2..5 {
            assert_abs_diff_eq!(dest[i], 4.0 * 0.7, epsilon = 1e-5);
        }
        // Subtract again with scale = -2.0 (net = -1.0 · prediction)
        pred.add_scaled(&[4.0], -2.0, &mut dest);
        for i in 0..2 {
            assert_abs_diff_eq!(dest[i], -4.0 * 3.0, epsilon = 1e-5);
        }
    }
}
