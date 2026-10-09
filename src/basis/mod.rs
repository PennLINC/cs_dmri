// SPDX-License-Identifier: MIT OR Apache-2.0
//! Basis-set abstraction: design-matrix construction and regularization shapes.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::qspace::GradientTable;

pub mod shore;

/// Diagonal Tikhonov regularization terms for a basis.
///
/// SHORE has two diagonals (radial `n` and angular `l`); for bases with a
/// single Laplacian operator only `primary` is filled.
#[derive(Debug, Clone)]
pub struct RegularizationDiagonals {
    pub primary: DVector<f64>,
    pub secondary: Option<DVector<f64>>,
}

/// Serialized parameters describing a basis (written to the JSON sidecar
/// alongside coefficients so synthesis and downstream code can reproduce it).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "name")]
pub enum BasisMetadata {
    #[serde(rename = "shore")]
    Shore {
        radial_order: u32,
        zeta: f64,
    },
}

pub trait Basis: Send + Sync {
    fn n_coeffs(&self) -> usize;

    /// Build the (n_grads × n_coeffs) design matrix M for a gradient table.
    /// Both `M` and `gtab` may be reused across many voxels, so this method
    /// is expected to be called once per fit, not per voxel.
    fn design_matrix(&self, gtab: &GradientTable) -> DMatrix<f64>;

    /// Diagonal Tikhonov terms (radial / angular for SHORE; single Laplacian
    /// for MAPMRI variants).
    fn regularization(&self) -> RegularizationDiagonals;

    /// Serialized parameters for the JSON sidecar.
    fn metadata(&self) -> BasisMetadata;
}
