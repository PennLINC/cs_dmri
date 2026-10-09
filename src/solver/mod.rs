// SPDX-License-Identifier: MIT OR Apache-2.0
//! Solver abstraction.
//!
//! `Problem` carries the per-voxel fit data; concrete solvers own their own
//! state (Cholesky factorizations, Lipschitz estimates, warm starts) and
//! implement the `Solver` trait. The voxel loop in `fit.rs` only sees the
//! trait, so swapping FISTA for ADMM / linfa / argmin doesn't churn the
//! public API.

use nalgebra::{DMatrix, DVector};

pub mod alpha;
pub mod csd;
pub mod fista;
pub mod icls;
pub mod shore_icls;
pub mod tikhonov;

pub use alpha::{
    AlphaConfigurable, AlphaPath, AlphaStrategy, VoxelAlphaResult, alpha_max, bic, bic_argmin,
    l2_anchored_argmax, log_path,
};
pub use csd::{CsdConfig, CsdDiagnostics, CsdSolver};
pub use icls::{IclsConfig, IclsDiagnostics, IclsSolver, IclsWorkspace};
pub use shore_icls::ShoreIclsSolver;

/// Per-voxel fit problem data shared with whatever solver is in use.
pub struct Problem<'a> {
    /// Design matrix, shape (n_grads × n_coeffs).
    pub design: &'a DMatrix<f64>,
    /// Measured signal, length n_grads.
    pub signal: &'a DVector<f64>,
}

/// Diagnostics returned alongside coefficients from a single voxel fit.
#[derive(Debug, Clone, Copy, Default)]
pub struct FitDiagnostics {
    pub iterations: u32,
    pub residual_l2: f64,
    pub converged: bool,
    /// Solver-reported regularization weight (alpha for L1, 0 for L2).
    pub alpha: f64,
    /// 1 = L1, 2 = L2 (matches qsirecon's `regularization_image`).
    pub regularization_kind: u8,
}

/// A regularized linear-model solver. Implementations are expected to be
/// `Send + Sync` so the voxel loop can `par_iter` over them.
pub trait Solver: Send + Sync {
    fn fit(&self, problem: &Problem<'_>) -> (DVector<f64>, FitDiagnostics);
}
