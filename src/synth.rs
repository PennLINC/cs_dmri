// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reverse pass: build a 4D NIfTI signal from a coefficient field + new gtab.

use nalgebra::DMatrix;
use ndarray::Array4;
use rayon::prelude::*;

use crate::basis::Basis;
use crate::qspace::GradientTable;

/// Synthesize a new 4D DWI from a coefficient field and a target gradient table.
///
/// `coefficients` is shape (X, Y, Z, n_coeffs) (matches what `fit_volume`
/// produces). `basis` must match the basis those coefficients were fit with.
pub fn synthesize_volume<B: Basis + ?Sized>(
    coefficients: &Array4<f32>,
    basis: &B,
    target_gtab: &GradientTable,
) -> Array4<f32> {
    synthesize_volume_reporting(coefficients, basis, target_gtab, || ())
}

/// Variant of [`synthesize_volume`] that calls `on_plane` once per completed
/// x-plane, for progress reporting. The callback runs on rayon worker
/// threads, so it must be `Sync`.
pub fn synthesize_volume_reporting<B, F>(
    coefficients: &Array4<f32>,
    basis: &B,
    target_gtab: &GradientTable,
    on_plane: F,
) -> Array4<f32>
where
    B: Basis + ?Sized,
    F: Fn() + Sync,
{
    let s = coefficients.shape();
    let (nx, ny, nz, n_coeffs) = (s[0], s[1], s[2], s[3]);
    assert_eq!(n_coeffs, basis.n_coeffs(), "coefficient dim != basis size");
    let n_grads = target_gtab.n_grads();

    let design: DMatrix<f64> = basis.design_matrix(target_gtab);

    let mut out = Array4::<f32>::zeros((nx, ny, nz, n_grads));

    // Parallelize across xy planes (cheap to slice; rayon handles the rest).
    let mut planes: Vec<(usize, ndarray::Array3<f32>)> = (0..nx)
        .into_par_iter()
        .map(|x| {
            let mut plane = ndarray::Array3::<f32>::zeros((ny, nz, n_grads));
            for y in 0..ny {
                for z in 0..nz {
                    // Skip empty voxels: their coefficients are exactly zero.
                    let any_nonzero = (0..n_coeffs).any(|k| coefficients[(x, y, z, k)] != 0.0);
                    if !any_nonzero {
                        continue;
                    }
                    let c = nalgebra::DVector::<f64>::from_iterator(
                        n_coeffs,
                        (0..n_coeffs).map(|k| coefficients[(x, y, z, k)] as f64),
                    );
                    let s_pred = &design * c;
                    for t in 0..n_grads {
                        plane[(y, z, t)] = s_pred[t] as f32;
                    }
                }
            }
            on_plane();
            (x, plane)
        })
        .collect();
    planes.sort_by_key(|(x, _)| *x);

    for (x, plane) in planes {
        for y in 0..ny {
            for z in 0..nz {
                for t in 0..n_grads {
                    out[(x, y, z, t)] = plane[(y, z, t)];
                }
            }
        }
    }

    out
}
