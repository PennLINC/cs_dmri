// SPDX-License-Identifier: MIT OR Apache-2.0
//! 3D-SHORE basis (BrainSuite-compatible) — radial Laguerre × even-ℓ
//! BrainSuite-ordered real spherical harmonics.
//!
//! Mirrors `qsirecon/utils/brainsuite_shore.py:493-528` (basis assembly) and
//! `qsirecon/utils/shm.py:246-320` (BrainSuite SH ordering) so the coefficient
//! layout is byte-for-byte compatible with qsirecon's existing outputs.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};

use crate::math::{assoc_legendre, cart2sphere, factorial, gamma, gen_laguerre};
use crate::qspace::GradientTable;

use super::{Basis, BasisMetadata, RegularizationDiagonals};

/// 3D-SHORE basis with BrainSuite ordering.
#[derive(Debug, Clone)]
pub struct ShoreBasis {
    /// Maximum radial order (qsirecon default: 6).
    pub radial_order: u32,
    /// Scale parameter ζ (qsirecon default: 700).
    pub zeta: f64,
    indices: Vec<(u32, u32, i32)>,
}

impl ShoreBasis {
    pub fn new(radial_order: u32, zeta: f64) -> Self {
        let mut indices = Vec::new();
        for n in 0..=radial_order {
            let mut ell = 0_u32;
            loop {
                if ell > n {
                    break;
                }
                for m in -(ell as i32)..=(ell as i32) {
                    indices.push((n, ell, m));
                }
                ell += 2;
            }
        }
        Self { radial_order, zeta, indices }
    }

    /// (n, ℓ, m) tuples in coefficient order. The `m` is the BrainSuite SH
    /// index within the ℓ block (descending real cosine, m=0, then sin terms
    /// with sign), not a standard real-SH m. Useful for tagging only.
    pub fn indices(&self) -> &[(u32, u32, i32)] {
        &self.indices
    }
}

/// κ(ζ, n, ℓ) = sqrt( 2 (n-ℓ)! / (ζ^1.5 · Γ(n + 3/2)) )
#[inline]
pub fn kappa(zeta: f64, n: u32, ell: u32) -> f64 {
    debug_assert!(ell <= n);
    let num = 2.0 * factorial(n - ell);
    let denom = zeta.powf(1.5) * gamma(n as f64 + 1.5);
    (num / denom).sqrt()
}

/// Radial part R_{n,ℓ}(q) of the SHORE basis at the given q magnitude.
pub fn shore_radial(n: u32, ell: u32, q: f64, zeta: f64) -> f64 {
    let q2_over_zeta = q * q / zeta;
    let lag = gen_laguerre(n - ell, ell as f64 + 0.5, q2_over_zeta);
    let envelope = (-q2_over_zeta * 0.5).exp();
    let radial_pow = q2_over_zeta.powf(ell as f64 / 2.0);
    kappa(zeta, n, ell) * lag * envelope * radial_pow
}

/// BrainSuite real-SH evaluation at degree ℓ and direction (θ, φ).
/// Returns 2ℓ+1 values in BrainSuite ordering.
///
/// Mirrors `real_sym_sh_brainsuite` in `qsirecon/utils/shm.py:246-320`.
pub fn brainsuite_sh_block(ell: u32, theta: f64, phi: f64) -> Vec<f64> {
    let n = (2 * ell + 1) as usize;
    if ell == 0 {
        return vec![1.0 / (4.0 * PI).sqrt()];
    }
    let cos_phi = phi.cos();

    // Pell[k] = sqrt((2ℓ+1)/(4π) · (ℓ-k)!/(ℓ+k)!) · P_ℓ^k(cos φ)   for k = 0..ℓ
    // (where P_ℓ^k carries the Condon–Shortley phase, matching scipy.special.lpmv).
    let two_ell_plus_one = 2.0 * ell as f64 + 1.0;
    let mut p: Vec<f64> = (0..=ell)
        .map(|k| {
            let norm = (two_ell_plus_one / (4.0 * PI) * factorial(ell - k) / factorial(ell + k))
                .sqrt();
            norm * assoc_legendre(ell, k, cos_phi)
        })
        .collect();

    // BrainSuite ordering inside the ℓ block:
    //   col 0..ℓ-1: sqrt(2) · Re(Y_ℓ^k) · cos(kθ) for k = ℓ, ℓ-1, ..., 1
    //   col ℓ:     P_ℓ^0
    //   col ℓ+1..2ℓ: sqrt(2) · (-1)^(k+1) · Im(Y_ℓ^k) · sin(kθ) for k = 1..ℓ
    //
    // The Condon–Shortley phase is already in P_ℓ^k.
    let sqrt2 = 2.0_f64.sqrt();
    let mut out = vec![0.0; n];
    for k in 1..=(ell as usize) {
        let col = (ell as usize) - k;
        out[col] = sqrt2 * p[k] * (k as f64 * theta).cos();
    }
    out[ell as usize] = p[0];
    for k in 1..=(ell as usize) {
        let col = (ell as usize) + k;
        let sign = if (k + 1) % 2 == 0 { 1.0 } else { -1.0 };
        out[col] = sqrt2 * sign * p[k] * (k as f64 * theta).sin();
    }
    // Move out of `p` to silence a compiler warning about unused mut.
    let _ = &mut p;
    out
}

impl Basis for ShoreBasis {
    fn n_coeffs(&self) -> usize {
        self.indices.len()
    }

    fn design_matrix(&self, gtab: &GradientTable) -> DMatrix<f64> {
        let n_rows = gtab.n_grads();
        let n_cols = self.n_coeffs();
        let qvals = gtab.qvals();
        let bvecs = &gtab.bvecs;

        // Per-direction (r, θ, φ). For b0 directions r = 0 and θ is conventionally 0.
        let coords: Vec<(f64, f64, f64)> = qvals
            .iter()
            .zip(bvecs.iter())
            .map(|(&q, v)| {
                let (r, mut theta, phi) = cart2sphere(v[0] * q, v[1] * q, v[2] * q);
                if theta.is_nan() {
                    theta = 0.0;
                }
                (r, theta, phi)
            })
            .collect();

        // Cache BrainSuite SH blocks per direction per even ℓ — they are reused
        // across all `n` for a given ℓ.
        let max_ell = (0..=self.radial_order)
            .step_by(2)
            .filter(|&e| e <= self.radial_order)
            .last()
            .unwrap_or(0);
        let n_ell_blocks = (max_ell / 2 + 1) as usize;
        let mut sh_cache: Vec<Vec<Vec<f64>>> = (0..n_ell_blocks)
            .map(|_| Vec::with_capacity(n_rows))
            .collect();
        for &(_, theta, phi) in &coords {
            for (i, ell) in (0..=max_ell).step_by(2).enumerate() {
                sh_cache[i].push(brainsuite_sh_block(ell, theta, phi));
            }
        }

        let mut m = DMatrix::<f64>::zeros(n_rows, n_cols);
        let mut col_idx = 0_usize;
        for n in 0..=self.radial_order {
            let mut ell = 0_u32;
            while ell <= n {
                let block_size = (2 * ell + 1) as usize;
                let ell_block_idx = (ell / 2) as usize;
                for (i, &(r, _, _)) in coords.iter().enumerate() {
                    let radial = shore_radial(n, ell, r, self.zeta);
                    let sh_block = &sh_cache[ell_block_idx][i];
                    for k in 0..block_size {
                        m[(i, col_idx + k)] = radial * sh_block[k];
                    }
                }
                col_idx += block_size;
                ell += 2;
            }
        }
        debug_assert_eq!(col_idx, n_cols);
        m
    }

    fn regularization(&self) -> RegularizationDiagonals {
        // N_shore: diag((n*(n+1))^2)
        // L_shore: diag((ℓ*(ℓ+1))^2)
        let mut n_diag = DVector::<f64>::zeros(self.indices.len());
        let mut l_diag = DVector::<f64>::zeros(self.indices.len());
        for (i, &(n, ell, _)) in self.indices.iter().enumerate() {
            let nn = (n * (n + 1)) as f64;
            let ll = (ell * (ell + 1)) as f64;
            n_diag[i] = nn * nn;
            l_diag[i] = ll * ll;
        }
        RegularizationDiagonals {
            primary: n_diag,
            secondary: Some(l_diag),
        }
    }

    fn metadata(&self) -> BasisMetadata {
        BasisMetadata::Shore {
            radial_order: self.radial_order,
            zeta: self.zeta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn coefficient_count_matches_qsirecon_layout() {
        // qsirecon's `shore_index_matrix(6)` produces 72 entries.
        let basis = ShoreBasis::new(6, 700.0);
        assert_eq!(basis.n_coeffs(), 72);

        let basis2 = ShoreBasis::new(2, 700.0);
        // n=0(ℓ=0:1) + n=1(ℓ=0:1) + n=2(ℓ=0:1, ℓ=2:5) = 8
        assert_eq!(basis2.n_coeffs(), 8);
    }

    #[test]
    fn brainsuite_sh_ell0_is_constant() {
        let v = brainsuite_sh_block(0, 0.3, 0.7);
        assert_eq!(v.len(), 1);
        assert_abs_diff_eq!(v[0], 1.0 / (4.0 * PI).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn brainsuite_sh_ell2_block_size() {
        let v = brainsuite_sh_block(2, 0.4, 1.1);
        assert_eq!(v.len(), 5);
        // m=0 column should be P_2^0 normalized: sqrt(5/(4π)) * 0.5 (3cos²φ - 1)
        let cos_phi = 1.1_f64.cos();
        let expected = (5.0 / (4.0 * PI)).sqrt() * 0.5 * (3.0 * cos_phi * cos_phi - 1.0);
        assert_abs_diff_eq!(v[2], expected, epsilon = 1e-10);
    }

    #[test]
    fn design_matrix_shape_and_q0_pattern() {
        // Build a small problem: 1 b0 + 6 directions on a unit sphere.
        let bvecs: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .into_iter()
        .map(|[x, y, z]: [f64; 3]| {
            let n = (x * x + y * y + z * z).sqrt().max(1e-12);
            [x / n, y / n, z / n]
        })
        .collect();
        let bvals = vec![0.0, 1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0];
        let gt = GradientTable::new(bvals, bvecs, Some(0.043), Some(0.011), None).unwrap();
        let basis = ShoreBasis::new(4, 700.0);
        let m = basis.design_matrix(&gt);
        assert_eq!(m.nrows(), 7);
        assert_eq!(m.ncols(), basis.n_coeffs());

        // Row 0 is a b0 (q=0). For ℓ>0 columns, the radial part vanishes
        // because (q²/ζ)^(ℓ/2) is 0. Only ℓ=0 columns can be non-zero.
        for (col, &(_, ell, _)) in basis.indices().iter().enumerate() {
            if ell > 0 {
                assert_abs_diff_eq!(m[(0, col)], 0.0, epsilon = 1e-12);
            }
        }
        // The very first column (n=0, ℓ=0) at q=0 should equal κ(ζ,0,0) · L_0^{1/2}(0) · 1/sqrt(4π).
        let expected = kappa(700.0, 0, 0) * 1.0 * 1.0 / (4.0 * PI).sqrt();
        assert_abs_diff_eq!(m[(0, 0)], expected, epsilon = 1e-10);
    }
}
