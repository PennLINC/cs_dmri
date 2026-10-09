// SPDX-License-Identifier: MIT OR Apache-2.0
//! Analytical SHORE → ODF Tournier-SH projection.
//!
//! `BrainSuiteShoreFit.odf_sh()` (qsirecon/utils/brainsuite_shore.py:324-363)
//! shows that the ODF, expanded in real spherical harmonics, is *linear* in
//! the SHORE coefficients. The (n, ℓ)-only radial factor `C·G·F` collapses
//! into a single scalar per (n, ℓ) pair, and BrainSuite's per-block ordering
//! converts to mrtrix3 (Tournier) ordering by a per-coefficient permutation
//! and a sign flip on the sin (negative-m) block.
//!
//! Result: a single dense matrix of shape `(n_sh × n_shore)` that turns any
//! SHORE coefficient vector into Tournier-ordered ODF SH coefficients.

use nalgebra::DMatrix;

use crate::basis::Basis;
use crate::basis::shore::ShoreBasis;
use crate::math::{factorial, gamma, hyp2f1};

/// Largest even ℓ ≤ `radial_order`. SHORE caps angular order at the radial
/// order because only `ℓ ≤ n` blocks exist.
pub fn default_lmax(radial_order: u32) -> u32 {
    radial_order - (radial_order % 2)
}

/// Number of Tournier SH coefficients up to and including order `lmax`.
/// (mrtrix3: only even orders contribute, but the index formula
/// `ℓ(ℓ+1)/2 + m` packs them into a contiguous (lmax+1)(lmax+2)/2 layout
/// with odd-order coefficients zero — same as mrtrix's on-disk format.)
pub fn n_tournier_sh_coeffs(lmax: u32) -> usize {
    ((lmax + 1) * (lmax + 2) / 2) as usize
}

/// Radial factor `C(n,ℓ) · G(n,ℓ) · F(n,ℓ)` from `odf_sh` in dipy/qsirecon.
fn radial_factor(n: u32, ell: u32, zeta: f64) -> f64 {
    let nf = n as f64;
    let lf = ell as f64;
    let sign = if ((n as i32) - (ell as i32) / 2) % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    let c_nl = sign * (2.0 * factorial(n - ell) / (zeta.powf(1.5) * gamma(nf + 1.5))).sqrt();
    let g_nl = (gamma(lf / 2.0 + 1.5) * gamma(nf + 1.5))
        / (gamma(lf + 1.5) * factorial(n - ell))
        * 0.5_f64.powf(-lf / 2.0 - 1.5);
    let f_nl = hyp2f1(-(nf) + lf, lf / 2.0 + 1.5, lf + 1.5, 2.0);
    c_nl * g_nl * f_nl
}

/// Tournier coefficient index and sign correction for a BrainSuite SH column
/// at position `k` inside an even-ℓ block (`k ∈ 0..2ℓ`).
///
/// Reasoning: BrainSuite's basis function differs from Tournier's by `sign`
/// (only on the sin / negative-m block). To preserve `c_BS · Y_BS = c_T · Y_T`
/// the coefficient must be multiplied by that same `sign`.
fn brainsuite_to_tournier(ell: u32, k: usize) -> (usize, f64) {
    let block_base = (ell * (ell + 1) / 2) as usize;
    let kl = ell as usize;
    if k < kl {
        // Cosine block: physical m = +(ℓ - k), no sign flip.
        let m_phys = (ell as i32) - (k as i32);
        ((block_base as i32 + m_phys) as usize, 1.0)
    } else if k == kl {
        // m = 0.
        (block_base, 1.0)
    } else {
        // Sine block: physical m = -(k - ℓ), sign = (-1)^((k-ℓ)+1).
        let j = (k as i32) - (ell as i32);
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
        ((block_base as i32 - j) as usize, sign)
    }
}

/// Build the analytical SHORE → Tournier-SH projection matrix.
///
/// Shape: `(n_tournier_sh_coeffs(lmax), basis.n_coeffs())`.
/// Apply as `sh_coeffs = M · shore_coeffs`.
pub fn shore_to_tournier_sh_matrix(basis: &ShoreBasis, lmax: u32) -> DMatrix<f64> {
    let n_rows = n_tournier_sh_coeffs(lmax);
    let n_cols = basis.n_coeffs();
    let mut m = DMatrix::<f64>::zeros(n_rows, n_cols);

    let mut col_idx = 0_usize;
    for n in 0..=basis.radial_order {
        let mut ell = 0_u32;
        while ell <= n {
            let block_size = (2 * ell + 1) as usize;
            if ell <= lmax {
                let radial = radial_factor(n, ell, basis.zeta);
                for k in 0..block_size {
                    let (row, sign) = brainsuite_to_tournier(ell, k);
                    m[(row, col_idx + k)] = radial * sign;
                }
            }
            col_idx += block_size;
            ell += 2;
        }
    }
    debug_assert_eq!(col_idx, n_cols);
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::shore::brainsuite_sh_block;
    use crate::math::{assoc_legendre, factorial};
    use approx::assert_abs_diff_eq;
    use nalgebra::DVector;
    use std::f64::consts::PI;

    /// Tournier (mrtrix3) real-SH evaluation up to order `lmax` at (θ, φ).
    /// Indexing matches `ℓ(ℓ+1)/2 + m`, with m ∈ [-ℓ, ℓ].
    /// Conventions per `dipy.reconst.shm.sh_to_sf` (mrtrix descoteaux/tournier):
    ///   m < 0:  Y = sqrt(2) · (2ℓ+1)/(4π) · √((ℓ-|m|)!/(ℓ+|m|)!) · P_ℓ^|m|(cos φ) · sin(|m|θ)
    ///   m = 0:  Y = √((2ℓ+1)/(4π)) · P_ℓ^0(cos φ)
    ///   m > 0:  Y = sqrt(2) · √((2ℓ+1)/(4π) · (ℓ-m)!/(ℓ+m)!) · P_ℓ^m(cos φ) · cos(mθ)
    /// `P_ℓ^m` carries the Condon–Shortley phase (matches scipy.special.lpmv).
    fn tournier_real_sh(lmax: u32, theta: f64, phi: f64) -> Vec<f64> {
        let n = ((lmax + 1) * (lmax + 2) / 2) as usize;
        let mut out = vec![0.0_f64; n];
        let cos_phi = phi.cos();
        let sqrt2 = 2.0_f64.sqrt();
        for ell in (0..=lmax).step_by(2) {
            let two_ell_plus_one = 2.0 * ell as f64 + 1.0;
            for m in (-(ell as i32))..=(ell as i32) {
                let am = m.unsigned_abs();
                let norm = (two_ell_plus_one / (4.0 * PI)
                    * factorial(ell - am)
                    / factorial(ell + am))
                    .sqrt();
                let p = norm * assoc_legendre(ell, am, cos_phi);
                let val = if m < 0 {
                    sqrt2 * p * (am as f64 * theta).sin()
                } else if m == 0 {
                    p
                } else {
                    sqrt2 * p * (am as f64 * theta).cos()
                };
                let idx = (ell * (ell + 1) / 2) as i32 + m;
                out[idx as usize] = val;
            }
        }
        out
    }

    #[test]
    fn default_lmax_is_largest_even_le_radial_order() {
        assert_eq!(default_lmax(0), 0);
        assert_eq!(default_lmax(2), 2);
        assert_eq!(default_lmax(3), 2);
        assert_eq!(default_lmax(4), 4);
        assert_eq!(default_lmax(6), 6);
        assert_eq!(default_lmax(7), 6);
    }

    #[test]
    fn radial_factor_matches_scipy() {
        // Reference values from scipy (zeta=700). Match dipy/qsirecon's
        // odf_sh() per-(n,ℓ) factor exactly.
        let zeta = 700.0;
        let cases = [
            (0u32, 0u32, 2.766_998_550_654e-2),
            (2, 0, 3.788_868_806_943e-2),
            (2, 2, -4.286_615_722_265e-2),
            (4, 0, 4.340_694_526_519e-2),
            (4, 2, -4.391_643_925_880e-2),
            (4, 4, 5.400_628_174_864e-2),
            (6, 0, 4.738_453_459_969e-2),
            (6, 2, -4.722_289_470_498e-2),
            (6, 4, 4.949_860_498_183e-2),
            (6, 6, -6.322_725_024_349e-2),
        ];
        for (n, ell, expected) in cases {
            let got = radial_factor(n, ell, zeta);
            assert_abs_diff_eq!(got, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn projection_matrix_shape() {
        let basis = ShoreBasis::new(6, 700.0);
        let m = shore_to_tournier_sh_matrix(&basis, 6);
        assert_eq!(m.nrows(), 28); // (7*8)/2
        assert_eq!(m.ncols(), 72);
    }

    #[test]
    fn brainsuite_to_tournier_index_examples() {
        // ℓ=2, block has 5 columns (k=0..4).
        // k=0 → cosine m=+2 → Tournier idx = 2*3/2 + 2 = 5
        // k=1 → cosine m=+1 → idx = 4
        // k=2 → m=0       → idx = 3
        // k=3 → sine m=-1, sign=(-1)^2=+1 → idx = 2
        // k=4 → sine m=-2, sign=(-1)^3=-1 → idx = 1
        let cases = [(0, 5, 1.0), (1, 4, 1.0), (2, 3, 1.0), (3, 2, 1.0), (4, 1, -1.0)];
        for (k, expected_idx, expected_sign) in cases {
            let (idx, sign) = brainsuite_to_tournier(2, k);
            assert_eq!(idx, expected_idx);
            assert_eq!(sign, expected_sign);
        }
    }

    #[test]
    fn projection_matches_brainsuite_eval_per_column() {
        // For every SHORE basis column k, the projected Tournier-SH coefficient
        // vector t = M_proj · e_k must satisfy
        //     ⟨t, Y_T(û)⟩ == radial(n,ℓ) · Y_BS[k_within_block](û)
        // for any direction û. Verifies index permutation, sin-block sign flip,
        // and radial-factor magnitude in one shot.
        let basis = ShoreBasis::new(6, 700.0);
        let lmax = 6_u32;
        let m_proj = shore_to_tournier_sh_matrix(&basis, lmax);
        let dirs = [
            (0.3, 0.7),
            (1.4, 0.5),
            (-1.1, 1.2),
            (2.7, 1.5),
            (0.0, 0.0),
        ];
        // Build the column → (n, ℓ, k_in_block) map (BrainSuite block ordering).
        let mut col_idx = 0_usize;
        for n in 0..=basis.radial_order {
            let mut ell = 0_u32;
            while ell <= n {
                let block_size = (2 * ell + 1) as usize;
                if ell <= lmax {
                    let radial = radial_factor(n, ell, basis.zeta);
                    for k in 0..block_size {
                        // t_k = M_proj * e_{col_idx + k}
                        let mut e = DVector::<f64>::zeros(basis.n_coeffs());
                        e[col_idx + k] = 1.0;
                        let t = &m_proj * &e;
                        for &(theta, phi) in &dirs {
                            let y_t = tournier_real_sh(lmax, theta, phi);
                            let bs_block = brainsuite_sh_block(ell, theta, phi);
                            let lhs: f64 = t.iter().zip(y_t.iter()).map(|(a, b)| a * b).sum();
                            let rhs = radial * bs_block[k];
                            assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-12);
                        }
                    }
                }
                col_idx += block_size;
                ell += 2;
            }
        }
    }

    #[test]
    fn ell0_block_is_first_row_only() {
        // For ℓ=0, every (n, ℓ=0, m=0) coefficient maps to Tournier index 0.
        let basis = ShoreBasis::new(6, 700.0);
        let m = shore_to_tournier_sh_matrix(&basis, 6);
        // Find the SHORE columns for ℓ=0.
        let ell0_cols: Vec<usize> = basis
            .indices()
            .iter()
            .enumerate()
            .filter_map(|(i, (_, ell, _))| if *ell == 0 { Some(i) } else { None })
            .collect();
        for &col in &ell0_cols {
            assert!(m[(0, col)].abs() > 0.0, "ℓ=0 column {col} should be nonzero");
            for row in 1..m.nrows() {
                assert_eq!(m[(row, col)], 0.0, "ℓ=0 column {col} should only touch row 0");
            }
        }
    }
}
