// SPDX-License-Identifier: MIT OR Apache-2.0
//! Spherical-harmonic helpers used by the multi-tissue reconstructions.
//!
//! Re-exports the MRtrix-basis primitives from `odx-rs` (used elsewhere in the
//! crate for ODF I/O) and adds the zonal-convolution scaling factors needed to
//! build forward operators from MRtrix-format response functions.

use std::f64::consts::PI;

pub use odx_rs::mrtrix_sh::{coefficient_index, lmax_for_ncoeffs, ncoeffs_for_lmax, sh2amp_cart};

/// Per-coefficient zonal-convolution scaling vector.
///
/// Convolving a spherical-harmonic FOD by a zonal response (whose only
/// non-zero components are the m=0 harmonics, with even-order coefficients
/// `[r₀, r₂, r₄, …]`) maps the SH coefficients of the FOD to the SH
/// coefficients of the predicted signal via per-ℓ scaling:
///
/// ```text
///   c'_{ℓ,m} = sqrt(4π / (2ℓ+1)) · r_ℓ · c_{ℓ,m}
/// ```
///
/// This routine returns one factor per SH coefficient (in MRtrix coefficient
/// order), so building a forward block per shell reduces to columnwise
/// multiplication of the SH-evaluation matrix by the returned vector.
///
/// Zonal coefficients beyond the supplied `zonal` slice (or at odd ℓ) are
/// treated as zero. Coefficients past `lmax` are not included in the output.
pub fn zonal_factors_per_coeff(zonal: &[f64], lmax: usize) -> Vec<f64> {
    let n = ncoeffs_for_lmax(lmax);
    let mut out = vec![0.0_f64; n];
    let mut l = 0usize;
    while l <= lmax {
        let r_l = zonal.get(l / 2).copied().unwrap_or(0.0);
        let factor = (4.0 * PI / (2.0 * l as f64 + 1.0)).sqrt() * r_l;
        let m_min = -(l as isize);
        let m_max = l as isize;
        for m in m_min..=m_max {
            out[coefficient_index(l, m)] = factor;
        }
        l += 2;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn zonal_factors_match_per_l_formula() {
        // Pure ℓ=0 response with r₀ = 1: every ℓ=0 coefficient should get
        // sqrt(4π) ≈ 3.5449; ℓ=2 coefficients should be zero.
        let factors = zonal_factors_per_coeff(&[1.0], 2);
        assert_eq!(factors.len(), 6);
        assert_abs_diff_eq!(factors[0], (4.0 * PI).sqrt(), epsilon = 1e-12);
        for f in &factors[1..] {
            assert_abs_diff_eq!(*f, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn zonal_factors_apply_per_l_scaling() {
        // r₀=2, r₂=3, lmax=2: ℓ=0 gets sqrt(4π)·2; each of the 5 ℓ=2 coeffs
        // gets sqrt(4π/5)·3. r₄ would be ignored (lmax=2).
        let factors = zonal_factors_per_coeff(&[2.0, 3.0, 99.0], 2);
        assert_abs_diff_eq!(factors[0], (4.0 * PI).sqrt() * 2.0, epsilon = 1e-12);
        let l2_factor = (4.0 * PI / 5.0).sqrt() * 3.0;
        for f in &factors[1..6] {
            assert_abs_diff_eq!(*f, l2_factor, epsilon = 1e-12);
        }
    }

    #[test]
    fn missing_zonal_entries_are_zero() {
        // Empty zonal slice → all zeros.
        let factors = zonal_factors_per_coeff(&[], 4);
        assert_eq!(factors.len(), 15);
        for f in &factors {
            assert_eq!(*f, 0.0);
        }
    }
}
