// SPDX-License-Identifier: MIT OR Apache-2.0
//! Closed-form propagator-derived scalar maps for BrainSuiteSHORE.
//!
//! Formulas adapted from dipy SHORE (`dipy/reconst/shore.py`) and dipy iso-MAPMRI
//! (`dipy/reconst/mapmri.py`, isotropic branch). The basis equivalence and
//! per-mode conversion factor `α(n,ℓ)` are documented in
//! `scripts/microstructure_math.md`.
//!
//! All scalar functions take the *raw* fitted BrainSuiteSHORE coefficient vector;
//! they internally divide by the predicted `Ê(0)` so the returned values match
//! dipy's normalization convention (propagator integrates to 1).

use std::f64::consts::PI;

use crate::ShoreBasis;
use crate::basis::shore::{brainsuite_sh_block, kappa};
use crate::math::{binomial_float, factorial, gamma, gen_laguerre, hyp2f1};

/// Cache of (n, ℓ, m) tuples and per-(n, ℓ) flat-block offsets, derived once
/// per [`ShoreBasis`] and reused across many voxels.
pub struct ScalarBasisInfo {
    pub radial_order: u32,
    pub zeta: f64,
    pub n_coeffs: usize,
    /// `indices[i] = (n, ℓ, m)` matching the flat coefficient layout.
    pub indices: Vec<(u32, u32, i32)>,
    /// `nl_blocks[k] = (n, ℓ, start, len)` — block offsets for each unique
    /// (n, ℓ) pair (in iteration order). `len = 2ℓ+1`.
    pub nl_blocks: Vec<NlBlock>,
}

#[derive(Debug, Clone, Copy)]
pub struct NlBlock {
    pub n: u32,
    pub ell: u32,
    pub start: usize,
    pub len: usize,
}

impl ScalarBasisInfo {
    pub fn from_basis(basis: &ShoreBasis) -> Self {
        let radial_order = basis.radial_order;
        let zeta = basis.zeta;
        let indices: Vec<(u32, u32, i32)> = basis.indices().to_vec();
        let n_coeffs = indices.len();

        let mut nl_blocks = Vec::new();
        let mut start = 0_usize;
        for n in 0..=radial_order {
            let mut ell = 0_u32;
            while ell <= n {
                let len = (2 * ell + 1) as usize;
                nl_blocks.push(NlBlock { n, ell, start, len });
                start += len;
                ell += 2;
            }
        }
        debug_assert_eq!(start, n_coeffs);

        Self { radial_order, zeta, n_coeffs, indices, nl_blocks }
    }

    /// Flat index of the `(n, 0, 0)` coefficient (used by RTOP, MSD, QIV).
    fn n00_index(&self, n: u32) -> usize {
        // (n, ℓ=0) has block size 1 and is the first block of each n.
        for blk in &self.nl_blocks {
            if blk.n == n && blk.ell == 0 {
                return blk.start;
            }
        }
        unreachable!("(n=0, ℓ=0) block must exist for every n in [0, N]");
    }
}

/// Predicted DWI signal at q=0 — the dipy normalization factor Σ c_i B_i.
///
/// Only ℓ=0 modes contribute at the origin (the (q²/ζ)^(ℓ/2) factor zeroes the
/// rest), and within ℓ=0 only `m=0` is non-trivial.
pub fn predict_e0(coefs: &[f64], info: &ScalarBasisInfo) -> f64 {
    let mut e0 = 0.0;
    let inv_sqrt_4pi = 1.0 / (4.0 * PI).sqrt();
    for n in 0..=info.radial_order {
        let idx = info.n00_index(n);
        let radial_at_zero = kappa(info.zeta, n, 0) * gen_laguerre(n, 0.5, 0.0);
        e0 += coefs[idx] * radial_at_zero * inv_sqrt_4pi;
    }
    e0
}

/// Return-to-Origin Probability — the propagator at the origin.
///
/// Closed form from `dipy/reconst/shore.py:382-403`, summation widened to the
/// full BrainSuite radial range (0..=N). Coefficients are normalized by `Ê(0)`
/// so the result is in dipy's convention (propagator integrates to 1). The
/// returned value is clamped to ≥ 0 — RTOP is physically a probability
/// density, and the SHORE basis truncation can drive raw estimates slightly
/// negative on noisy fits (matches `np.clip(rtop, 0, ...)` in dipy).
pub fn rtop(coefs: &[f64], info: &ScalarBasisInfo) -> f64 {
    let e0 = predict_e0(coefs, info);
    if !e0.is_finite() || e0 == 0.0 {
        return f64::NAN;
    }
    let zeta = info.zeta;
    let mut acc = 0.0;
    for n in 0..=info.radial_order {
        let idx = info.n00_index(n);
        let c_norm = coefs[idx] / e0;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let weight = (16.0 * PI * zeta.powf(1.5) * gamma(n as f64 + 1.5)
            / factorial(n))
        .sqrt();
        acc += c_norm * sign * weight;
    }
    acc.max(0.0)
}

/// Mean Squared Displacement (closed form from `dipy/reconst/shore.py:428-463`).
/// Clamped to ≥ 0 — MSD is physically `∫|r|²P(r) d³r ≥ 0`.
pub fn msd(coefs: &[f64], info: &ScalarBasisInfo) -> f64 {
    let e0 = predict_e0(coefs, info);
    if !e0.is_finite() || e0 == 0.0 {
        return f64::NAN;
    }
    let zeta = info.zeta;
    let mut acc = 0.0;
    for n in 0..=info.radial_order {
        let idx = info.n00_index(n);
        let c_norm = coefs[idx] / e0;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let weight = (9.0 * gamma(n as f64 + 1.5)
            / (8.0 * PI.powi(6) * zeta.powf(3.5) * factorial(n)))
        .sqrt();
        let hyp = hyp2f1(-(n as f64), 2.5, 1.5, 2.0);
        acc += c_norm * sign * weight * hyp;
    }
    acc.max(0.0)
}

/// Q-space Inverse Variance — `QIV = 1 / ∫ E(q)|q|² d³q`.
///
/// Hosseinbor 2013 / dipy iso-MAPMRI definition (`dipy/reconst/mapmri.py:754-797`).
/// Implemented directly from the closed-form q-space second-moment integral
/// over the BrainSuiteSHORE basis, then inverted — much cleaner than trying to
/// extend dipy's clever iso-MAPMRI linear-in-coefficient kernel to BrainSuite's
/// larger basis subset (the latter anti-correlates with truth because of how
/// the formula's alternating signs interact with extra odd-Laguerre modes).
///
/// Derivation: only ℓ=0 modes contribute (angular orthogonality). For each
/// `(n, 0, 0)` mode the radial integral against `q²·q²dq` is a standard
/// Laguerre integral (Gradshteyn 7.414.4) yielding
/// `6√(2π) · κ(ζ,n,0) · ζ^{5/2} · Γ(n+3/2)/n! · ₂F₁(−n, 5/2; 3/2; 2)`.
pub fn qiv(coefs: &[f64], info: &ScalarBasisInfo) -> f64 {
    let e0 = predict_e0(coefs, info);
    if !e0.is_finite() || e0 == 0.0 {
        return f64::NAN;
    }
    let zeta = info.zeta;
    let zeta_pow = zeta.powf(2.5);
    let prefactor = 6.0 * (2.0 * PI).sqrt() * zeta_pow;
    let mut q2_integral = 0.0;
    for n in 0..=info.radial_order {
        let idx = info.n00_index(n);
        let c_norm = coefs[idx] / e0;
        let kappa_n = kappa(zeta, n, 0);
        let g = gamma(n as f64 + 1.5);
        let n_fact = factorial(n);
        let hyp = hyp2f1(-(n as f64), 2.5, 1.5, 2.0);
        q2_integral += c_norm * kappa_n * g / n_fact * hyp;
    }
    let q2 = prefactor * q2_integral;
    // QIV is `1 / ∫E|q|² d³q`; the integral is positive for any physical signal,
    // but a noisy fit can produce a near-zero or negative integral. Clamp to
    // NaN for non-positive integrals so the output volume is interpretable.
    if !q2.is_finite() || q2 <= 0.0 {
        return f64::NAN;
    }
    1.0 / q2
}

/// Non-Gaussianity — coefficient energy ratio between the (0,0,0) Gaussian mode
/// and the full coefficient vector. From `dipy/reconst/mapmri.py:799-824`.
pub fn ng(coefs: &[f64], info: &ScalarBasisInfo) -> f64 {
    let e0 = predict_e0(coefs, info);
    if !e0.is_finite() || e0 == 0.0 {
        return f64::NAN;
    }
    let mut total: f64 = 0.0;
    for &c in coefs {
        let cn = c / e0;
        total += cn * cn;
    }
    if total == 0.0 {
        return 0.0;
    }
    let c0 = coefs[0] / e0;
    let arg = 1.0 - (c0 * c0) / total;
    arg.max(0.0).sqrt()
}

/// Return-to-Axis Probability — propagator integrated over the plane normal
/// to the principal direction `dir`.
///
/// From `dipy/reconst/mapmri.py:627-679`, generalized to all BrainSuite
/// (n, ℓ, m) modes by setting `j = n − ℓ + 1` (the dipy kernel formula is
/// derived as a Laguerre integral identity that holds for any non-negative
/// Laguerre order).
///
/// Returns NaN when `dir` is the zero vector (no detected peak).
pub fn rtap(coefs: &[f64], info: &ScalarBasisInfo, dir: [f64; 3]) -> f64 {
    direction_dependent_propagator(coefs, info, dir, |j, ell| rtap_kappa(j, ell), |zeta| {
        let mu = 1.0 / (2.0 * PI * zeta.sqrt());
        1.0 / (mu * mu)
    })
}

/// Return-to-Plane Probability — propagator integrated along `dir`.
pub fn rtpp(coefs: &[f64], info: &ScalarBasisInfo, dir: [f64; 3]) -> f64 {
    direction_dependent_propagator(coefs, info, dir, |j, ell| rtpp_kappa(j, ell), |zeta| {
        let mu = 1.0 / (2.0 * PI * zeta.sqrt());
        1.0 / mu
    })
}

fn direction_dependent_propagator<KFn, PFn>(
    coefs: &[f64],
    info: &ScalarBasisInfo,
    dir: [f64; 3],
    kappa_kernel: KFn,
    prefactor: PFn,
) -> f64
where
    KFn: Fn(u32, u32) -> f64,
    PFn: Fn(f64) -> f64,
{
    let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return f64::NAN;
    }
    let e0 = predict_e0(coefs, info);
    if !e0.is_finite() || e0 == 0.0 {
        return f64::NAN;
    }
    let zeta = info.zeta;
    let inv_norm = 1.0 / norm;
    let dx = dir[0] * inv_norm;
    let dy = dir[1] * inv_norm;
    let dz = dir[2] * inv_norm;
    let (_, theta, phi) = crate::math::cart2sphere(dx, dy, dz);

    let mut acc = 0.0;
    for blk in &info.nl_blocks {
        let j = (blk.n - blk.ell + 1) as u32;
        let kernel = kappa_kernel(j, blk.ell);
        if kernel == 0.0 {
            continue;
        }
        let alpha = alpha_factor(blk.n, blk.ell, zeta);
        let scale = kernel / alpha;
        let sh_block = brainsuite_sh_block(blk.ell, theta, phi);
        let mut block_inner = 0.0;
        for k in 0..blk.len {
            block_inner += (coefs[blk.start + k] / e0) * sh_block[k];
        }
        acc += scale * block_inner;
    }
    // Clamp to ≥ 0 — RTAP and RTPP are restriction probabilities and noisy
    // fits can push raw estimates slightly negative.
    (prefactor(zeta) * acc).max(0.0)
}

/// Per-mode conversion factor α(n, ℓ) defined in `microstructure_math.md` §1.
///
/// `c_iso = c_BS / α(n, ℓ)` when `j = n − ℓ + 1`.
fn alpha_factor(n: u32, ell: u32, zeta: f64) -> f64 {
    let sign = if (ell / 2) % 2 == 0 { 1.0 } else { -1.0 };
    let num = 2.0 * PI * zeta.powf(1.5) * gamma(n as f64 + 1.5);
    let denom = (2.0_f64).powi(ell as i32) * factorial(n - ell);
    sign * (num / denom).sqrt()
}

/// Per-(j, ℓ) RTAP kernel — eq C11 of Fick 2016b, dipy iso-MAPMRI form.
///
/// Matches `dipy/reconst/mapmri.py:653-664`: per-mode kernel
/// `κ_{j,ℓ} · matsum`, then the entire vector multiplied by 2 after the
/// (j, ℓ, m) loop (line 664: `rtap_vec *= 2`).
fn rtap_kappa(j: u32, ell: u32) -> f64 {
    let jf = j as f64;
    let ellf = ell as f64;
    let kappa = (-1.0_f64).powi(j as i32 - 1) * (2.0_f64).powf(-(ellf + 3.0) / 2.0) / PI;
    let mut sum = 0.0;
    for k in 0..j {
        let kf = k as f64;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let bin = binomial_float(jf + ellf - 0.5, j - k - 1);
        let g = gamma((ellf + 1.0) / 2.0 + kf);
        let half_pow = (0.5_f64).powf((ellf + 1.0) / 2.0 + kf);
        sum += sign * bin * g / (factorial(k) * half_pow);
    }
    // The post-loop factor of 2 from `rtap_vec *= 2` in dipy.
    2.0 * kappa * sum
}

/// Per-(j, ℓ) RTPP kernel — eq C11 of Fick 2016b for the on-axis integral.
fn rtpp_kappa(j: u32, ell: u32) -> f64 {
    let jf = j as f64;
    let ellf = ell as f64;
    let prefactor = (-0.5_f64).powf(ellf / 2.0) / PI.sqrt();
    let mut sum = 0.0;
    for k in 0..j {
        let kf = k as f64;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let bin = binomial_float(jf + ellf - 0.5, j - k - 1);
        let g = gamma(ellf / 2.0 + kf + 0.5);
        let half_pow = (0.5_f64).powf(ellf / 2.0 + 0.5 + kf);
        sum += sign * bin * g / (factorial(k) * half_pow);
    }
    prefactor * sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShoreBasis;
    use approx::assert_abs_diff_eq;

    #[test]
    fn n00_indexing_lines_up_with_basis() {
        let basis = ShoreBasis::new(6, 700.0);
        let info = ScalarBasisInfo::from_basis(&basis);
        for n in 0..=6 {
            let i = info.n00_index(n);
            assert_eq!(info.indices[i], (n, 0, 0));
        }
    }

    #[test]
    fn ng_zero_for_pure_gaussian_coefficient() {
        let basis = ShoreBasis::new(4, 700.0);
        let info = ScalarBasisInfo::from_basis(&basis);
        let mut coefs = vec![0.0; info.n_coeffs];
        coefs[0] = 1.0; // only the (0,0,0) Gaussian mode is populated
        // ng = sqrt(1 - c0²/sum c²) = 0 when only c0 is non-zero.
        assert_abs_diff_eq!(ng(&coefs, &info), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn alpha_factor_matches_manual_for_n0_l0() {
        // α(0, 0) = + sqrt(2π · ζ^1.5 · Γ(3/2) / 1)
        let zeta: f64 = 700.0;
        let expected = (2.0 * PI * zeta.powf(1.5) * gamma(1.5)).sqrt();
        assert_abs_diff_eq!(alpha_factor(0, 0, zeta), expected, epsilon = 1e-9);
    }

    #[test]
    fn rtap_returns_nan_for_zero_direction() {
        let basis = ShoreBasis::new(4, 700.0);
        let info = ScalarBasisInfo::from_basis(&basis);
        let coefs = vec![1.0; info.n_coeffs];
        let v = rtap(&coefs, &info, [0.0, 0.0, 0.0]);
        assert!(v.is_nan());
    }
}
