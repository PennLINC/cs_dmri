// SPDX-License-Identifier: MIT OR Apache-2.0
//! Internal math helpers: factorials, gamma, associated Legendre, generalized Laguerre.

use std::f64::consts::PI;

/// Exact factorial via lookup; falls back to gamma for large n.
#[inline]
pub fn factorial(n: u32) -> f64 {
    const TABLE: [f64; 21] = [
        1.0,
        1.0,
        2.0,
        6.0,
        24.0,
        120.0,
        720.0,
        5040.0,
        40320.0,
        362_880.0,
        3_628_800.0,
        39_916_800.0,
        479_001_600.0,
        6_227_020_800.0,
        87_178_291_200.0,
        1_307_674_368_000.0,
        20_922_789_888_000.0,
        355_687_428_096_000.0,
        6_402_373_705_728_000.0,
        121_645_100_408_832_000.0,
        2_432_902_008_176_640_000.0,
    ];
    if (n as usize) < TABLE.len() {
        TABLE[n as usize]
    } else {
        gamma((n + 1) as f64)
    }
}

/// Lanczos approximation to the gamma function (good to ~14 digits for x > 0).
pub fn gamma(x: f64) -> f64 {
    // Reflection for x < 0.5
    if x < 0.5 {
        return PI / ((PI * x).sin() * gamma(1.0 - x));
    }
    let g = 7.0;
    let coeffs = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = coeffs[0];
    let t = x + g + 0.5;
    for (i, &c) in coeffs.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
}

/// Associated Legendre function P_l^m(x) with the Condon–Shortley phase
/// (i.e. matches scipy.special.lpmv for m >= 0).
///
/// Stable forward recurrence in `l` starting from P_m^m.
pub fn assoc_legendre(l: u32, m: u32, x: f64) -> f64 {
    debug_assert!(m <= l);
    // P_m^m(x) = (-1)^m * (2m-1)!! * (1 - x^2)^(m/2)
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).max(0.0).sqrt(); // sqrt(1 - x^2)
        let mut fact = 1.0;
        for _ in 0..m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if l == m {
        return pmm;
    }
    // P_{m+1}^m(x) = x * (2m + 1) * P_m^m(x)
    let mut pmmp1 = x * (2.0 * m as f64 + 1.0) * pmm;
    if l == m + 1 {
        return pmmp1;
    }
    // P_l^m(x) = ((2l - 1) * x * P_{l-1}^m(x) - (l + m - 1) * P_{l-2}^m(x)) / (l - m)
    let mut pll = 0.0;
    for ll in (m + 2)..=l {
        let llf = ll as f64;
        let mf = m as f64;
        pll = ((2.0 * llf - 1.0) * x * pmmp1 - (llf + mf - 1.0) * pmm) / (llf - mf);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

/// Generalized Laguerre polynomial L_n^alpha(x) via the standard recurrence.
///
/// Recurrence:
/// L_0 = 1
/// L_1 = 1 + alpha - x
/// (k+1) L_{k+1} = (2k + 1 + alpha - x) L_k - (k + alpha) L_{k-1}
pub fn gen_laguerre(n: u32, alpha: f64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut lkm1 = 1.0_f64;
    let mut lk = 1.0 + alpha - x;
    if n == 1 {
        return lk;
    }
    for k in 1..n {
        let kf = k as f64;
        let lkp1 = ((2.0 * kf + 1.0 + alpha - x) * lk - (kf + alpha) * lkm1) / (kf + 1.0);
        lkm1 = lk;
        lk = lkp1;
    }
    lk
}

/// Gauss hypergeometric ₂F₁(a, b; c; z) via the power series.
///
/// In the SHORE→ODF projection, `a = ℓ - n` is always a non-positive even
/// integer, so the series terminates after at most `n - ℓ + 1` terms.
pub fn hyp2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    for k in 0..200 {
        let kf = k as f64;
        term *= (a + kf) * (b + kf) / ((c + kf) * (kf + 1.0)) * z;
        sum += term;
        if term == 0.0 {
            break;
        }
        if term.abs() < 1e-15 * sum.abs().max(1.0) {
            break;
        }
    }
    sum
}

/// Real-valued binomial coefficient `(n choose k) = Γ(n+1) / (Γ(k+1) Γ(n-k+1))`,
/// matching dipy's `binomialfloat`. `n` may be a real number; `k` is a
/// non-negative integer.
pub fn binomial_float(n: f64, k: u32) -> f64 {
    gamma(n + 1.0) / (factorial(k) * gamma(n - k as f64 + 1.0))
}

/// Cartesian (x, y, z) -> (r, theta, phi) where theta is azimuth and phi is polar
/// (matching dipy.core.geometry.cart2sphere).
#[inline]
pub fn cart2sphere(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    let theta = y.atan2(x);
    let phi = if r > 0.0 {
        (z / r).clamp(-1.0, 1.0).acos()
    } else {
        0.0
    };
    (r, theta, phi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn factorial_basics() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3_628_800.0);
    }

    #[test]
    fn gamma_matches_factorial() {
        for n in 1..=10 {
            assert_abs_diff_eq!(gamma(n as f64), factorial(n - 1), epsilon = 1e-6);
        }
    }

    #[test]
    fn legendre_known_values() {
        // P_0^0(x) = 1
        assert_abs_diff_eq!(assoc_legendre(0, 0, 0.3), 1.0, epsilon = 1e-12);
        // P_1^0(x) = x
        assert_abs_diff_eq!(assoc_legendre(1, 0, 0.3), 0.3, epsilon = 1e-12);
        // P_1^1(x) = -sqrt(1 - x^2)
        assert_abs_diff_eq!(
            assoc_legendre(1, 1, 0.3),
            -((1.0_f64 - 0.09).sqrt()),
            epsilon = 1e-12
        );
        // P_2^0(x) = 0.5 (3 x^2 - 1)
        assert_abs_diff_eq!(
            assoc_legendre(2, 0, 0.4),
            0.5 * (3.0 * 0.16 - 1.0),
            epsilon = 1e-12
        );
        // P_2^2(x) = 3 (1 - x^2)
        assert_abs_diff_eq!(assoc_legendre(2, 2, 0.4), 3.0 * (1.0 - 0.16), epsilon = 1e-12);
    }

    #[test]
    fn hyp2f1_terminating_series() {
        // Reference values from scipy.special.hyp2f1 (matches the dipy SHORE
        // ODF formula at the relevant arguments). All instances where the
        // first argument is 0 must return exactly 1.
        // hyp2f1(-n + ell, ell/2 + 1.5, ell + 1.5, 2)
        let cases = [
            (0_i32, 0_u32, 1.0_f64),
            (-2, 0, 1.0),
            (-4, 0, 1.0),
            (-6, 0, 1.0),
            (-2, 2, 3.650_793_650_793_651e-1),
            (-4, 2, 2.274_392_274_392_287e-1),
            (-2, 4, 2.167_832_167_832_167e-1),
        ];
        for (a, ell, expected) in cases {
            let got = hyp2f1(
                a as f64,
                ell as f64 / 2.0 + 1.5,
                ell as f64 + 1.5,
                2.0,
            );
            assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn laguerre_known_values() {
        // L_0(x) = 1
        assert_abs_diff_eq!(gen_laguerre(0, 0.5, 0.7), 1.0, epsilon = 1e-12);
        // L_1^a(x) = 1 + a - x
        assert_abs_diff_eq!(gen_laguerre(1, 0.5, 0.7), 1.0 + 0.5 - 0.7, epsilon = 1e-12);
        // L_2^a(x) = ((1+a)(2+a)/2) - (2+a) x + x^2 / 2
        let x = 0.7;
        let a = 0.5;
        let expected = (1.0 + a) * (2.0 + a) / 2.0 - (2.0 + a) * x + x * x / 2.0;
        assert_abs_diff_eq!(gen_laguerre(2, a, x), expected, epsilon = 1e-12);
    }
}
