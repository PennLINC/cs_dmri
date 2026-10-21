// SPDX-License-Identifier: MIT OR Apache-2.0
//! Parametric dense-b response: axially symmetric DKI + offset (Morez et al. 2021).
//!
//! Implements the response model of Morez, Sijbers, Vanhevel & Jeurissen,
//! *"Constrained spherical deconvolution of nonspherically sampled diffusion
//! MRI data"*, Hum Brain Mapp 42:521–538 (2021), Eqs. (7)–(11) and (18), as a
//! third dense-b row estimator next to the kernel and smooth fits — so the
//! non-parametric responses can be compared head-to-head with the published
//! parametric one under the same tissue selection, scaling and deconvolution.
//!
//! The response is one closed-form function of `(b, cos θ)` fit to all pooled
//! samples at once:
//!
//! ```text
//!   R(b, c) = S0 · exp(−b D(c) + b² W(c)) + C
//!   D(c)    = D⊥ + (D∥ − D⊥) c²                  (axially symmetric 2nd-order tensor)
//!   W(c)    = w₀ + w₂ c² + w₄ c⁴                 (axially symmetric 4th-order tensor)
//! ```
//!
//! — seven parameters `[S0, D∥, D⊥, w₀, w₂, w₄, C]` for WM ("DKI AS + offset"),
//! and four `[S0, D, W, C]` for the isotropic GM/CSF ("DKI ISO + offset"). `W`
//! is the paper's `W(g)`, i.e. it absorbs the `MD²/6` of the kurtosis
//! convention. `c = |cos θ|` to the fiber axis.
//!
//! **Constraint.** The signal must never grow with b on `[0, bmax]` in any
//! direction (Eq. 10). `dR/db = −S0 · E · (D(c) − 2b W(c))`, so with `S0 ≥ 0`
//! this is `D(c) − 2b W(c) ≥ 0` for every `b ≤ bmax` — linear in b, hence
//! exactly `D(c) ≥ 0` and `D(c) − 2 bmax W(c) ≥ 0` (Eq. 11 is the second). Both
//! are enforced on a dense set of `c ∈ [0, 1]`. `bmax` is the top of the dense-b
//! grid, so rows extrapolated past the data (gradient deviation) stay monotone
//! too. Beyond the paper, `S0 ≥ 0` and `C ≥ 0` (a noise floor is non-negative).
//!
//! **Fit.** Weighted nonlinear least squares (Eq. 18) by constrained
//! Gauss–Newton: each step solves the linearized problem for the new parameter
//! vector directly — the constraints are homogeneous in the parameters, which
//! the ICLS solver requires — then backtracks along the segment from the
//! current iterate, which stays feasible because the feasible set is convex.
//! Initialized from a log-linear DKI fit without the offset.
//!
//! **Rows.** Each dense-b row is the exact zonal projection of `R(b, ·)` — the
//! coefficients deconvolution needs by the convolution theorem. The paper fits a
//! zonal series to `R` on a quarter circle instead; on a dense uniform sampling
//! in `c` that fit and this projection coincide.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};

use crate::math::assoc_legendre;
use crate::multitissue::consh::DenseBGrid;
use crate::solver::icls::{IclsConfig, IclsSolver};
use crate::{CsDmriError, Result};

/// b is carried in ms/µm² (`b / 1000` in s/mm²) inside the fit, so diffusivities
/// are O(1) µm²/ms and the normal equations are well scaled.
const B_SCALE: f64 = 1e-3;
/// Number of `c ∈ [0, 1]` points the monotonicity constraint is enforced on.
const N_CONSTRAINT_COS: usize = 33;
/// Gauss–Legendre nodes on `c ∈ [0, 1]` for the zonal projection of a row
/// (exact for the `P_l` part up to degree 127; `R` itself is smooth in `c`).
const N_PROJECTION: usize = 64;
/// Gauss–Newton iteration cap.
const MAX_GN_ITER: usize = 200;

/// Fitted DKI + offset response parameters. Diffusivities in µm²/ms
/// (= 10⁻³ mm²/s), `w` in (µm²/ms)², `s0` and `offset` on the response's
/// common-reference signal scale. Isotropic fits have `d_par == d_perp` and
/// `w = [W, 0, 0]`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct DkiResponseParams {
    pub s0: f64,
    pub d_par: f64,
    pub d_perp: f64,
    /// `W(c) = w[0] + w[1] c² + w[2] c⁴`.
    pub w: [f64; 3],
    pub offset: f64,
    pub isotropic: bool,
}

impl DkiResponseParams {
    /// Apparent diffusivity along `c = |cos θ|`, µm²/ms.
    pub fn diffusivity(&self, c: f64) -> f64 {
        self.d_perp + (self.d_par - self.d_perp) * c * c
    }

    /// `W(c)`, (µm²/ms)².
    pub fn kurtosis_term(&self, c: f64) -> f64 {
        let c2 = c * c;
        self.w[0] + self.w[1] * c2 + self.w[2] * c2 * c2
    }

    /// Response signal at `b` (s/mm²) and `c = |cos θ|`.
    pub fn signal(&self, b: f64, c: f64) -> f64 {
        let bs = b * B_SCALE;
        self.s0 * (-bs * self.diffusivity(c) + bs * bs * self.kurtosis_term(c)).exp() + self.offset
    }

    /// Number of free parameters (7 anisotropic, 4 isotropic).
    pub fn n_params(&self) -> usize {
        if self.isotropic { 4 } else { 7 }
    }
}

/// A fitted response and its goodness of fit over the pooled samples.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct DkiResponseFit {
    pub params: DkiResponseParams,
    /// Root mean squared residual (Eq. 19; weighted when weights were given).
    pub rmsr: f64,
    /// Akaike Information Criterion `n log(RMSR²) + 2 nθ` (Eq. 21).
    pub aic: f64,
    pub n_samples: usize,
    pub iterations: usize,
    pub converged: bool,
}

/// The parametric fits behind a [`ResponseEstimator::Dki`] response set.
///
/// [`ResponseEstimator::Dki`]: crate::multitissue::response_estimation::ResponseEstimator::Dki
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct DkiResponseFits {
    pub wm: DkiResponseFit,
    pub gm: DkiResponseFit,
    pub csf: DkiResponseFit,
}

/// Parameter layout `θ = [S0, d…, w…, C]`: `D(c) = Σ d_k φ_k(c)`,
/// `W(c) = Σ w_k ψ_k(c)`. Anisotropic: `φ = [c², 1 − c²]` (so `d = [D∥, D⊥]`),
/// `ψ = [1, c², c⁴]`. Isotropic: `φ = ψ = [1]`.
struct Layout {
    nd: usize,
    nw: usize,
}

impl Layout {
    fn new(isotropic: bool) -> Self {
        if isotropic { Self { nd: 1, nw: 1 } } else { Self { nd: 2, nw: 3 } }
    }
    fn n(&self) -> usize {
        2 + self.nd + self.nw
    }
    fn phi(&self, c: f64, out: &mut [f64]) {
        if self.nd == 1 {
            out[0] = 1.0;
        } else {
            out[0] = c * c;
            out[1] = 1.0 - c * c;
        }
    }
    fn psi(&self, c: f64, out: &mut [f64]) {
        if self.nw == 1 {
            out[0] = 1.0;
        } else {
            out[0] = 1.0;
            out[1] = c * c;
            out[2] = c * c * c * c;
        }
    }
    fn params(&self, t: &[f64]) -> DkiResponseParams {
        let (d_par, d_perp) = if self.nd == 1 { (t[1], t[1]) } else { (t[1], t[2]) };
        let w0 = 1 + self.nd;
        let w = if self.nw == 1 { [t[w0], 0.0, 0.0] } else { [t[w0], t[w0 + 1], t[w0 + 2]] };
        DkiResponseParams { s0: t[0], d_par, d_perp, w, offset: t[self.n() - 1], isotropic: self.nd == 1 }
    }

    /// Homogeneous constraint rows `C θ ≥ 0` on `θ` (or, with `with_s0_c =
    /// false`, on the log-linear initializer's `[ln S0, d…, w…]`).
    fn constraints(&self, bmax_s: f64, with_s0_c: bool) -> DMatrix<f64> {
        let n = if with_s0_c { self.n() } else { self.n() - 1 };
        let cs: Vec<f64> = if self.nd == 1 {
            vec![0.0]
        } else {
            (0..N_CONSTRAINT_COS).map(|j| j as f64 / (N_CONSTRAINT_COS - 1) as f64).collect()
        };
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let (mut phi, mut psi) = (vec![0.0; self.nd], vec![0.0; self.nw]);
        for &c in &cs {
            self.phi(c, &mut phi);
            self.psi(c, &mut psi);
            // D(c) ≥ 0  and  D(c) − 2 bmax W(c) ≥ 0
            for k in 0..2 {
                let mut r = vec![0.0; n];
                for (j, &p) in phi.iter().enumerate() {
                    r[1 + j] = p;
                }
                if k == 1 {
                    for (j, &p) in psi.iter().enumerate() {
                        r[1 + self.nd + j] = -2.0 * bmax_s * p;
                    }
                }
                rows.push(r);
            }
        }
        if with_s0_c {
            let mut r = vec![0.0; n];
            r[0] = 1.0; // S0 ≥ 0
            rows.push(r);
            let mut r = vec![0.0; n];
            r[n - 1] = 1.0; // C ≥ 0
            rows.push(r);
        }
        DMatrix::from_fn(rows.len(), n, |i, j| rows[i][j])
    }
}

/// Weighted constrained least squares `min ‖√w (A x − y)‖²  s.t. C x ≥ 0`, via
/// its normal equations: `G = AᵀWA = L Lᵀ`, solved as `½‖Lᵀx − L⁻¹AᵀWy‖²`.
fn constrained_lsq(g: DMatrix<f64>, rhs: DVector<f64>, cmat: DMatrix<f64>) -> Result<DVector<f64>> {
    let n = g.nrows();
    let scale = (0..n).map(|i| g[(i, i)]).fold(0.0_f64, f64::max).max(1e-300);
    let g = g + DMatrix::<f64>::identity(n, n) * (1e-12 * scale);
    let chol = nalgebra::Cholesky::new(g)
        .ok_or_else(|| CsDmriError::Other("DKI response fit: normal equations not positive definite".into()))?;
    let l = chol.l();
    let b = l
        .solve_lower_triangular(&rhs)
        .ok_or_else(|| CsDmriError::Other("DKI response fit: triangular solve failed".into()))?;
    let solver = IclsSolver::new(l.transpose(), cmat, IclsConfig { max_iter: 10_000, ..IclsConfig::default() });
    let (x, diag) = solver.solve(&b);
    if !diag.converged {
        return Err(CsDmriError::Other("DKI response fit: constrained step did not converge".into()));
    }
    Ok(x)
}

/// Fit the DKI + offset response to pooled samples. `pool_b` in s/mm²;
/// `pool_cos` the samples' `|cos θ|` to the fiber axis, or `None` for an
/// isotropic tissue; `pool_y` the signal (b=0 samples included — they fix
/// `S0 + C`); `pool_w` optional least-squares weights. `bmax` (s/mm²) is the
/// upper end of the range the response must be non-increasing on.
pub fn fit_dki_response(
    pool_b: &[f64],
    pool_cos: Option<&[f64]>,
    pool_y: &[f64],
    pool_w: Option<&[f64]>,
    bmax: f64,
) -> Result<DkiResponseFit> {
    let lay = Layout::new(pool_cos.is_none());
    let n = lay.n();
    let ns = pool_b.len();
    if ns < n {
        return Err(CsDmriError::Other(format!("DKI response fit: {ns} samples for {n} parameters")));
    }
    let bmax_s = bmax * B_SCALE;
    let cos = |i: usize| pool_cos.map_or(0.0, |c| c[i]);
    let wt = |i: usize| pool_w.map_or(1.0, |w| w[i].max(0.0));
    let (mut phi, mut psi) = (vec![0.0; lay.nd], vec![0.0; lay.nw]);

    // ---- initializer: log-linear DKI without offset, weights w·y² ----
    let ymax = pool_y.iter().cloned().fold(0.0_f64, f64::max);
    let nl = n - 1;
    let mut g = DMatrix::<f64>::zeros(nl, nl);
    let mut c = DVector::<f64>::zeros(nl);
    let mut a = vec![0.0; nl];
    for i in 0..ns {
        let y = pool_y[i];
        if y <= 1e-6 * ymax {
            continue;
        }
        let bs = pool_b[i] * B_SCALE;
        lay.phi(cos(i), &mut phi);
        lay.psi(cos(i), &mut psi);
        a[0] = 1.0;
        for j in 0..lay.nd {
            a[1 + j] = -bs * phi[j];
        }
        for j in 0..lay.nw {
            a[1 + lay.nd + j] = bs * bs * psi[j];
        }
        let w = wt(i) * y * y;
        let ly = y.ln();
        for p in 0..nl {
            c[p] += w * a[p] * ly;
            for q in 0..nl {
                g[(p, q)] += w * a[p] * a[q];
            }
        }
    }
    let x0 = constrained_lsq(g, c, lay.constraints(bmax_s, false))?;
    let mut theta = vec![0.0; n];
    theta[0] = x0[0].exp();
    theta[1..nl].copy_from_slice(&x0.as_slice()[1..nl]);
    theta[n - 1] = 0.0;

    let sse = |t: &[f64]| -> f64 {
        let p = lay.params(t);
        (0..ns).map(|i| wt(i) * (pool_y[i] - p.signal(pool_b[i], cos(i))).powi(2)).sum()
    };
    let cmat = lay.constraints(bmax_s, true);
    let mut cur = sse(&theta);
    let mut converged = false;
    let mut iterations = 0;
    let mut jrow = vec![0.0; n];
    for it in 0..MAX_GN_ITER {
        iterations = it + 1;
        // Linearize f(θ') ≈ f(θ) + J(θ' − θ); solve for θ' directly.
        let p = lay.params(&theta);
        let mut g = DMatrix::<f64>::zeros(n, n);
        let mut rhs = DVector::<f64>::zeros(n);
        for i in 0..ns {
            let bs = pool_b[i] * B_SCALE;
            let ci = cos(i);
            lay.phi(ci, &mut phi);
            lay.psi(ci, &mut psi);
            let e = (-bs * p.diffusivity(ci) + bs * bs * p.kurtosis_term(ci)).exp();
            jrow[0] = e;
            for j in 0..lay.nd {
                jrow[1 + j] = -p.s0 * e * bs * phi[j];
            }
            for j in 0..lay.nw {
                jrow[1 + lay.nd + j] = p.s0 * e * bs * bs * psi[j];
            }
            jrow[n - 1] = 1.0;
            let f = p.s0 * e + p.offset;
            let jt: f64 = jrow.iter().zip(&theta).map(|(a, b)| a * b).sum();
            let target = pool_y[i] - f + jt;
            let w = wt(i);
            for a in 0..n {
                rhs[a] += w * jrow[a] * target;
                for b in 0..n {
                    g[(a, b)] += w * jrow[a] * jrow[b];
                }
            }
        }
        let proposal = constrained_lsq(g, rhs, cmat.clone())?;
        // Backtrack along θ → θ'; the segment is feasible (convex constraints).
        let mut step = 1.0;
        let mut accepted = None;
        while step > 1e-8 {
            let cand: Vec<f64> = (0..n).map(|k| theta[k] + step * (proposal[k] - theta[k])).collect();
            let s = sse(&cand);
            if s < cur {
                accepted = Some((cand, s));
                break;
            }
            step *= 0.5;
        }
        match accepted {
            Some((cand, s)) => {
                let rel = (cur - s) / cur.max(1e-300);
                theta = cand;
                cur = s;
                if rel < 1e-12 {
                    converged = true;
                    break;
                }
            }
            None => {
                converged = true; // no descent along the GN direction: stationary
                break;
            }
        }
    }

    let wsum: f64 = (0..ns).map(wt).sum::<f64>().max(1e-300);
    let mse = cur / wsum;
    Ok(DkiResponseFit {
        params: lay.params(&theta),
        rmsr: mse.sqrt(),
        aic: ns as f64 * mse.max(1e-300).ln() + 2.0 * n as f64,
        n_samples: ns,
        iterations,
        converged,
    })
}

/// Dense-b zonal rows of a fitted response: row `k` is the projection of
/// `R(grid.bs[k], ·)` onto `Y_{l0}`, `l = 0, 2, …, lmax` (MRtrix normalization,
/// `√((2l+1)/4π) P_l`). An isotropic response gives `lmax = 0` rows, `√4π R(b)`.
pub fn dki_dense_rows(params: &DkiResponseParams, grid: &DenseBGrid, lmax: usize) -> Vec<Vec<f64>> {
    let nz = lmax / 2 + 1;
    let (nodes, weights) = gauss_legendre_01(N_PROJECTION);
    let integrate = |f: &dyn Fn(f64) -> f64| -> f64 { nodes.iter().zip(&weights).map(|(&x, &w)| w * f(x)).sum() };
    grid.bs
        .iter()
        .map(|&b| {
            if params.isotropic || nz == 1 {
                let mut row = vec![0.0; nz];
                // Isotropic: R does not depend on c. Anisotropic at lmax 0: the mean.
                row[0] = (4.0 * PI).sqrt() * integrate(&|c| params.signal(b, c));
                return row;
            }
            (0..nz)
                .map(|j| {
                    let l = 2 * j;
                    let norm = ((2 * l + 1) as f64 / (4.0 * PI)).sqrt();
                    // ∫_{S²} R Y_l0 = 2π ∫_{−1}^{1} R Y_l0 dc = 4π ∫_0^1 (even integrand)
                    4.0 * PI * norm * integrate(&|c| params.signal(b, c) * assoc_legendre(l as u32, 0, c))
                })
                .collect()
        })
        .collect()
}

/// `n`-point Gauss–Legendre nodes and weights on `[0, 1]` (Newton on `P_n`).
fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut xs, mut ws) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            // P_n(x) and P_n'(x) by the three-term recurrence.
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        xs.push(0.5 * (x + 1.0));
        ws.push(1.0 / ((1.0 - x * x) * dp * dp)); // 2/((1−x²)P'²), halved for [0, 1]
    }
    (xs, ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wm_truth() -> DkiResponseParams {
        DkiResponseParams {
            s0: 1.0,
            d_par: 1.9,
            d_perp: 0.45,
            w: [0.02, 0.05, 0.12],
            offset: 0.01,
            isotropic: false,
        }
    }

    /// Samples on b = 0 and shells 1000…4000 at a spread of |cos θ|, plus an
    /// off-shell (Cartesian-like) scatter in b.
    fn samples(truth: &DkiResponseParams, aniso: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut b, mut c, mut y) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..10 {
            b.push(0.0);
            c.push(k as f64 / 9.0);
        }
        for &bv in &[1000.0, 2000.0, 3000.0, 4000.0] {
            for k in 0..30 {
                b.push(bv);
                c.push(k as f64 / 29.0);
            }
        }
        for k in 0..200 {
            b.push(37.0 * k as f64 % 4000.0);
            c.push(((k * 7919) % 101) as f64 / 100.0);
        }
        if !aniso {
            c.iter_mut().for_each(|x| *x = 0.0);
        }
        for i in 0..b.len() {
            y.push(truth.signal(b[i], c[i]));
        }
        (b, c, y)
    }

    #[test]
    fn recovers_planted_anisotropic_parameters() {
        let truth = wm_truth();
        let (b, c, y) = samples(&truth, true);
        let fit = fit_dki_response(&b, Some(&c), &y, None, 4000.0).unwrap();
        assert!(fit.converged);
        let p = fit.params;
        assert_eq!(p.n_params(), 7);
        for (got, want) in [
            (p.s0, truth.s0),
            (p.d_par, truth.d_par),
            (p.d_perp, truth.d_perp),
            (p.w[0], truth.w[0]),
            (p.w[1], truth.w[1]),
            (p.w[2], truth.w[2]),
            (p.offset, truth.offset),
        ] {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
        assert!(fit.rmsr < 1e-6);
    }

    #[test]
    fn recovers_planted_isotropic_parameters() {
        let truth = DkiResponseParams {
            s0: 2.6,
            d_par: 2.8,
            d_perp: 2.8,
            w: [0.3, 0.0, 0.0],
            offset: 0.03,
            isotropic: true,
        };
        let (b, _, y) = samples(&truth, false);
        let fit = fit_dki_response(&b, None, &y, None, 4000.0).unwrap();
        let p = fit.params;
        assert_eq!(p.n_params(), 4);
        assert!((p.s0 - 2.6).abs() < 1e-4);
        assert!((p.d_par - 2.8).abs() < 1e-4);
        assert!((p.w[0] - 0.3).abs() < 1e-4);
        assert!((p.offset - 0.03).abs() < 1e-4);
    }

    #[test]
    fn monotonicity_constraint_holds_when_data_rise_with_b() {
        // A signal that turns back up at high b (W large): the unconstrained DKI
        // fit would follow it; the constrained one must be non-increasing on
        // [0, bmax] in every direction.
        let rising = DkiResponseParams {
            s0: 1.0,
            d_par: 1.0,
            d_perp: 1.0,
            w: [0.3, 0.0, 0.0],
            offset: 0.0,
            isotropic: false,
        };
        let (b, c, y) = samples(&rising, true);
        let bmax = 4000.0;
        let fit = fit_dki_response(&b, Some(&c), &y, None, bmax).unwrap();
        let p = fit.params;
        for j in 0..=20 {
            let cc = j as f64 / 20.0;
            let mut prev = f64::INFINITY;
            for k in 0..=80 {
                let s = p.signal(bmax * k as f64 / 80.0, cc);
                assert!(s <= prev + 1e-9, "signal rises at c={cc}, b={}", bmax * k as f64 / 80.0);
                prev = s;
            }
        }
        assert!(p.offset >= 0.0 && p.s0 >= 0.0);
    }

    #[test]
    fn dense_rows_are_the_zonal_projection() {
        let p = wm_truth();
        let grid = DenseBGrid::new(3000.0, 1000.0, 0.0);
        let rows = dki_dense_rows(&p, &grid, 8);
        assert_eq!(rows.len(), 4);
        // b = 0: R = S0 + C everywhere → only r_0 = √4π (S0 + C).
        assert!((rows[0][0] - (4.0 * PI).sqrt() * (p.s0 + p.offset)).abs() < 1e-10);
        for &r in &rows[0][1..] {
            assert!(r.abs() < 1e-10);
        }
        // The rows reconstruct R(b, c) up to the lmax-8 truncation.
        for (k, row) in rows.iter().enumerate().skip(1) {
            let b = grid.bs[k];
            for &cc in &[0.0, 0.3, 0.7, 1.0] {
                let amp: f64 = row
                    .iter()
                    .enumerate()
                    .map(|(j, r)| {
                        let l = 2 * j;
                        r * ((2 * l + 1) as f64 / (4.0 * PI)).sqrt() * assoc_legendre(l as u32, 0, cc)
                    })
                    .sum();
                assert!((amp - p.signal(b, cc)).abs() < 0.02, "b={b} c={cc}: {amp} vs {}", p.signal(b, cc));
            }
            // WM zonal sign pattern for a fiber-like response.
            assert!(row[0] > 0.0 && row[1] < 0.0 && row[2] > 0.0);
        }
    }
}
