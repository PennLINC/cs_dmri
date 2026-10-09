// SPDX-License-Identifier: MIT OR Apache-2.0
//! Robust diffusion-tensor fit via RESTORE.
//!
//! Implements the algorithm of Chang, Jones & Pierpaoli, *"RESTORE: Robust
//! Estimation of Tensors by Outlier REjection"*, Magnetic Resonance in
//! Medicine 53(5), 1088-1095 (2005). Designed for clinical-quality DWI
//! where a subset of volumes per voxel suffer from motion-induced signal
//! dropouts. Vanilla weighted least squares treats those outliers as if
//! they fit log-Gaussian noise; RESTORE detects them via Studentized
//! residuals and downweights via the Geman-McClure M-estimator, iterating
//! until weights stabilize.
//!
//! Outputs per voxel: tensor `D`, `S₀`, `FA`, `MD`, plus an outlier mask /
//! count that's independently useful as a QC diagnostic for clinical data.
//!
//! License posture: algorithm sourced from the 2005 MRM paper. dipy's
//! BSD-3 `RestoreModel` was consulted for sanity-checking sign/scale
//! conventions; no MRtrix code (`dwi2tensor -method restore`, MPL-2.0)
//! was read.

use nalgebra::{Cholesky, DMatrix, DVector, Matrix3};
use ndarray::{Array3, Array4};

use crate::io::dwi::DwiData;
use crate::voxel_loop;
use crate::{CsDmriError, Result};

/// Configuration for the RESTORE iterative reweighting.
#[derive(Debug, Clone, Copy)]
pub struct RestoreConfig {
    /// Maximum reweighting iterations.
    pub max_iter: usize,
    /// Convergence threshold on relative change in tensor coefficients.
    pub tol: f64,
    /// Geman-McClure weight below which a measurement counts as an outlier
    /// (used only for the outlier-mask diagnostic; doesn't affect the fit).
    pub outlier_threshold: f64,
    /// Lower bound on signal values before taking log (avoids `log 0`).
    pub min_signal: f64,
}

impl Default for RestoreConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-6,
            // GM weight 0.04 ↔ |r| ≈ 2·σ. Threshold 0.5 (the IRLS
            // half-weight point) corresponds to ~0.6·σ and would flag
            // most honest measurements; 0.04 is the conventional 2σ cut
            // for outlier detection. The fit itself is unaffected by this
            // value — only the diagnostic count.
            outlier_threshold: 0.04,
            min_signal: 1e-6,
        }
    }
}

/// Per-voxel RESTORE fit result.
#[derive(Debug, Clone, Copy)]
pub struct DtiVoxelResult {
    /// 3×3 symmetric diffusion tensor.
    pub d: Matrix3<f64>,
    /// `S₀` (signal at b=0, in the same units as the input DWI).
    pub s0: f64,
    /// Fractional anisotropy ∈ [0, 1] (post-clipped from the eigendecomposition;
    /// noise-induced negative eigenvalues can push the raw value > 1).
    pub fa: f64,
    /// Mean diffusivity (trace of `D` / 3).
    pub md: f64,
    /// Sorted eigenvalues `λ₁ ≥ λ₂ ≥ λ₃` of `D`.
    pub eigenvalues: [f64; 3],
    /// Principal eigenvector (column of D's eigenbasis matching `λ₁`).
    pub principal_dir: [f64; 3],
    /// Number of measurements with final weight `< outlier_threshold`.
    pub n_outliers: u32,
    /// Total measurements used (`n_grads`).
    pub n_measurements: u32,
    /// Did the reweighting converge within `max_iter`?
    pub converged: bool,
    /// Number of reweighting iterations actually performed.
    pub iterations: u32,
}

impl DtiVoxelResult {
    /// Outlier fraction `n_outliers / n_measurements`.
    pub fn outlier_fraction(&self) -> f64 {
        if self.n_measurements == 0 {
            0.0
        } else {
            self.n_outliers as f64 / self.n_measurements as f64
        }
    }

    /// Empty / zero-filled result (used for voxels excluded by the mask).
    pub fn zero(n_measurements: u32) -> Self {
        Self {
            d: Matrix3::zeros(),
            s0: 0.0,
            fa: 0.0,
            md: 0.0,
            eigenvalues: [0.0; 3],
            principal_dir: [0.0, 0.0, 0.0],
            n_outliers: 0,
            n_measurements,
            converged: false,
            iterations: 0,
        }
    }
}

/// Build the design matrix `X` for the linearised log-domain DTI model.
///
/// ```text
///   log S_i ≈ log S₀  −  b_i · gᵢᵀ D gᵢ
/// ```
///
/// becomes `log S = X · θ` with
/// `θ = [log S₀, D_xx, D_yy, D_zz, D_xy, D_xz, D_yz]` and
/// `X[i, :] = [1, −b·gx², −b·gy², −b·gz², −2·b·gx·gy, −2·b·gx·gz, −2·b·gy·gz]`.
pub fn build_design_matrix(bvals: &[f64], bvecs: &[[f64; 3]]) -> DMatrix<f64> {
    assert_eq!(bvals.len(), bvecs.len());
    let n = bvals.len();
    let mut x = DMatrix::<f64>::zeros(n, 7);
    for i in 0..n {
        let b = bvals[i];
        let g = bvecs[i];
        x[(i, 0)] = 1.0;
        x[(i, 1)] = -b * g[0] * g[0];
        x[(i, 2)] = -b * g[1] * g[1];
        x[(i, 3)] = -b * g[2] * g[2];
        x[(i, 4)] = -2.0 * b * g[0] * g[1];
        x[(i, 5)] = -2.0 * b * g[0] * g[2];
        x[(i, 6)] = -2.0 * b * g[1] * g[2];
    }
    x
}

/// Per-voxel RESTORE fit. `signal` is raw DWI signal (not log-transformed).
/// `design` is the 7-column matrix from [`build_design_matrix`], shared
/// across all voxels in a volume.
pub fn fit_voxel_restore(
    signal: &[f64],
    design: &DMatrix<f64>,
    cfg: &RestoreConfig,
) -> DtiVoxelResult {
    let n = signal.len();
    assert_eq!(design.nrows(), n);

    // Clamp signals away from zero so `ln` doesn't blow up. Voxels with
    // significant zero-signal volumes are unreliable anyway; the clamp
    // gives them a stable (non-NaN) result.
    let mut log_signal = DVector::<f64>::zeros(n);
    let mut wls_log_weights = DVector::<f64>::zeros(n); // S_i² for log-space WLS
    for i in 0..n {
        let s = signal[i].max(cfg.min_signal);
        log_signal[i] = s.ln();
        wls_log_weights[i] = s * s;
    }

    // Initial WLS fit (no GM reweighting yet — equivalent to iter 0).
    let mut weights = wls_log_weights.clone();
    let mut theta = match solve_wls(design, &log_signal, &weights) {
        Some(t) => t,
        None => return DtiVoxelResult::zero(n as u32),
    };

    // RESTORE iteration: GM reweighting on top of the WLS log weights.
    let mut prev_theta = theta.clone();
    let mut converged = false;
    let mut iters_done = 0u32;
    let mut gm_weights = vec![1.0_f64; n];
    for iter in 0..cfg.max_iter {
        iters_done = (iter + 1) as u32;

        // Residuals in log-space.
        let pred = design * &theta;
        let mut residuals = vec![0.0_f64; n];
        for i in 0..n {
            residuals[i] = log_signal[i] - pred[i];
        }

        // σ from MAD of residuals: σ̂ = 1.4826 · median(|r_i|).
        let mut abs_res: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        abs_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = abs_res[n / 2];
        let sigma = (1.4826 * mad).max(1e-10);
        let sigma_sq = sigma * sigma;

        // Geman-McClure weight per measurement.
        for i in 0..n {
            let r = residuals[i];
            let denom = sigma_sq + r * r;
            gm_weights[i] = (sigma_sq / denom).powi(2);
            weights[i] = wls_log_weights[i] * gm_weights[i];
        }

        // Re-solve WLS with the combined weights.
        theta = match solve_wls(design, &log_signal, &weights) {
            Some(t) => t,
            None => return DtiVoxelResult::zero(n as u32),
        };

        // Convergence: relative change in the 6 tensor components (skip
        // log S₀ at index 0 — its scale is unrelated to the diffusivities).
        let diff: f64 = (1..7)
            .map(|j| (theta[j] - prev_theta[j]).powi(2))
            .sum::<f64>()
            .sqrt();
        let mag: f64 = (1..7)
            .map(|j| theta[j].powi(2))
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        if diff / mag < cfg.tol {
            converged = true;
            break;
        }
        prev_theta = theta.clone();
    }

    // Build the symmetric tensor matrix from θ.
    let s0 = theta[0].exp();
    let d = Matrix3::<f64>::new(
        theta[1], theta[4], theta[5],
        theta[4], theta[2], theta[6],
        theta[5], theta[6], theta[3],
    );

    // Eigendecomposition. nalgebra's `SymmetricEigen` returns unsorted
    // eigenvalues/eigenvectors; sort them in descending order so `λ₁` is
    // the principal direction.
    let eigen = d.symmetric_eigen();
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| {
        eigen.eigenvalues[b]
            .partial_cmp(&eigen.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let evals = [
        eigen.eigenvalues[idx[0]],
        eigen.eigenvalues[idx[1]],
        eigen.eigenvalues[idx[2]],
    ];
    let v1 = eigen.eigenvectors.column(idx[0]);
    let principal_dir = [v1[0], v1[1], v1[2]];

    let md = (evals[0] + evals[1] + evals[2]) / 3.0;
    let var: f64 = evals.iter().map(|l| (l - md).powi(2)).sum();
    let mag_sq: f64 = evals.iter().map(|l| l * l).sum::<f64>().max(1e-30);
    let fa_raw = (1.5 * var / mag_sq).sqrt();
    let fa = fa_raw.clamp(0.0, 1.0);

    // Outlier count: GM weight < threshold counts as an outlier. This uses
    // the GM weight only (independent of S_i²), so the threshold is
    // interpretable as "this measurement contributed less than X% of an
    // ideal log-Gaussian sample".
    let n_outliers = gm_weights
        .iter()
        .filter(|&&w| w < cfg.outlier_threshold)
        .count() as u32;

    DtiVoxelResult {
        d,
        s0,
        fa,
        md,
        eigenvalues: evals,
        principal_dir,
        n_outliers,
        n_measurements: n as u32,
        converged,
        iterations: iters_done,
    }
}

/// Solve the weighted normal equations `(XᵀWX + εI) θ = XᵀWy` via Cholesky.
/// The `εI` term stabilises the factorisation when weights span many orders
/// of magnitude (typical for noisy clinical data — and required for the
/// RESTORE iteration where outlier weights collapse to zero). Returns
/// `None` if even the regularised system isn't PD.
fn solve_wls(
    design: &DMatrix<f64>,
    log_signal: &DVector<f64>,
    weights: &DVector<f64>,
) -> Option<DVector<f64>> {
    let n = log_signal.len();
    let p = design.ncols();
    // Form W·X without materialising W: row i of WX = w_i · row i of X.
    let mut wx = DMatrix::<f64>::zeros(n, p);
    for i in 0..n {
        let w = weights[i];
        for j in 0..p {
            wx[(i, j)] = w * design[(i, j)];
        }
    }
    let mut xt_w_x = design.transpose() * &wx;
    // Per-element diagonal regularizer: each diagonal gets a tiny fraction
    // of itself. Scales with each parameter's natural magnitude (log_S₀
    // ≪ tensor components in our normal equations), so the regularizer
    // never biases one parameter relative to another. Negligible vs
    // honest data, just enough to keep Cholesky stable when one weight
    // crashes to ~0 during RESTORE iteration.
    for i in 0..p {
        let d = xt_w_x[(i, i)];
        if d.abs() > 0.0 {
            xt_w_x[(i, i)] = d + d.abs() * 1e-12;
        } else {
            xt_w_x[(i, i)] = 1e-12;
        }
    }
    let xt_w_y = design.transpose() * &log_signal.component_mul(weights);
    let chol = Cholesky::new(xt_w_x)?;
    Some(chol.solve(&xt_w_y))
}

// ------------------ Volume driver ------------------

/// Per-volume RESTORE outputs.
pub struct DtiVolumeResult {
    /// `S₀` per voxel.
    pub s0: Array3<f32>,
    /// Fractional anisotropy per voxel.
    pub fa: Array3<f32>,
    /// Mean diffusivity per voxel (μm²/ms if input bvals are s/mm² × 1000⁻¹...
    /// in practice the units follow the bval convention — same as cs-fit).
    pub md: Array3<f32>,
    /// Outlier fraction per voxel (RESTORE's diagnostic; useful as QC).
    pub outlier_fraction: Array3<f32>,
    /// Tensor as a 4D array of 6 lower-triangular components per voxel
    /// `[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]` (BIDS-style ordering).
    pub tensor: Array4<f32>,
    /// Principal eigenvector per voxel (3 channels).
    pub principal_dir: Array4<f32>,
    /// Per-voxel iteration count (only when `compute_diagnostics=true`).
    pub iterations: Option<Array3<u32>>,
    /// Per-voxel convergence flag.
    pub converged: Option<Array3<u8>>,
}

/// Per-voxel diagnostic toggle.
#[derive(Debug, Clone, Copy, Default)]
pub struct DtiFitConfig {
    pub compute_diagnostics: bool,
}

/// Fit RESTORE to every masked voxel of `dwi`.
pub fn fit_volume_restore(
    dwi: &DwiData,
    cfg: &RestoreConfig,
    diag_cfg: DtiFitConfig,
) -> Result<DtiVolumeResult> {
    fit_volume_restore_reporting(dwi, cfg, diag_cfg, || ())
}

/// Variant that fires `on_voxel` once per completed voxel (for progress).
pub fn fit_volume_restore_reporting<F>(
    dwi: &DwiData,
    cfg: &RestoreConfig,
    diag_cfg: DtiFitConfig,
    on_voxel: F,
) -> Result<DtiVolumeResult>
where
    F: Fn() + Sync,
{
    fit_volume_restore_graddev(dwi, cfg, diag_cfg, None, on_voxel)
}

/// [`fit_volume_restore_reporting`] with an optional gradient-deviation field:
/// voxels whose `T` is not the identity get a design matrix built from their
/// effective table ([`crate::graddev`]), so the tensor — and its principal
/// direction — describes the tissue, not the scanner's deviation.
pub fn fit_volume_restore_graddev<F>(
    dwi: &DwiData,
    cfg: &RestoreConfig,
    diag_cfg: DtiFitConfig,
    graddev: Option<&crate::graddev::VoxelGradDev>,
    on_voxel: F,
) -> Result<DtiVolumeResult>
where
    F: Fn() + Sync,
{
    let s = dwi.data.shape();
    let (nx, ny, nz, nt) = (s[0], s[1], s[2], s[3]);
    if nt != dwi.gtab.bvals.len() {
        return Err(CsDmriError::Dimension(format!(
            "DTI: DWI has {} volumes but bval has {} entries",
            nt,
            dwi.gtab.bvals.len()
        )));
    }
    let design = build_design_matrix(&dwi.gtab.bvals, &dwi.gtab.bvecs);

    let results = voxel_loop::run_init(
        &dwi.mask,
        on_voxel,
        || (Vec::<f64>::with_capacity(nt), Vec::<f64>::new(), Vec::<[f64; 3]>::new()),
        |(signal_buf, bvals, dirs), x, y, z| {
            let view = dwi.data.slice(ndarray::s![x, y, z, ..]);
            signal_buf.clear();
            signal_buf.extend(view.iter().map(|&v| v as f64));
            match graddev {
                Some(gd) if !gd.is_identity(x, y, z) => {
                    gd.effective_table_into(dwi, x, y, z, bvals, dirs);
                    fit_voxel_restore(signal_buf, &build_design_matrix(bvals, dirs), cfg)
                }
                _ => fit_voxel_restore(signal_buf, &design, cfg),
            }
        },
    );

    let mut s0 = Array3::<f32>::zeros((nx, ny, nz));
    let mut fa = Array3::<f32>::zeros((nx, ny, nz));
    let mut md = Array3::<f32>::zeros((nx, ny, nz));
    let mut outlier_fraction = Array3::<f32>::zeros((nx, ny, nz));
    let mut tensor = Array4::<f32>::zeros((nx, ny, nz, 6));
    let mut principal_dir = Array4::<f32>::zeros((nx, ny, nz, 3));
    let mut iters_map = diag_cfg
        .compute_diagnostics
        .then(|| Array3::<u32>::zeros((nx, ny, nz)));
    let mut converged_map = diag_cfg
        .compute_diagnostics
        .then(|| Array3::<u8>::zeros((nx, ny, nz)));

    for ((x, y, z), v) in results {
        s0[(x, y, z)] = v.s0 as f32;
        fa[(x, y, z)] = v.fa as f32;
        md[(x, y, z)] = v.md as f32;
        outlier_fraction[(x, y, z)] = v.outlier_fraction() as f32;
        // BIDS-style 6-component lower triangular: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz.
        tensor[(x, y, z, 0)] = v.d[(0, 0)] as f32;
        tensor[(x, y, z, 1)] = v.d[(0, 1)] as f32;
        tensor[(x, y, z, 2)] = v.d[(0, 2)] as f32;
        tensor[(x, y, z, 3)] = v.d[(1, 1)] as f32;
        tensor[(x, y, z, 4)] = v.d[(1, 2)] as f32;
        tensor[(x, y, z, 5)] = v.d[(2, 2)] as f32;
        for k in 0..3 {
            principal_dir[(x, y, z, k)] = v.principal_dir[k] as f32;
        }
        if let Some(arr) = iters_map.as_mut() {
            arr[(x, y, z)] = v.iterations;
        }
        if let Some(arr) = converged_map.as_mut() {
            arr[(x, y, z)] = if v.converged { 1 } else { 0 };
        }
    }

    Ok(DtiVolumeResult {
        s0,
        fa,
        md,
        outlier_fraction,
        tensor,
        principal_dir,
        iterations: iters_map,
        converged: converged_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Synthesize a tensor signal: S(b, g) = S₀ · exp(−b · gᵀD g).
    fn forward(d: &Matrix3<f64>, s0: f64, bvals: &[f64], bvecs: &[[f64; 3]]) -> Vec<f64> {
        bvals
            .iter()
            .zip(bvecs.iter())
            .map(|(&b, g)| {
                let gv = nalgebra::Vector3::new(g[0], g[1], g[2]);
                let gtdg = gv.dot(&(d * gv));
                s0 * (-b * gtdg).exp()
            })
            .collect()
    }

    fn icosahedral_bvals_bvecs(b: f64) -> (Vec<f64>, Vec<[f64; 3]>) {
        // 12 evenly-distributed directions (vertices of an icosahedron, normalized).
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let n = (1.0 + phi * phi).sqrt();
        let raw = [
            [0.0, 1.0, phi],
            [0.0, -1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, -1.0, -phi],
            [1.0, phi, 0.0],
            [-1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, -1.0],
        ];
        let mut bvals = vec![0.0]; // one b=0
        let mut bvecs: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
        for v in &raw {
            bvals.push(b);
            bvecs.push([v[0] / n, v[1] / n, v[2] / n]);
        }
        (bvals, bvecs)
    }

    #[test]
    fn recovers_isotropic_tensor_no_noise() {
        let d_true = Matrix3::<f64>::identity() * 1e-3;
        let s0_true = 1000.0;
        let (bvals, bvecs) = icosahedral_bvals_bvecs(1000.0);
        let signal = forward(&d_true, s0_true, &bvals, &bvecs);
        let design = build_design_matrix(&bvals, &bvecs);
        let result = fit_voxel_restore(&signal, &design, &RestoreConfig::default());
        assert!(result.converged);
        assert_abs_diff_eq!(result.s0, s0_true, epsilon = 1e-3);
        assert_abs_diff_eq!(result.md, 1e-3, epsilon = 1e-9);
        assert!(result.fa < 1e-6, "isotropic FA should be ~0, got {}", result.fa);
        assert_eq!(result.n_outliers, 0);
    }

    #[test]
    fn recovers_anisotropic_tensor_no_noise() {
        let d_true = Matrix3::<f64>::new(
            1.7e-3, 0.0, 0.0,
            0.0, 0.3e-3, 0.0,
            0.0, 0.0, 0.3e-3,
        );
        let s0_true = 1500.0;
        let (bvals, bvecs) = icosahedral_bvals_bvecs(1000.0);
        let signal = forward(&d_true, s0_true, &bvals, &bvecs);
        let design = build_design_matrix(&bvals, &bvecs);
        let result = fit_voxel_restore(&signal, &design, &RestoreConfig::default());
        assert!(result.converged);
        // Expected MD = (1.7 + 0.3 + 0.3)/3 = 0.7666... × 1e-3
        assert_abs_diff_eq!(result.md, (1.7 + 0.3 + 0.3) / 3.0 * 1e-3, epsilon = 1e-9);
        // Expected FA: from λ = (1.7, 0.3, 0.3)*1e-3
        // λ̄ = 0.7667e-3
        // var = (1.7-0.7667)² + 2*(0.3-0.7667)² × 1e-6 = (0.9333² + 2*0.4667²) × 1e-6
        let lambda = [1.7e-3_f64, 0.3e-3, 0.3e-3];
        let mean = (1.7 + 0.3 + 0.3) / 3.0 * 1e-3;
        let v: f64 = lambda.iter().map(|l| (l - mean).powi(2)).sum();
        let m: f64 = lambda.iter().map(|l| l * l).sum();
        let fa_expected = (1.5 * v / m).sqrt();
        assert_abs_diff_eq!(result.fa, fa_expected, epsilon = 1e-6);
        // Principal direction along x.
        assert!(result.principal_dir[0].abs() > 0.99);
    }

    #[test]
    fn rejects_obvious_outlier_volume() {
        // Anisotropic ground truth + 1 grossly corrupted measurement.
        let d_true = Matrix3::<f64>::new(
            1.7e-3, 0.0, 0.0,
            0.0, 0.3e-3, 0.0,
            0.0, 0.0, 0.3e-3,
        );
        let s0_true = 1500.0;
        let (bvals, bvecs) = icosahedral_bvals_bvecs(1000.0);
        let mut signal = forward(&d_true, s0_true, &bvals, &bvecs);
        // Corrupt one DWI volume to a tiny value (motion-induced dropout).
        signal[5] = 5.0;
        let design = build_design_matrix(&bvals, &bvecs);
        let result = fit_voxel_restore(&signal, &design, &RestoreConfig::default());
        // RESTORE should flag the outlier and recover the true MD/FA closely.
        assert!(result.n_outliers >= 1, "expected ≥1 outlier, got {}", result.n_outliers);
        assert_abs_diff_eq!(result.md, (1.7 + 0.3 + 0.3) / 3.0 * 1e-3, epsilon = 5e-5);
        // Naive WLS (no reweighting) would have FA badly corrupted — this
        // RESTORE fit should land within 5% of the analytical FA.
        let lambda = [1.7e-3_f64, 0.3e-3, 0.3e-3];
        let mean = (1.7 + 0.3 + 0.3) / 3.0 * 1e-3;
        let v: f64 = lambda.iter().map(|l| (l - mean).powi(2)).sum();
        let m: f64 = lambda.iter().map(|l| l * l).sum();
        let fa_expected = (1.5 * v / m).sqrt();
        assert!(
            (result.fa - fa_expected).abs() < 0.05,
            "FA = {}, expected ≈ {}",
            result.fa,
            fa_expected
        );
    }
}
