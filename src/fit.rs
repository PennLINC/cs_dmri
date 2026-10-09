// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-voxel parallel fit driver.

use nalgebra::DVector;
use ndarray::Array4;

use crate::io::dwi::DwiData;
use crate::solver::{AlphaConfigurable, AlphaStrategy, Problem, Solver};
use crate::solver::tikhonov::TikhonovSolver;
use crate::voxel_loop;

/// Voxel-loop configuration.
#[derive(Debug, Clone, Copy)]
pub struct FitConfig {
    /// If true, also collect per-voxel R² and residual maps.
    pub compute_diagnostics: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            compute_diagnostics: false,
        }
    }
}

/// Per-voxel diagnostic outputs (mirroring qsirecon's r2_image / regularization_image).
///
/// `alpha` is `None` for solvers that don't have an α dial (Tikhonov); for
/// FISTA-with-α-strategy it carries the chosen α at every masked voxel.
pub struct VolumeDiagnostics {
    pub r2: ndarray::Array3<f32>,
    pub residual_l2: ndarray::Array3<f32>,
    pub iterations: ndarray::Array3<u32>,
    pub regularization_kind: ndarray::Array3<u8>,
    pub alpha: Option<ndarray::Array3<f32>>,
    /// BIC at the chosen α; `Some` only for α-strategy fits.
    pub bic: Option<ndarray::Array3<f32>>,
    /// L2 (Tikhonov) reference RSS per voxel; `Some` only for the
    /// `AlphaStrategy::PathL2Anchored` orchestration so callers can audit
    /// `RSS_L1 / RSS_L2` after the fact.
    pub rss_l2: Option<ndarray::Array3<f32>>,
}

pub struct FitResult {
    pub coefficients: Array4<f32>,
    pub diagnostics: Option<VolumeDiagnostics>,
}

/// Fit `solver` to every masked voxel of `dwi`, in parallel.
///
/// `design` is the precomputed design matrix `M` corresponding to `dwi.gtab`
/// (the same one passed to the solver's constructor). It is borrowed read-only
/// across the parallel voxel loop and is required to assemble per-voxel
/// `Problem` views.
pub fn fit_volume<S>(
    dwi: &DwiData,
    design: &nalgebra::DMatrix<f64>,
    solver: &S,
    n_coeffs: usize,
    config: FitConfig,
) -> FitResult
where
    S: Solver,
{
    fit_volume_reporting(dwi, design, solver, n_coeffs, config, || ())
}

/// Variant of [`fit_volume`] that calls `on_voxel` once per completed voxel.
///
/// Use to drive a progress heartbeat from the CLI bins. The callback runs on
/// rayon worker threads, so it must be `Sync`; in practice it's an
/// `AtomicUsize::fetch_add` inside a `Heartbeat`.
pub fn fit_volume_reporting<S, F>(
    dwi: &DwiData,
    design: &nalgebra::DMatrix<f64>,
    solver: &S,
    n_coeffs: usize,
    config: FitConfig,
    on_voxel: F,
) -> FitResult
where
    S: Solver,
    F: Fn() + Sync,
{
    let s = dwi.data.shape();
    let (nx, ny, nz, nt) = (s[0], s[1], s[2], s[3]);

    let mut coefficients = Array4::<f32>::zeros((nx, ny, nz, n_coeffs));
    let mut r2 = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));
    let mut resid = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));
    let mut iters = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<u32>::zeros((nx, ny, nz)));
    let mut regk = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<u8>::zeros((nx, ny, nz)));

    struct VoxelFit {
        coeffs: DVector<f64>,
        r2: f32,
        residual_l2: f32,
        iterations: u32,
        regularization_kind: u8,
    }

    let results = voxel_loop::run_init(
        &dwi.mask,
        on_voxel,
        || DVector::<f64>::zeros(nt),
        |signal, x, y, z| {
            let signal_view = dwi.data.slice(ndarray::s![x, y, z, ..]);
            for (dst, &src) in signal.iter_mut().zip(signal_view.iter()) {
                *dst = src as f64;
            }
            let mean = signal.mean();
            let problem = Problem {
                design,
                signal: &*signal,
            };
            let (coef, diag) = solver.fit(&problem);
            let ss_res = diag.residual_l2 * diag.residual_l2;
            let ss_tot: f64 = signal.iter().map(|s| (s - mean).powi(2)).sum();
            let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
            VoxelFit {
                coeffs: coef,
                r2: r2 as f32,
                residual_l2: diag.residual_l2 as f32,
                iterations: diag.iterations,
                regularization_kind: diag.regularization_kind,
            }
        },
    );

    for ((x, y, z), v) in results {
        for k in 0..n_coeffs {
            coefficients[(x, y, z, k)] = v.coeffs[k] as f32;
        }
        if let Some(arr) = r2.as_mut() {
            arr[(x, y, z)] = v.r2;
        }
        if let Some(arr) = resid.as_mut() {
            arr[(x, y, z)] = v.residual_l2;
        }
        if let Some(arr) = iters.as_mut() {
            arr[(x, y, z)] = v.iterations;
        }
        if let Some(arr) = regk.as_mut() {
            arr[(x, y, z)] = v.regularization_kind;
        }
    }

    let diagnostics = if config.compute_diagnostics {
        Some(VolumeDiagnostics {
            r2: r2.unwrap(),
            residual_l2: resid.unwrap(),
            iterations: iters.unwrap(),
            regularization_kind: regk.unwrap(),
            alpha: None,
            bic: None,
            rss_l2: None,
        })
    } else {
        None
    };

    FitResult {
        coefficients,
        diagnostics,
    }
}

/// Variant of [`fit_volume`] that picks α per voxel via an [`AlphaStrategy`].
///
/// `base_solver` is cloned once per rayon worker thread and mutated in place
/// (`set_alpha`) inside the per-voxel closure. `Send + Sync + Clone` on the
/// solver lets us avoid contention on a shared mutable solver.
///
/// Returns the chosen α per voxel (in the diagnostics struct, when
/// `config.compute_diagnostics` is true) plus aggregate stats describing the
/// distribution of αs across the mask.
pub struct AlphaStrategyFit {
    pub result: FitResult,
    /// `None` for `AlphaStrategy::Fixed` (one global α was used);
    /// `Some((median, p10, p90))` otherwise.
    pub alpha_distribution: Option<(f64, f64, f64)>,
    /// The α actually applied — single value for Fixed, otherwise the median.
    pub representative_alpha: f64,
}

pub fn fit_volume_with_alpha_strategy<S>(
    dwi: &DwiData,
    design: &nalgebra::DMatrix<f64>,
    base_solver: &S,
    strategy: &AlphaStrategy,
    n_coeffs: usize,
    config: FitConfig,
) -> AlphaStrategyFit
where
    S: AlphaConfigurable + Send + Sync,
{
    fit_volume_with_alpha_strategy_reporting(
        dwi,
        design,
        base_solver,
        strategy,
        n_coeffs,
        config,
        || (),
    )
}

/// Variant of [`fit_volume_with_alpha_strategy`] that calls `on_voxel` once
/// per completed voxel, for progress reporting.
pub fn fit_volume_with_alpha_strategy_reporting<S, F>(
    dwi: &DwiData,
    design: &nalgebra::DMatrix<f64>,
    base_solver: &S,
    strategy: &AlphaStrategy,
    n_coeffs: usize,
    config: FitConfig,
    on_voxel: F,
) -> AlphaStrategyFit
where
    S: AlphaConfigurable + Send + Sync,
    F: Fn() + Sync,
{
    let s = dwi.data.shape();
    let (nx, ny, nz, nt) = (s[0], s[1], s[2], s[3]);

    let mut coefficients = Array4::<f32>::zeros((nx, ny, nz, n_coeffs));
    let mut r2 = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));
    let mut resid = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));
    let mut iters = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<u32>::zeros((nx, ny, nz)));
    let mut regk = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<u8>::zeros((nx, ny, nz)));
    let mut alpha_map = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));
    let mut bic_map = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));

    struct VoxelFit {
        coeffs: DVector<f64>,
        r2: f32,
        residual_l2: f32,
        iterations: u32,
        alpha: f64,
        bic: f64,
    }

    let results = voxel_loop::run_init(
        &dwi.mask,
        on_voxel,
        || (base_solver.clone(), DVector::<f64>::zeros(nt)),
        |(solver, signal), x, y, z| {
            let signal_view = dwi.data.slice(ndarray::s![x, y, z, ..]);
            for (dst, &src) in signal.iter_mut().zip(signal_view.iter()) {
                *dst = src as f64;
            }
            let mean = signal.mean();
            let problem = Problem {
                design,
                signal: &*signal,
            };
            let res = strategy.resolve(solver, &problem);
            let ss_res = res.residual_l2 * res.residual_l2;
            let ss_tot: f64 = signal.iter().map(|s| (s - mean).powi(2)).sum();
            let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
            VoxelFit {
                coeffs: res.coef,
                r2: r2 as f32,
                residual_l2: res.residual_l2 as f32,
                iterations: res.iterations,
                alpha: res.alpha,
                bic: res.bic,
            }
        },
    );

    let mut alphas_seen: Vec<f64> = Vec::with_capacity(results.len());
    for ((x, y, z), v) in &results {
        let (x, y, z) = (*x, *y, *z);
        for k in 0..n_coeffs {
            coefficients[(x, y, z, k)] = v.coeffs[k] as f32;
        }
        if let Some(arr) = r2.as_mut() {
            arr[(x, y, z)] = v.r2;
        }
        if let Some(arr) = resid.as_mut() {
            arr[(x, y, z)] = v.residual_l2;
        }
        if let Some(arr) = iters.as_mut() {
            arr[(x, y, z)] = v.iterations;
        }
        if let Some(arr) = regk.as_mut() {
            arr[(x, y, z)] = 1; // FISTA = L1 = 1 (mirrors fit_volume)
        }
        if let Some(arr) = alpha_map.as_mut() {
            arr[(x, y, z)] = v.alpha as f32;
        }
        if let Some(arr) = bic_map.as_mut() {
            arr[(x, y, z)] = v.bic as f32;
        }
        alphas_seen.push(v.alpha);
    }

    let diagnostics = if config.compute_diagnostics {
        Some(VolumeDiagnostics {
            r2: r2.unwrap(),
            residual_l2: resid.unwrap(),
            iterations: iters.unwrap(),
            regularization_kind: regk.unwrap(),
            alpha: alpha_map,
            bic: bic_map,
            rss_l2: None,
        })
    } else {
        None
    };

    let (alpha_distribution, representative_alpha) = match strategy {
        AlphaStrategy::Fixed { alpha } => (None, *alpha),
        _ => {
            let mut sorted = alphas_seen.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted.is_empty() {
                (Some((0.0, 0.0, 0.0)), 0.0)
            } else {
                let pct = |q: f64| -> f64 {
                    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
                    sorted[i.min(sorted.len() - 1)]
                };
                let median = pct(0.5);
                let p10 = pct(0.10);
                let p90 = pct(0.90);
                (Some((median, p10, p90)), median)
            }
        }
    };

    AlphaStrategyFit {
        result: FitResult {
            coefficients,
            diagnostics,
        },
        alpha_distribution,
        representative_alpha,
    }
}

/// Variant of [`fit_volume_with_alpha_strategy_reporting`] for the
/// `AlphaStrategy::PathL2Anchored` mode. Uses a shared `TikhonovSolver`
/// reference (read-only across rayon workers, no per-worker clones) for
/// the per-voxel L2 reference fit. Falls back to standard reporting for
/// other strategy variants.
///
/// Adds an `rss_l2` diagnostic NIfTI when `config.compute_diagnostics` is
/// set so callers can audit the chosen-α RSS against the L2 reference.
pub fn fit_volume_with_alpha_strategy_l2_anchored_reporting<S, F>(
    dwi: &DwiData,
    design: &nalgebra::DMatrix<f64>,
    base_solver: &S,
    l2_solver: &TikhonovSolver,
    strategy: &AlphaStrategy,
    n_coeffs: usize,
    config: FitConfig,
    on_voxel: F,
) -> AlphaStrategyFit
where
    S: AlphaConfigurable + Send + Sync,
    F: Fn() + Sync,
{
    let s = dwi.data.shape();
    let (nx, ny, nz, nt) = (s[0], s[1], s[2], s[3]);

    let mut coefficients = Array4::<f32>::zeros((nx, ny, nz, n_coeffs));
    let mut r2 = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));
    let mut resid = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));
    let mut iters = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<u32>::zeros((nx, ny, nz)));
    let mut regk = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<u8>::zeros((nx, ny, nz)));
    let mut alpha_map = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));
    let mut bic_map = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));
    let mut rss_l2_map = config
        .compute_diagnostics
        .then(|| ndarray::Array3::<f32>::zeros((nx, ny, nz)));

    struct VoxelFit {
        coeffs: DVector<f64>,
        r2: f32,
        residual_l2: f32,
        iterations: u32,
        alpha: f64,
        bic: f64,
        rss_l2: f32,
    }

    let results = voxel_loop::run_init(
        &dwi.mask,
        on_voxel,
        || (base_solver.clone(), DVector::<f64>::zeros(nt)),
        |(solver, signal), x, y, z| {
            let signal_view = dwi.data.slice(ndarray::s![x, y, z, ..]);
            for (dst, &src) in signal.iter_mut().zip(signal_view.iter()) {
                *dst = src as f64;
            }
            let mean = signal.mean();
            let problem = Problem {
                design,
                signal: &*signal,
            };
            // L2 reference RSS (closed-form, cached Cholesky in l2_solver).
            let (_, l2_diag) = l2_solver.fit(&problem);
            let rss_l2 = l2_diag.residual_l2 * l2_diag.residual_l2;
            let res = strategy.resolve_with_l2_ref(solver, l2_solver, &problem);
            let ss_res = res.residual_l2 * res.residual_l2;
            let ss_tot: f64 = signal.iter().map(|s| (s - mean).powi(2)).sum();
            let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
            VoxelFit {
                coeffs: res.coef,
                r2: r2 as f32,
                residual_l2: res.residual_l2 as f32,
                iterations: res.iterations,
                alpha: res.alpha,
                bic: res.bic,
                rss_l2: rss_l2 as f32,
            }
        },
    );

    let mut alphas_seen: Vec<f64> = Vec::with_capacity(results.len());
    for ((x, y, z), v) in &results {
        let (x, y, z) = (*x, *y, *z);
        for k in 0..n_coeffs {
            coefficients[(x, y, z, k)] = v.coeffs[k] as f32;
        }
        if let Some(arr) = r2.as_mut() {
            arr[(x, y, z)] = v.r2;
        }
        if let Some(arr) = resid.as_mut() {
            arr[(x, y, z)] = v.residual_l2;
        }
        if let Some(arr) = iters.as_mut() {
            arr[(x, y, z)] = v.iterations;
        }
        if let Some(arr) = regk.as_mut() {
            arr[(x, y, z)] = 1; // FISTA = L1 = 1
        }
        if let Some(arr) = alpha_map.as_mut() {
            arr[(x, y, z)] = v.alpha as f32;
        }
        if let Some(arr) = bic_map.as_mut() {
            arr[(x, y, z)] = v.bic as f32;
        }
        if let Some(arr) = rss_l2_map.as_mut() {
            arr[(x, y, z)] = v.rss_l2;
        }
        alphas_seen.push(v.alpha);
    }

    let diagnostics = if config.compute_diagnostics {
        Some(VolumeDiagnostics {
            r2: r2.unwrap(),
            residual_l2: resid.unwrap(),
            iterations: iters.unwrap(),
            regularization_kind: regk.unwrap(),
            alpha: alpha_map,
            bic: bic_map,
            rss_l2: rss_l2_map,
        })
    } else {
        None
    };

    let (alpha_distribution, representative_alpha) = match strategy {
        AlphaStrategy::Fixed { alpha } => (None, *alpha),
        _ => {
            let mut sorted = alphas_seen.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted.is_empty() {
                (Some((0.0, 0.0, 0.0)), 0.0)
            } else {
                let pct = |q: f64| -> f64 {
                    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
                    sorted[i.min(sorted.len() - 1)]
                };
                let median = pct(0.5);
                let p10 = pct(0.10);
                let p90 = pct(0.90);
                (Some((median, p10, p90)), median)
            }
        }
    };

    AlphaStrategyFit {
        result: FitResult {
            coefficients,
            diagnostics,
        },
        alpha_distribution,
        representative_alpha,
    }
}
