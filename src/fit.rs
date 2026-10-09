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

// ------------------------------------------------------------------------
// Whole-volume SHORE fit: the solver dispatch `cs-fit` exposes, as a library
// call so the CLI and the Python bindings share one implementation.
// ------------------------------------------------------------------------

use crate::basis::Basis;
use crate::basis::shore::ShoreBasis;
use crate::io::coeffs::{ChosenAlpha, SolverMetadata};
use crate::solver::AlphaPath;
use crate::solver::IclsConfig;
use crate::solver::fista::FistaSolver;
use crate::solver::shore_icls::ShoreIclsSolver;
use crate::{CsDmriError, Result};

/// How an L1 fit picks its per-voxel α. Mirrors `cs-fit --alpha-mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Fixed,
    AlphaRatio,
    PathBic,
    L2Anchored,
}

/// Validated [`AlphaStrategy`] for `mode`. `path_eps = None` takes the
/// mode-dependent default: 1e-3 for path-BIC (backward compatible) and 1e-4
/// for L2-anchored (lets the slack constraint bind on the whole path).
pub fn build_alpha_strategy(
    mode: AlphaMode,
    alpha: f64,
    alpha_ratio: f64,
    path_n_alphas: usize,
    path_eps: Option<f64>,
    slack: f64,
) -> Result<AlphaStrategy> {
    let bad = CsDmriError::Other;
    let path = |default_eps: f64| -> Result<AlphaPath> {
        if path_n_alphas < 2 {
            return Err(bad(format!("--path-n-alphas must be ≥ 2, got {path_n_alphas}")));
        }
        let eps = path_eps.unwrap_or(default_eps);
        if !(eps > 0.0 && eps < 1.0) {
            return Err(bad(format!("--path-eps must lie in (0, 1), got {eps}")));
        }
        Ok(AlphaPath { n: path_n_alphas, eps })
    };
    Ok(match mode {
        AlphaMode::Fixed => AlphaStrategy::Fixed { alpha },
        AlphaMode::AlphaRatio => {
            if !(alpha_ratio > 0.0 && alpha_ratio < 1.0) {
                return Err(bad(format!("--alpha-ratio must lie in (0, 1), got {alpha_ratio}")));
            }
            AlphaStrategy::AlphaMaxRatio { ratio: alpha_ratio }
        }
        AlphaMode::PathBic => AlphaStrategy::PathBic { path: path(1e-3)? },
        AlphaMode::L2Anchored => {
            let path = path(1e-4)?;
            if slack < 0.0 {
                return Err(bad(format!("--slack must be ≥ 0, got {slack}")));
            }
            AlphaStrategy::PathL2Anchored { path, slack }
        }
    })
}

/// Regularization for [`fit_shore`].
#[derive(Debug, Clone)]
pub enum ShoreRegularization {
    /// FISTA L1 with a per-voxel (or fixed) α strategy. `seed_alpha` only
    /// feeds the α-independent Lipschitz cache; the strategy overwrites it.
    L1 {
        strategy: AlphaStrategy,
        max_iter: u32,
        tol: f64,
        non_negative: bool,
        seed_alpha: f64,
    },
    /// Closed-form Tikhonov.
    L2,
    /// Goldfarb-Idnani ICLS with non-negative ODF amplitudes on a dense sphere,
    /// at `lmax = default_lmax(radial_order)`.
    AmpNonNeg { icls: IclsConfig },
}

/// Everything [`fit_shore`] needs besides the data.
#[derive(Debug, Clone)]
pub struct ShoreFitSpec {
    pub radial_order: u32,
    pub zeta: f64,
    pub regularization: ShoreRegularization,
    /// Tikhonov weights for `L2`, and for the L2 reference of `PathL2Anchored`.
    pub lambda_n: f64,
    pub lambda_l: f64,
    pub compute_diagnostics: bool,
}

impl Default for ShoreFitSpec {
    /// `cs-fit` defaults: radial order 6, ζ = 700, L1 with L2-anchored α.
    fn default() -> Self {
        Self {
            radial_order: 6,
            zeta: 700.0,
            regularization: ShoreRegularization::L1 {
                strategy: AlphaStrategy::PathL2Anchored {
                    path: AlphaPath { n: 20, eps: 1e-4 },
                    slack: 0.05,
                },
                max_iter: 1000,
                tol: 1e-6,
                non_negative: false,
                seed_alpha: 1.0,
            },
            lambda_n: 1e-8,
            lambda_l: 1e-8,
            compute_diagnostics: false,
        }
    }
}

/// Output of [`fit_shore`].
pub struct ShoreFitOutput {
    pub basis: ShoreBasis,
    pub result: FitResult,
    /// Solver description for the coefficient sidecar.
    pub solver: SolverMetadata,
    /// `(median, p10, p90)` of the per-voxel α, for per-voxel α strategies.
    pub alpha_distribution: Option<(f64, f64, f64)>,
}

/// Fit a SHORE basis to every masked voxel of `dwi` with the solver `spec`
/// selects. `on_voxel` runs once per fitted voxel, on rayon workers.
pub fn fit_shore<F>(dwi: &DwiData, spec: &ShoreFitSpec, on_voxel: F) -> Result<ShoreFitOutput>
where
    F: Fn() + Sync,
{
    let basis = ShoreBasis::new(spec.radial_order, spec.zeta);
    let design = basis.design_matrix(&dwi.gtab);
    let regularization = basis.regularization();
    let cfg = FitConfig {
        compute_diagnostics: spec.compute_diagnostics,
    };
    let n_coeffs = basis.n_coeffs();
    let tikhonov =
        || TikhonovSolver::new(design.clone(), &regularization, spec.lambda_n, spec.lambda_l);

    let (result, solver, alpha_distribution) = match &spec.regularization {
        ShoreRegularization::L1 {
            strategy,
            max_iter,
            tol,
            non_negative,
            seed_alpha,
        } => {
            let base =
                FistaSolver::new(design.clone(), seed_alpha.max(1e-12), *max_iter, *tol, *non_negative);
            let fit = if matches!(strategy, AlphaStrategy::PathL2Anchored { .. }) {
                fit_volume_with_alpha_strategy_l2_anchored_reporting(
                    dwi, &design, &base, &tikhonov(), strategy, n_coeffs, cfg, on_voxel,
                )
            } else {
                fit_volume_with_alpha_strategy_reporting(
                    dwi, &design, &base, strategy, n_coeffs, cfg, on_voxel,
                )
            };
            let chosen_alpha = match (strategy, fit.alpha_distribution) {
                (AlphaStrategy::Fixed { alpha }, _) => ChosenAlpha::Global { alpha: *alpha },
                (_, Some((median, p10, p90))) => ChosenAlpha::PerVoxel { median, p10, p90 },
                (_, None) => ChosenAlpha::Global {
                    alpha: fit.representative_alpha,
                },
            };
            let meta = SolverMetadata::Fista {
                alpha_strategy: strategy.clone(),
                chosen_alpha,
                non_negative: *non_negative,
                max_iter: *max_iter,
                tol: *tol,
            };
            (fit.result, meta, fit.alpha_distribution)
        }
        ShoreRegularization::L2 => {
            let r = fit_volume_reporting(dwi, &design, &tikhonov(), n_coeffs, cfg, on_voxel);
            let meta = SolverMetadata::Tikhonov {
                lambda_n: spec.lambda_n,
                lambda_l: spec.lambda_l,
            };
            (r, meta, None)
        }
        ShoreRegularization::AmpNonNeg { icls } => {
            let lmax = crate::odf::default_lmax(spec.radial_order);
            let solver = ShoreIclsSolver::new(design.clone(), &basis, lmax, *icls);
            let r = fit_volume_reporting(dwi, &design, &solver, n_coeffs, cfg, on_voxel);
            let meta = SolverMetadata::ShoreIcls {
                lmax,
                n_constraint_dirs: solver.n_constraint_dirs(),
                max_iter: icls.max_iter,
                tol: icls.tol,
                epsilon: icls.epsilon,
            };
            (r, meta, None)
        }
    };
    Ok(ShoreFitOutput {
        basis,
        result,
        solver,
        alpha_distribution,
    })
}

/// Per-voxel sparsity: fraction of strictly-nonzero coefficients. FISTA's
/// soft-threshold returns exact zeros, so no tolerance is needed; for L2 this
/// is essentially 1.0 everywhere.
pub fn sparsity_map(coeffs: &Array4<f32>, mask: &ndarray::Array3<bool>) -> ndarray::Array3<f32> {
    let s = coeffs.shape();
    let (nx, ny, nz, nk) = (s[0], s[1], s[2], s[3]);
    let denom = nk as f32;
    let mut out = ndarray::Array3::<f32>::zeros((nx, ny, nz));
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if !mask[(x, y, z)] {
                    continue;
                }
                let nnz = (0..nk).filter(|&k| coeffs[(x, y, z, k)] != 0.0).count();
                out[(x, y, z)] = nnz as f32 / denom;
            }
        }
    }
    out
}

/// RMSE = ‖Mc − s‖₂ / √n_grads, from the per-voxel residual L2 norm.
pub fn rmse_from_residual_l2(residual_l2: &ndarray::Array3<f32>, n_grads: usize) -> ndarray::Array3<f32> {
    let denom = (n_grads as f32).sqrt();
    residual_l2.mapv(|r| r / denom)
}

/// Mean of `arr` over `mask`, 0 for an empty mask.
pub fn mean_in_mask(arr: &ndarray::Array3<f32>, mask: &ndarray::Array3<bool>) -> f64 {
    let (sum, count) = arr
        .iter()
        .zip(mask.iter())
        .filter(|(_, m)| **m)
        .fold((0.0_f64, 0_usize), |(s, c), (v, _)| (s + *v as f64, c + 1));
    if count == 0 { 0.0 } else { sum / count as f64 }
}
