// SPDX-License-Identifier: MIT OR Apache-2.0
//! SS3T-CSD per-voxel orchestration.
//!
//! Implements the alternating two-tissue fixed-point loop of
//! Dhollander & Connelly (ISMRM 2016, abstract 3010), reproducing the
//! algorithm of MRtrix3Tissue's `bin/ss3t_csd_beta1` Python wrapper without
//! depending on the binary.
//!
//! Volume-invariant work (response scaling, augmented-signal layout, the
//! three CSD solvers, per-shell predictors) lives in [`Ss3tPlan`]; the
//! per-voxel routine [`fit_voxel`] is the inner kernel called from the rayon
//! voxel loop.

use nalgebra::{DMatrix, DVector};

use crate::multitissue::forward::{PerShellPredictor, ShellPlan, TissueSlot, build_h};
use crate::multitissue::response::{ResponseError, TissueResponse};
use crate::qspace::{GradientTable, Shell};
use crate::sh::{ncoeffs_for_lmax, sh2amp_cart};
use crate::solver::icls::{IclsConfig, IclsSolver, IclsWorkspace};
use crate::{CsDmriError, Result};

/// How to choose the WM SH order for each voxel.
#[derive(Debug, Clone, PartialEq)]
pub enum LmaxWmStrategy {
    /// Use a single fixed lmax for every voxel (matches qsirecon's default
    /// of 8). Fastest; one inner solve sweep per voxel.
    Fixed(usize),
    /// Per-voxel: fit at every lmax in `candidates`, pick the lmax that
    /// minimises BIC = `n·log(rss/n) + k·log(n)` where k is the count of
    /// fitted coefficients. Mirrors `cs-fit --alpha-mode path-bic` but for
    /// SH order rather than L1 weight. CSF/GM voxels auto-select lmax=0
    /// (cheap; trivial inner CSD); only fiber-rich WM benefits from lmax=8.
    PathBic(Vec<usize>),
}

impl LmaxWmStrategy {
    /// Candidates to consider: 1 entry for `Fixed`, all entries for `PathBic`.
    pub fn candidates(&self) -> Vec<usize> {
        match self {
            Self::Fixed(l) => vec![*l],
            Self::PathBic(c) => c.clone(),
        }
    }

    /// Largest candidate — the buffer width for output WM coefficient arrays.
    pub fn max_lmax(&self) -> usize {
        match self {
            Self::Fixed(l) => *l,
            Self::PathBic(c) => c.iter().copied().max().unwrap_or(0),
        }
    }
}

impl Default for LmaxWmStrategy {
    fn default() -> Self {
        Self::Fixed(8)
    }
}

/// Configuration for the SS3T algorithm.
#[derive(Debug, Clone)]
pub struct Ss3tConfig {
    /// Number of outer iterations (default 3, must be ≥ 2).
    pub niter: u32,
    /// b=0 contribution as a percentage of the non-b=0 volumes (default 10).
    pub bzero_pct: f64,
    /// WM SH order strategy (default `Fixed(8)` — qsirecon parity).
    pub lmax_wm: LmaxWmStrategy,
    /// Inner ICLS solver configuration. The same config is shared by all
    /// three fits (GM+CSF init, GM+WM, GM+CSF refit).
    pub icls: IclsConfig,
}

impl Default for Ss3tConfig {
    fn default() -> Self {
        Self {
            niter: 3,
            bzero_pct: 10.0,
            lmax_wm: LmaxWmStrategy::default(),
            icls: IclsConfig::default(),
        }
    }
}

/// The three response functions consumed by SS3T.
#[derive(Debug, Clone)]
pub struct Ss3tResponses {
    pub wm: TissueResponse,
    pub gm: TissueResponse,
    pub csf: TissueResponse,
}

impl Ss3tResponses {
    /// Validate shapes (GM/CSF isotropic; WM b=0 row isotropic; all three
    /// have exactly two shells matching the SS3T single-shell+b0 contract).
    pub fn validate(&self) -> std::result::Result<(), ResponseError> {
        self.gm.require_isotropic("gm response")?;
        self.csf.require_isotropic("csf response")?;
        self.wm.require_b0_isotropic("wm response")?;
        Ok(())
    }
}

/// Per-voxel diagnostics returned alongside the recovered FODs.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ss3tVoxelDiagnostics {
    /// Sum of inner CSD iterations across all (3·niter + 1) inner solves.
    pub total_inner_iter: usize,
    /// True iff every inner CSD call converged.
    pub all_converged: bool,
    /// L2 norm of the final residual `‖wdwi − pred_total‖₂` (in augmented units).
    pub residual_l2: f64,
}

/// Volume-invariant precompute used by every per-voxel SS3T fit.
///
/// Building the plan does the bulk of the per-volume work: response scaling,
/// shell partitioning, SH-evaluation matrices on each shell and on the
/// non-negativity sphere, three Cholesky factorisations (one per inner CSD
/// problem), and predictor blocks for forward subtraction.
pub struct Ss3tPlan {
    /// Number of augmented signal rows (== n_grads — we don't reorder, only weight).
    pub n_aug: usize,
    /// Effective WM lmax after clamping to the response file.
    pub lmax_wm: usize,
    /// Number of WM SH coefficients (= ncoeffs_for_lmax(lmax_wm)).
    pub n_sh_wm: usize,
    /// b=0 weighting factor `w = sqrt(n_dwi · bzero_pct / (n_b0 · 100))`.
    pub bzero_sw: f64,
    /// Per-augmented-row signal weight (`bzero_sw` for b=0 rows, `1.0` otherwise).
    pub signal_weight: Vec<f64>,
    /// ICLS solver for `[c_gm, c_csf]` fits (init + residual2 refit).
    pub icls_gm_csf: IclsSolver,
    /// ICLS solver for `[c_gm, c_wm…]` fit (residual1).
    pub icls_gm_wm: IclsSolver,
    /// Predictor: maps `c_csf` to per-row CSF signal contribution.
    pub pred_csf: PerShellPredictor,
    /// Predictor: maps `c_wm` to per-row WM signal contribution.
    pub pred_wm: PerShellPredictor,
    /// Predictor: maps `c_gm` to per-row GM signal contribution. (Built so the
    /// final-residual diagnostic can include all three tissues.)
    pub pred_gm: PerShellPredictor,
    /// Outer-iteration count carried from the config.
    pub niter: u32,
}

impl Ss3tPlan {
    /// Build the plan from the gradient table + responses + config. Used
    /// when `cfg.lmax_wm` is `Fixed`. For `PathBic`, build an
    /// [`Ss3tVolumePlan`] instead (one `Ss3tPlan` per candidate lmax).
    pub fn build(
        gtab: &GradientTable,
        responses: &Ss3tResponses,
        cfg: &Ss3tConfig,
    ) -> Result<Self> {
        let lmax_wm = match &cfg.lmax_wm {
            LmaxWmStrategy::Fixed(l) => *l,
            LmaxWmStrategy::PathBic(_) => {
                return Err(CsDmriError::Other(
                    "ss3t: Ss3tPlan::build expects Fixed lmax strategy; use Ss3tVolumePlan::build for PathBic"
                        .into(),
                ));
            }
        };
        Self::build_for_lmax(gtab, responses, lmax_wm, cfg)
    }

    /// Build a plan for a specific WM SH order. Used directly by
    /// [`Ss3tVolumePlan::build`] when iterating over path-BIC candidates;
    /// `cfg.lmax_wm` is ignored in favor of the explicit `lmax_wm` argument.
    pub fn build_for_lmax(
        gtab: &GradientTable,
        responses: &Ss3tResponses,
        lmax_wm: usize,
        cfg: &Ss3tConfig,
    ) -> Result<Self> {
        if cfg.niter < 2 {
            return Err(CsDmriError::Other(format!(
                "ss3t: niter must be ≥ 2, got {}",
                cfg.niter
            )));
        }
        if cfg.bzero_pct <= 0.0 {
            return Err(CsDmriError::Other(format!(
                "ss3t: bzero_pct must be > 0, got {}",
                cfg.bzero_pct
            )));
        }
        responses.validate().map_err(CsDmriError::from)?;

        let shells = gtab.shells(50.0);
        if shells.len() != 2 || shells[0].b > gtab.b0_threshold || shells[1].b <= gtab.b0_threshold
        {
            return Err(CsDmriError::Other(format!(
                "ss3t: need exactly one b=0 shell and one DWI shell, got shells with mean b-values {:?}",
                shells.iter().map(|s| s.b).collect::<Vec<_>>()
            )));
        }
        let b0_shell = &shells[0];
        let dwi_shell = &shells[1];
        let n_b0 = b0_shell.indices.len();
        let n_dwi = dwi_shell.indices.len();
        if n_b0 == 0 || n_dwi == 0 {
            return Err(CsDmriError::Other(
                "ss3t: both b=0 and DWI shells must be non-empty".into(),
            ));
        }
        if responses.wm.n_shells() != 2
            || responses.gm.n_shells() != 2
            || responses.csf.n_shells() != 2
        {
            return Err(CsDmriError::Other(
                "ss3t: each response file must have exactly 2 shells (b=0 and a single DWI shell)".into(),
            ));
        }

        let bzero_sw = ((n_dwi as f64) * cfg.bzero_pct / ((n_b0 as f64) * 100.0)).sqrt();

        let requested_lmax = lmax_wm;
        let lmax_wm = lmax_wm.min(responses.wm.lmax);
        if lmax_wm % 2 != 0 {
            return Err(CsDmriError::Other(format!(
                "ss3t: lmax_wm must be even, got {}",
                lmax_wm
            )));
        }
        if lmax_wm < requested_lmax {
            eprintln!(
                "[ss3t] WM response file declares lmax={}, requested lmax_wm={} — clamping to {}.",
                responses.wm.lmax, requested_lmax, lmax_wm
            );
        }
        let n_sh_wm = ncoeffs_for_lmax(lmax_wm);

        // Scale row-0 of every response by bzero_sw. The b=0 row of each
        // response is, by validation, isotropic (only r_0 nonzero), so this
        // amounts to one scalar multiply per tissue.
        let scaled = scale_b0(&responses, bzero_sw);

        // Augmented signal weight per input row.
        let n_aug = gtab.n_grads();
        let mut signal_weight = vec![1.0_f64; n_aug];
        for &i in &b0_shell.indices {
            signal_weight[i] = bzero_sw;
        }

        // Per-shell forward plans (use the original gradient indices as
        // augmented row indices — we don't reorder).
        let shell_plans = build_shell_plans(gtab, &shells)?;

        // ICLS solver for GM+CSF fits (lmax = (0, 0); 2 coefficients total).
        let h_gm_csf = build_h(
            n_aug,
            &shell_plans,
            &[
                TissueSlot { response: &scaled.gm, lmax: 0 },
                TissueSlot { response: &scaled.csf, lmax: 0 },
            ],
        );
        let constraint_gm_csf = DMatrix::<f64>::identity(2, 2);
        let icls_gm_csf = IclsSolver::new(h_gm_csf.h, constraint_gm_csf, cfg.icls);

        // ICLS solver for GM+WM fit (lmax = (0, lmax_wm); 1 + n_sh_wm coefficients).
        let h_gm_wm = build_h(
            n_aug,
            &shell_plans,
            &[
                TissueSlot { response: &scaled.gm, lmax: 0 },
                TissueSlot { response: &scaled.wm, lmax: lmax_wm },
            ],
        );
        let constraint_gm_wm = build_gm_wm_constraint(lmax_wm, n_sh_wm);
        let icls_gm_wm = IclsSolver::new(h_gm_wm.h, constraint_gm_wm, cfg.icls);

        // Predictors used for residual subtraction.
        let pred_csf = PerShellPredictor::new(
            n_aug,
            &shell_plans,
            &TissueSlot { response: &scaled.csf, lmax: 0 },
        );
        let pred_wm = PerShellPredictor::new(
            n_aug,
            &shell_plans,
            &TissueSlot { response: &scaled.wm, lmax: lmax_wm },
        );
        let pred_gm = PerShellPredictor::new(
            n_aug,
            &shell_plans,
            &TissueSlot { response: &scaled.gm, lmax: 0 },
        );

        Ok(Self {
            n_aug,
            lmax_wm,
            n_sh_wm,
            bzero_sw,
            signal_weight,
            icls_gm_csf,
            icls_gm_wm,
            pred_csf,
            pred_wm,
            pred_gm,
            niter: cfg.niter,
        })
    }
}

/// Per-voxel scratch buffers for the SS3T inner loop.
///
/// Bundles the two ICLS workspaces (one for GM+CSF, one for GM+WM) plus the
/// reusable augmented-signal vector and per-iteration residual. Build once
/// per rayon worker thread via [`Ss3tPlan::workspace`] and feed into
/// [`fit_voxel_into`] for every voxel that worker handles.
pub struct Ss3tVoxelWorkspace {
    pub icls_gm_csf: IclsWorkspace,
    pub icls_gm_wm: IclsWorkspace,
    pub wdwi: DVector<f64>,
    pub residual: DVector<f64>,
}

impl Ss3tPlan {
    /// Allocate a per-worker workspace sized for this plan.
    pub fn workspace(&self) -> Ss3tVoxelWorkspace {
        Ss3tVoxelWorkspace {
            icls_gm_csf: self.icls_gm_csf.workspace(),
            icls_gm_wm: self.icls_gm_wm.workspace(),
            wdwi: DVector::<f64>::zeros(self.n_aug),
            residual: DVector::<f64>::zeros(self.n_aug),
        }
    }
}

/// Per-voxel SS3T fit. `signal` length must equal `plan.n_aug` (== n_grads).
///
/// Returns `(c_wm, c_gm, c_csf, diagnostics)`. The WM coefficient vector is in
/// MRtrix-basis ordering at `plan.lmax_wm`; the GM and CSF outputs are scalar
/// l=0 coefficients (with the `bzero_sw` scaling already absorbed into the
/// effective response, so the returned values are directly comparable to
/// MRtrix3Tissue's `out_GM` / `out_CSF`).
///
/// Convenience wrapper around [`fit_voxel_into`] that allocates a fresh
/// workspace on every call. Inside hot loops, build a workspace once via
/// [`Ss3tPlan::workspace`] and call `fit_voxel_into` repeatedly.
pub fn fit_voxel(
    signal: &[f64],
    plan: &Ss3tPlan,
) -> (DVector<f64>, f64, f64, Ss3tVoxelDiagnostics) {
    let mut ws = plan.workspace();
    fit_voxel_into(signal, plan, &mut ws)
}

/// Per-voxel SS3T fit into a caller-supplied workspace. Reuses the inner
/// ICLS workspaces (J, R, active set, scratch vectors) and the augmented
/// signal / residual buffers across calls — zero allocation per voxel past
/// the first.
pub fn fit_voxel_into(
    signal: &[f64],
    plan: &Ss3tPlan,
    ws: &mut Ss3tVoxelWorkspace,
) -> (DVector<f64>, f64, f64, Ss3tVoxelDiagnostics) {
    assert_eq!(signal.len(), plan.n_aug);

    // Augment in place: weighted-b0 + b≠0 signal in original gradient order.
    for ((dst, &s), &w) in ws
        .wdwi
        .iter_mut()
        .zip(signal.iter())
        .zip(plan.signal_weight.iter())
    {
        *dst = s * w;
    }

    let mut total_iter = 0usize;
    let mut all_converged = true;

    // Initialise GM+CSF.
    let (init, diag) = plan.icls_gm_csf.solve_into(&ws.wdwi, &mut ws.icls_gm_csf);
    total_iter += diag.iterations;
    all_converged &= diag.converged;
    let mut c_gm = init[0];
    let mut c_csf = init[1];
    let mut c_wm = DVector::<f64>::zeros(plan.n_sh_wm);

    for _ in 0..plan.niter {
        // residual1 = wdwi - pred_csf(c_csf)
        ws.residual.copy_from(&ws.wdwi);
        plan.pred_csf.add_scaled(&[c_csf], -1.0, &mut ws.residual);

        // Fit GM + WM on residual1.
        let (r1, d1) = plan.icls_gm_wm.solve_into(&ws.residual, &mut ws.icls_gm_wm);
        total_iter += d1.iterations;
        all_converged &= d1.converged;
        // r1[0] is GM (discarded by SS3T per the abstract); r1[1..] is WM.
        c_wm.copy_from_slice(&r1.as_slice()[1..1 + plan.n_sh_wm]);

        // residual2 = wdwi - pred_wm(c_wm)
        ws.residual.copy_from(&ws.wdwi);
        plan.pred_wm
            .add_scaled(c_wm.as_slice(), -1.0, &mut ws.residual);

        // Fit GM + CSF on residual2.
        let (r2, d2) = plan.icls_gm_csf.solve_into(&ws.residual, &mut ws.icls_gm_csf);
        total_iter += d2.iterations;
        all_converged &= d2.converged;
        c_gm = r2[0];
        c_csf = r2[1];
    }

    // Final residual diagnostic with all three tissues subtracted.
    ws.residual.copy_from(&ws.wdwi);
    plan.pred_csf.add_scaled(&[c_csf], -1.0, &mut ws.residual);
    plan.pred_gm.add_scaled(&[c_gm], -1.0, &mut ws.residual);
    plan.pred_wm
        .add_scaled(c_wm.as_slice(), -1.0, &mut ws.residual);

    let diagnostics = Ss3tVoxelDiagnostics {
        total_inner_iter: total_iter,
        all_converged,
        residual_l2: ws.residual.norm(),
    };

    (c_wm, c_gm, c_csf, diagnostics)
}

/// Volume-level plan supporting both Fixed and PathBic lmax strategies. Holds
/// one [`Ss3tPlan`] per candidate lmax (a singleton vec for `Fixed`).
pub struct Ss3tVolumePlan {
    /// One plan per candidate lmax, in the same order as `candidates`.
    pub plans: Vec<Ss3tPlan>,
    /// Candidate lmax values.
    pub candidates: Vec<usize>,
    /// Largest lmax across candidates — width of the WM SH coefficient buffer.
    pub max_n_sh_wm: usize,
    /// b=0 weighting factor (same for every candidate; copied here so the
    /// volume driver can write it to the sidecar without reaching into a
    /// specific plan).
    pub bzero_sw: f64,
    /// Number of augmented signal rows.
    pub n_aug: usize,
    /// Strategy used to build this plan.
    pub strategy: LmaxWmStrategy,
}

impl Ss3tVolumePlan {
    /// Build the volume plan from gradient table + responses + config. Does
    /// the per-candidate `Ss3tPlan::build_for_lmax` once each, in serial; the
    /// per-voxel work (parallel) reuses these plans.
    pub fn build(
        gtab: &GradientTable,
        responses: &Ss3tResponses,
        cfg: &Ss3tConfig,
    ) -> Result<Self> {
        let candidates = cfg.lmax_wm.candidates();
        if candidates.is_empty() {
            return Err(CsDmriError::Other(
                "ss3t: lmax_wm strategy must have at least one candidate".into(),
            ));
        }
        let mut plans = Vec::with_capacity(candidates.len());
        for &lmax in &candidates {
            plans.push(Ss3tPlan::build_for_lmax(gtab, responses, lmax, cfg)?);
        }
        let max_n_sh_wm = plans.iter().map(|p| p.n_sh_wm).max().unwrap_or(0);
        let bzero_sw = plans[0].bzero_sw;
        let n_aug = plans[0].n_aug;
        Ok(Self {
            plans,
            candidates,
            max_n_sh_wm,
            bzero_sw,
            n_aug,
            strategy: cfg.lmax_wm.clone(),
        })
    }

    /// Allocate one [`Ss3tVoxelWorkspace`] per candidate plan. Pass to
    /// [`fit_voxel_path_bic_into`].
    pub fn workspaces(&self) -> Vec<Ss3tVoxelWorkspace> {
        self.plans.iter().map(|p| p.workspace()).collect()
    }
}

/// Per-voxel result with path-BIC selection metadata.
pub struct Ss3tVoxelResult {
    /// WM SH coefficients, length = `volume_plan.max_n_sh_wm`. For the
    /// chosen lmax `< max`, higher-order coefficients are zero-padded so
    /// every voxel's output sits at the same dim across the whole volume.
    pub c_wm: DVector<f64>,
    pub c_gm: f64,
    pub c_csf: f64,
    /// Lmax with the smallest BIC (or the only candidate, for Fixed).
    pub chosen_lmax: usize,
    /// BIC at the chosen lmax. `n·log(rss/n) + k·log(n)`.
    pub min_bic: f64,
    /// Diagnostics from the fit at the chosen lmax (not summed across the
    /// other candidates).
    pub diagnostics: Ss3tVoxelDiagnostics,
}

/// Fit every candidate lmax for one voxel, pick the BIC-minimising fit. For
/// `LmaxWmStrategy::Fixed`, only one candidate exists — equivalent to
/// `fit_voxel_into` plus a BIC computation.
///
/// `workspaces` must have the same length and order as `volume_plan.plans`.
pub fn fit_voxel_path_bic_into(
    signal: &[f64],
    volume_plan: &Ss3tVolumePlan,
    workspaces: &mut [Ss3tVoxelWorkspace],
) -> Ss3tVoxelResult {
    assert_eq!(workspaces.len(), volume_plan.plans.len());
    let n_aug = volume_plan.n_aug;
    let n = n_aug as f64;

    let mut best_bic = f64::INFINITY;
    let mut best_idx = 0_usize;
    let mut best_c_wm: Option<DVector<f64>> = None;
    let mut best_c_gm = 0.0;
    let mut best_c_csf = 0.0;
    let mut best_diag = Ss3tVoxelDiagnostics::default();

    for (idx, plan) in volume_plan.plans.iter().enumerate() {
        let (c_wm, c_gm, c_csf, diag) = fit_voxel_into(signal, plan, &mut workspaces[idx]);
        // BIC: n * ln(rss / n) + k * ln(n). k = n_sh_wm + 2 (GM + CSF + WM).
        let rss = diag.residual_l2 * diag.residual_l2;
        let k = (plan.n_sh_wm + 2) as f64;
        // Guard against rss == 0 (perfect fit on degenerate data).
        let bic = if rss > 0.0 {
            n * (rss / n).ln() + k * n.ln()
        } else {
            f64::NEG_INFINITY
        };
        if bic < best_bic {
            best_bic = bic;
            best_idx = idx;
            best_c_wm = Some(c_wm);
            best_c_gm = c_gm;
            best_c_csf = c_csf;
            best_diag = diag;
        }
    }

    // Pad chosen WM coefficients up to max_n_sh_wm.
    let chosen_c_wm = best_c_wm.expect("at least one candidate must have been fit");
    let mut padded = DVector::<f64>::zeros(volume_plan.max_n_sh_wm);
    for i in 0..chosen_c_wm.len() {
        padded[i] = chosen_c_wm[i];
    }

    Ss3tVoxelResult {
        c_wm: padded,
        c_gm: best_c_gm,
        c_csf: best_c_csf,
        chosen_lmax: volume_plan.candidates[best_idx],
        min_bic: best_bic,
        diagnostics: best_diag,
    }
}

fn scale_b0(responses: &Ss3tResponses, w: f64) -> Ss3tResponses {
    Ss3tResponses {
        wm: scale_b0_row(&responses.wm, w),
        gm: scale_b0_row(&responses.gm, w),
        csf: scale_b0_row(&responses.csf, w),
    }
}

fn scale_b0_row(r: &TissueResponse, w: f64) -> TissueResponse {
    let mut out = r.clone();
    if let Some(row) = out.coeffs.first_mut() {
        for c in row.iter_mut() {
            *c *= w;
        }
    }
    out
}

fn build_shell_plans(gtab: &GradientTable, shells: &[Shell]) -> Result<Vec<ShellPlan>> {
    let mut out = Vec::with_capacity(shells.len());
    for shell in shells {
        let dirs: Vec<[f32; 3]> = shell
            .indices
            .iter()
            .map(|&i| {
                let v = gtab.bvecs[i];
                [v[0] as f32, v[1] as f32, v[2] as f32]
            })
            .collect();
        out.push(ShellPlan {
            b: shell.b,
            dirs_ras: dirs,
            aug_indices: shell.indices.clone(),
        });
    }
    Ok(out)
}

/// Build the GM+WM constraint matrix:
///   row 0:  [1, 0, 0, …]            — c_gm ≥ 0
///   rows 1..1+nv: [0, Y_sphere(d_i)] — WM amplitude at each sphere direction ≥ 0
fn build_gm_wm_constraint(lmax_wm: usize, n_sh_wm: usize) -> DMatrix<f64> {
    let dirs = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
    let y = sh2amp_cart(dirs, lmax_wm);
    let n_dirs = dirs.len();
    let n_cols = 1 + n_sh_wm;
    let n_rows = 1 + n_dirs;
    let mut a = DMatrix::<f64>::zeros(n_rows, n_cols);
    a[(0, 0)] = 1.0;
    for i in 0..n_dirs {
        for j in 0..n_sh_wm {
            a[(1 + i, 1 + j)] = y[(i, j)] as f64;
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qspace::GradientTable;
    use approx::assert_abs_diff_eq;

    /// Build a three-tissue plan from synthetic responses on a moderately
    /// dense gradient scheme: 6 b=0 + 60 DWI directions sampled from the
    /// dsistudio sphere. 60 directions is enough to make a 1+45 = 46-unknown
    /// GM+WM system well-determined at lmax_wm=8.
    fn synth_plan() -> (Ss3tPlan, GradientTable) {
        let sphere = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
        // Stride to get ~60 evenly spread directions.
        let step = sphere.len() / 60;
        let dwi_dirs: Vec<[f64; 3]> = sphere
            .iter()
            .step_by(step.max(1))
            .take(60)
            .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
            .collect();
        let n_dwi = dwi_dirs.len();
        let n_b0 = 6;
        let mut bvals = vec![0.0; n_b0];
        let mut bvecs: Vec<[f64; 3]> = vec![[0.0; 3]; n_b0];
        for d in &dwi_dirs {
            bvals.push(1000.0);
            bvecs.push(*d);
        }
        let _ = n_dwi;
        let gtab =
            GradientTable::new(bvals, bvecs, Some(0.05), Some(0.012), None).unwrap();

        // WM response: b=0 isotropic, dwi anisotropic. SDM ordering required
        // by SS3T is SDM(WM) < SDM(GM) < SDM(CSF) — WM has the *least* decay,
        // CSF the most. Mirroring biology: WM ≈ 0.6, GM ≈ 0.4, CSF ≈ 0.1 at
        // b=1000.
        let wm = TissueResponse {
            coeffs: vec![
                vec![1.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.6, -0.2, 0.05, -0.01, 0.001],
            ],
            lmax: 8,
        };
        let gm = TissueResponse {
            coeffs: vec![vec![1.0], vec![0.4]],
            lmax: 0,
        };
        let csf = TissueResponse {
            coeffs: vec![vec![1.0], vec![0.1]],
            lmax: 0,
        };
        let responses = Ss3tResponses { wm, gm, csf };
        let cfg = Ss3tConfig::default();
        let plan = Ss3tPlan::build(&gtab, &responses, &cfg).unwrap();
        (plan, gtab)
    }

    #[test]
    fn plan_builds_with_correct_dims() {
        let (plan, gtab) = synth_plan();
        let n_b0 = 6;
        let n_dwi = gtab.n_grads() - n_b0;
        assert_eq!(plan.n_aug, n_b0 + n_dwi);
        assert_eq!(plan.lmax_wm, 8);
        assert_eq!(plan.n_sh_wm, ncoeffs_for_lmax(8));
        // bzero_sw = sqrt(n_dwi · 10 / (n_b0 · 100))
        let expected = (n_dwi as f64 * 10.0 / (n_b0 as f64 * 100.0)).sqrt();
        assert_abs_diff_eq!(plan.bzero_sw, expected, epsilon = 1e-12);
        for i in 0..n_b0 {
            assert_abs_diff_eq!(plan.signal_weight[i], plan.bzero_sw, epsilon = 1e-12);
        }
        for i in n_b0..plan.n_aug {
            assert_abs_diff_eq!(plan.signal_weight[i], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn pure_csf_voxel_recovers_csf_only() {
        let (plan, _) = synth_plan();
        // Synthesize a pure-CSF voxel — only CSF contributes.
        // CSF coef = 0.8 → predicted signal = 0.8 · csf.r_0[shell] per row.
        let mut wdwi = DVector::<f64>::zeros(plan.n_aug);
        plan.pred_csf.add_scaled(&[0.8], 1.0, &mut wdwi);
        // Convert from augmented back to "raw" by dividing out the weight.
        let raw: Vec<f64> = (0..plan.n_aug)
            .map(|i| wdwi[i] / plan.signal_weight[i])
            .collect();
        let (c_wm, c_gm, c_csf, diag) = fit_voxel(&raw, &plan);
        assert!(diag.all_converged, "CSD inner solves did not all converge");
        // CSF should recover near 0.8; GM and WM near 0.
        assert!((c_csf - 0.8).abs() < 0.05, "c_csf = {}", c_csf);
        assert!(c_gm.abs() < 0.05, "c_gm = {}", c_gm);
        for (i, &c) in c_wm.iter().enumerate() {
            assert!(c.abs() < 0.05, "c_wm[{}] = {}", i, c);
        }
    }

    #[test]
    fn three_tissue_synthesis_roundtrip_matches_input_signal() {
        // Plant a 3-tissue voxel, fit, re-synthesize from the recovered
        // coefficients, and verify the prediction matches the input. With only
        // two shells (b=0 and b=1000), an isotropic-only WM is degenerate
        // with GM/CSF and the decomposition is not unique — but the prediction
        // is. So we test that the algorithm finds *some* valid fixed point
        // (the inner CSD solver and outer alternation are self-consistent).
        let (plan, _) = synth_plan();
        let mut wdwi = DVector::<f64>::zeros(plan.n_aug);
        plan.pred_csf.add_scaled(&[0.2], 1.0, &mut wdwi);
        plan.pred_gm.add_scaled(&[0.3], 1.0, &mut wdwi);
        let mut wm_in = vec![0.0_f64; plan.n_sh_wm];
        wm_in[0] = 0.5 / (4.0 * std::f64::consts::PI).sqrt();
        wm_in[3] = 0.04; // (l=2, m=0) — adds anisotropic content.
        plan.pred_wm.add_scaled(&wm_in, 1.0, &mut wdwi);
        let raw: Vec<f64> = (0..plan.n_aug)
            .map(|i| wdwi[i] / plan.signal_weight[i])
            .collect();

        let (c_wm, c_gm, c_csf, _) = fit_voxel(&raw, &plan);

        // Re-synthesize the augmented signal from the fitted coefficients.
        let mut resynth = DVector::<f64>::zeros(plan.n_aug);
        plan.pred_csf.add_scaled(&[c_csf], 1.0, &mut resynth);
        plan.pred_gm.add_scaled(&[c_gm], 1.0, &mut resynth);
        plan.pred_wm.add_scaled(c_wm.as_slice(), 1.0, &mut resynth);

        let diff = (&wdwi - &resynth).norm();
        let sig = wdwi.norm();
        assert!(
            diff / sig < 0.01,
            "relative roundtrip residual = {}",
            diff / sig
        );
        // SS3T constraints: GM and CSF coefficients are non-negative.
        assert!(c_gm >= -1e-6, "c_gm = {}", c_gm);
        assert!(c_csf >= -1e-6, "c_csf = {}", c_csf);
    }
}
