// SPDX-License-Identifier: MIT OR Apache-2.0
//! Continuous-b multi-shell multi-tissue CSD for non-shelled q-space (CS-DSI).
//!
//! Standard MSMT-CSD needs shells: it bins gradients by b-value and uses one
//! response row per shell. A Cartesian CS-DSI q-space grid is not shelled, and
//! rounding its b-values into pseudo-shells discards q-space information.
//!
//! Continuous-b MSMT avoids the binning. It lays a **dense b-grid**
//! (`0, step, 2·step, …`) over q-space, assigns each gradient to its nearest
//! grid point, and gives every grid point its own response row (the tissue
//! response evaluated *at that b*). Assembling the forward operator over these
//! fine per-b groups — instead of a few coarse shells — makes the deconvolution
//! continuous in b while reusing the crate's [`build_h`] assembly and
//! [`IclsSolver`] verbatim. On a genuinely shelled acquisition it reduces
//! exactly to per-shell MSMT (see the `dense_b_equals_shelled_forward` test).
//!
//! Unlike SS3T's alternating two-tissue fixed point, this is a straight joint
//! multi-tissue fit: one inequality-constrained least squares over
//! `[c_csf, c_gm, c_wm…]` with `c_csf ≥ 0`, `c_gm ≥ 0`, and the WM fODF
//! amplitude ≥ 0 on a dense sphere.
//!
//! Deconvolution of non-shelled data against responses defined continuously in
//! b was also described independently by Morez, Sijbers, Vanhevel & Jeurissen,
//! *"Constrained spherical deconvolution of nonspherically sampled diffusion MRI
//! data"*, Hum Brain Mapp 42:521–538 (2021). Their parametric response model is
//! available as one of the dense-b response estimators
//! ([`crate::multitissue::response_dki`]).

use std::sync::Arc;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array3, Array4};

use crate::graddev::VoxelGradDev;
use crate::io::dwi::DwiData;
use crate::multitissue::forward::{ShellPlan, TissueSlot, build_h};
use crate::multitissue::response::TissueResponse;
use crate::qspace::GradientTable;
use crate::sh::{ncoeffs_for_lmax, sh2amp_cart, zonal_factors_per_coeff};
use crate::solver::icls::{IclsConfig, IclsDiagnostics, IclsSolver, IclsWorkspace};
use crate::voxel_loop;
use crate::{CsDmriError, Result};

/// A dense b-grid: `bs[k] = k · step`, `k = 0 … n`. Grid point 0 is b=0.
#[derive(Debug, Clone)]
pub struct DenseBGrid {
    pub step: f64,
    /// Grid b-values, ascending, `bs[0] == 0.0`.
    pub bs: Vec<f64>,
}

impl DenseBGrid {
    /// Build a grid covering `0 ..= max_bval + pad` in steps of `step`.
    pub fn new(max_bval: f64, step: f64, pad: f64) -> Self {
        assert!(step > 0.0, "dense b-grid step must be > 0");
        let top = (max_bval + pad).max(0.0);
        let n = (top / step).ceil() as usize;
        let bs = (0..=n).map(|k| k as f64 * step).collect();
        Self { step, bs }
    }

    /// Grid covering a gradient table (bmax from `gtab`, default pad = 2·step).
    pub fn for_gtab(gtab: &GradientTable, step: f64) -> Self {
        let bmax = gtab.bvals.iter().cloned().fold(0.0_f64, f64::max);
        Self::new(bmax, step, 2.0 * step)
    }

    /// Nearest grid index for a b-value (clamped into range).
    #[inline]
    pub fn nearest(&self, b: f64) -> usize {
        let k = (b / self.step).round() as isize;
        k.clamp(0, self.bs.len() as isize - 1) as usize
    }

    /// Bracketing grid index and weight for linear interpolation in b:
    /// `b ≈ (1 − w)·bs[k] + w·bs[k + 1]`, with `w ∈ [0, 1]` (`w = 0` on a grid
    /// point, so interpolation reduces to [`nearest`](Self::nearest) there).
    /// Clamped to the grid ends.
    #[inline]
    pub fn bracket(&self, b: f64) -> (usize, f64) {
        if self.bs.len() < 2 {
            return (0, 0.0);
        }
        let x = (b / self.step).max(0.0);
        let k = (x.floor() as usize).min(self.bs.len() - 2);
        (k, (x - k as f64).clamp(0.0, 1.0))
    }

    pub fn len(&self) -> usize {
        self.bs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bs.is_empty()
    }
}

/// Assign every gradient to its nearest dense-b grid point and emit one
/// [`ShellPlan`] per grid point (in grid order). Grid points with no gradients
/// get an empty plan — [`build_h`] skips them, and keeping them keeps the plan
/// index in 1:1 correspondence with the response row index (`response.coeffs[k]`
/// is the response at `grid.bs[k]`), so no remapping is needed downstream.
pub fn build_dense_b_plans(gtab: &GradientTable, grid: &DenseBGrid) -> Vec<ShellPlan> {
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); grid.len()];
    for (i, &b) in gtab.bvals.iter().enumerate() {
        buckets[grid.nearest(b)].push(i);
    }
    buckets
        .into_iter()
        .enumerate()
        .map(|(k, indices)| {
            let dirs_ras: Vec<[f32; 3]> = indices
                .iter()
                .map(|&i| {
                    let v = gtab.bvecs[i];
                    [v[0] as f32, v[1] as f32, v[2] as f32]
                })
                .collect();
            ShellPlan { b: grid.bs[k], dirs_ras, aug_indices: indices }
        })
        .collect()
}

/// The three dense-b tissue responses. Each must have exactly `grid.len()` rows
/// (one zonal-SH row per grid point). Row `k` is the response at `grid.bs[k]`.
#[derive(Debug, Clone)]
pub struct ConshResponses {
    pub wm: TissueResponse,
    pub gm: TissueResponse,
    pub csf: TissueResponse,
}

/// Configuration for the continuous-b MSMT fit.
#[derive(Debug, Clone)]
pub struct ConshConfig {
    /// WM SH order (clamped to the WM response's lmax). Default 8.
    pub lmax_wm: usize,
    /// Dense b-grid step in s/mm². Default 50.
    pub step: f64,
    /// Inner ICLS solver configuration.
    pub icls: IclsConfig,
    /// Tikhonov ridge (L2) applied to the WM fODF's orientation-bearing SH
    /// coefficients (l≥2 only; the isotropic l=0 and the GM/CSF amplitudes are
    /// left undamped). 0 = current behavior (non-negativity only). A positive
    /// value damps the ill-conditioned high-order terms that drive run-to-run
    /// peak jitter, matching the regularization a stable CSD relies on.
    pub wm_l2_ridge: f64,
    /// Ascending WM-l≥2 ridge path (first entry the reference, usually 0.0) for
    /// the **per-voxel L2-anchored** ridge. When it has ≥2 entries, each voxel
    /// picks the most-damped λ whose data residual stays within
    /// `(1 + l2_anchor_slack)` of the reference (λ=path[0]) fit — so ambiguous
    /// crossings get heavy damping while well-determined single-fiber voxels
    /// stay sharp. Empty ⇒ use the single global `wm_l2_ridge`.
    pub wm_ridge_path: Vec<f64>,
    /// Fractional residual slack for the per-voxel anchor selection.
    pub l2_anchor_slack: f64,
    /// If true, the ridge λ (global and path) is **scale-relative**: it is
    /// multiplied by the mean WM-l≥2 HᵀH diagonal (a per-acquisition data-weight
    /// scalar), so λ becomes a dimensionless fraction and one setting works across
    /// acquisitions (shelled vs Cartesian, different bmax) while the ridge keeps
    /// its conditioning-aware constant-across-coeffs shape. If false, λ is an
    /// absolute addition to the Hessian diagonal (acquisition-scale-dependent).
    pub wm_ridge_relative: bool,
    /// Fit a fourth, **b-independent** compartment that absorbs the Rician noise
    /// floor. Default true.
    ///
    /// In magnitude data the noise floor is additive in *absolute* units
    /// (`≈ σ·√(π/2)`), but the multi-tissue model is multiplicative with
    /// S0-normalized responses. A voxel whose S0 is far below that of the voxels
    /// the responses were estimated from therefore carries a relative floor the
    /// three tissue responses cannot reproduce — and since WM is the only
    /// compartment retaining appreciable amplitude at high b, the non-negative
    /// fit dumps the whole excess into the WM fODF. That is what produces the
    /// large spurious fODFs in low-S0 CSF (cisterns, subarachnoid space) and in
    /// any non-brain voxel left inside the mask.
    ///
    /// The flat column costs one degree of freedom, needs no noise estimate, and
    /// is self-calibrating per voxel — which matters most for CS acquisitions,
    /// where the reconstruction makes σ non-stationary and hard to estimate.
    /// It goes to ~0 wherever there is no floor to absorb.
    pub fit_noise_floor: bool,
    /// Quantitative (AFD) mode: rescale the fit back to **absolute** signal units
    /// by multiplying every fitted coefficient — WM, GM, CSF *and* the noise
    /// floor — by that voxel's b=0 signal `S0`. Default false.
    ///
    /// The solve itself is unchanged. It fits `FOD ∗ R = S/S0`, and both the
    /// least-squares objective and the non-negativity cone are positively
    /// homogeneous, so `c · S0` is *exactly* the coefficient vector that would
    /// have been obtained by deconvolving the absolute signal `S` with the same
    /// (S0=1-scale) kernel. Doing it in this order — normalize, fit, un-normalize
    /// — also keeps the ridge penalty-to-data ratio constant across voxels, which
    /// fitting absolute `S` under a fixed λ would not.
    ///
    /// This restores the property Apparent Fiber Density and SIFT2 depend on: the
    /// fODF amplitude is proportional to the DWI signal (∝ intra-cellular volume
    /// at high b). Per-voxel `S/S0` normalization removes it, and both
    /// Smith et al. 2022 (*Quantitative streamlines tractography*) and Dhollander
    /// et al. 2021 (*Fixel-based analysis*) require its absence for cross-subject
    /// FD/FBC comparison.
    ///
    /// The responses are scaled by one common per-scan reference (the
    /// single-fiber-WM median b=0, so `r_0(b=0) ≈ √4π` for WM), preserving the
    /// relative amplitudes between tissues exactly as MRtrix's `amp2response`
    /// responses do. The `AFD_ref` half of the scaling becomes common across
    /// subjects by sharing a group kernel (`consh-response mean`, the
    /// `responsemean` analogue) — MRtrix's own mechanism. The remaining
    /// per-subject `DWI_ref` scale is left to downstream global intensity
    /// normalization — so `mtnormalise` must stay enabled, and it stops being a
    /// no-op here: the tissue sum becomes `≈ 0.2821 · S0(v)`, whose smooth
    /// component is the bias field and whose per-voxel residual is the AFD
    /// contrast.
    ///
    /// See `docs/quantitative-consh.md`.
    pub quantitative: bool,
    /// Interpolate the response linearly in b between the two bracketing grid
    /// rows, instead of snapping each measurement to its nearest row (±step/2).
    /// Default false. Turned on with a gradient-deviation field, whose
    /// few-percent per-voxel b changes would otherwise be lost to the snapping.
    /// Identical to nearest for b-values that sit on grid points.
    pub b_interp: bool,
    /// Extend the dense-b grid to at least this b (s/mm²). `None` = the table's
    /// bmax. A gradient-deviation field raises b above the nominal bmax, and
    /// the responses must then carry rows up to the largest effective b.
    pub grid_bmax: Option<f64>,
}

impl Default for ConshConfig {
    fn default() -> Self {
        Self {
            lmax_wm: 8,
            step: 50.0,
            icls: IclsConfig::default(),
            wm_l2_ridge: 0.0,
            wm_ridge_path: Vec::new(),
            l2_anchor_slack: 0.0,
            wm_ridge_relative: false,
            fit_noise_floor: true,
            quantitative: false,
            b_interp: false,
            grid_bmax: None,
        }
    }
}

impl DenseBGrid {
    /// Grid for a table, extended to `grid_bmax` when that is larger.
    pub fn for_gtab_min_bmax(gtab: &GradientTable, step: f64, grid_bmax: Option<f64>) -> Self {
        let bmax = gtab.bvals.iter().cloned().fold(0.0_f64, f64::max);
        Self::new(bmax.max(grid_bmax.unwrap_or(0.0)), step, 2.0 * step)
    }
}

/// Zonal `l=0` coefficient giving an isotropic compartment a forward column of
/// exactly `1.0` per measurement.
///
/// `build_h` forms `Y_00 · √(4π) · r_0` and `Y_00 = 1/(2√π)`, so the assembled
/// column equals `r_0`. Using `2√π` matches the scale the GM/CSF responses carry
/// on their b=0 row, which keeps the fitted floor amplitude directly comparable
/// to `c_gm` / `c_csf` (all three are ≈0.2821 for a compartment filling the
/// voxel). Divide by 0.2821 to read the floor as a fraction of S0.
fn floor_zonal() -> f64 {
    2.0 * std::f64::consts::PI.sqrt()
}

/// A flat (b-independent) response spanning the dense-b grid: the basis function
/// for the Rician noise floor. See [`ConshConfig::fit_noise_floor`].
fn flat_floor_response(n_rows: usize) -> TissueResponse {
    TissueResponse { coeffs: vec![vec![floor_zonal()]; n_rows], lmax: 0 }
}

/// Volume-invariant precompute for the continuous-b MSMT fit: the dense-b
/// grouping, the joint forward operator `H = [H_csf | H_gm | H_wm]`, the
/// non-negativity constraint, and the ICLS solver (with its cached Cholesky).
pub struct ConshMsmtPlan {
    pub n_aug: usize,
    pub lmax_wm: usize,
    pub n_sh_wm: usize,
    /// `(start_col, width)` for csf, gm, wm within the coefficient vector.
    pub csf_col: usize,
    pub gm_col: usize,
    pub wm_col: (usize, usize),
    /// Column of the flat noise-floor compartment, when enabled.
    pub floor_col: Option<usize>,
    /// Joint ICLS solver for `[c_csf, c_gm, c_wm…]` (global-ridge / no-ridge).
    pub icls: IclsSolver,
    /// Per-voxel L2-anchored ridge path: `(λ, solver)` ascending in λ. Empty
    /// unless `ConshConfig::wm_ridge_path` had ≥2 entries. When present,
    /// [`fit_voxel_into`](Self::fit_voxel_into) selects per voxel instead of
    /// using `icls`.
    pub ridge_path: Vec<(f64, IclsSolver)>,
    /// Fractional residual slack for the per-voxel anchor selection.
    pub anchor_slack: f64,
    pub grid: DenseBGrid,
    /// What a per-voxel table needs to rebuild `H` and its solvers
    /// ([`fit_voxel_table_into`](Self::fit_voxel_table_into)).
    voxel: VoxelPrecompute,
}

/// Table-independent pieces for assembling `H` and its solvers per voxel.
struct VoxelPrecompute {
    /// `(start_col, width)` and SH order of each tissue slot, in column order.
    slots: Vec<((usize, usize), usize)>,
    /// `factors[slot][k]`: per-coefficient zonal factors of slot at grid row k.
    factors: Vec<Vec<Vec<f64>>>,
    n_cols: usize,
    /// `(start_col, width)` of the WM block (the ridge acts on its l≥2 part).
    wm_col: (usize, usize),
    /// Rows at or below this b keep their nearest grid row: their direction is
    /// not meaningful, so they must not pick up anisotropic response terms.
    b0_threshold: f64,
    b_interp: bool,
    constraint: Arc<DMatrix<f64>>,
    icls_cfg: IclsConfig,
    wm_l2_ridge: f64,
    wm_ridge_path: Vec<f64>,
    /// From the nominal table, so λ means the same in every voxel.
    ridge_scale: f64,
}

impl VoxelPrecompute {
    /// Forward operator for one gradient table (`dirs` in the fit's frame), one
    /// row per measurement. With `b_interp` off this equals [`build_h`] over
    /// [`build_dense_b_plans`] exactly; with it on, each row's zonal factors are
    /// interpolated between the two grid rows bracketing its b (H is linear in
    /// the response, so this is the response at that b).
    fn assemble_h(&self, grid: &DenseBGrid, bvals: &[f64], dirs: &[[f64; 3]]) -> DMatrix<f64> {
        let n = bvals.len();
        let dirs32: Vec<[f32; 3]> = dirs.iter().map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]).collect();
        let mut h = DMatrix::<f64>::zeros(n, self.n_cols);
        for (s, &((c0, _), lmax)) in self.slots.iter().enumerate() {
            let y = sh2amp_cart(&dirs32, lmax);
            for (row, &b) in bvals.iter().enumerate() {
                let (k, w) = if self.b_interp && b > self.b0_threshold {
                    grid.bracket(b)
                } else {
                    (grid.nearest(b), 0.0)
                };
                let fk = &self.factors[s][k];
                if w > 0.0 {
                    let fk1 = &self.factors[s][k + 1];
                    for j in 0..fk.len().min(fk1.len()) {
                        h[(row, c0 + j)] = y[(row, j)] as f64 * ((1.0 - w) * fk[j] + w * fk1[j]);
                    }
                } else {
                    for j in 0..fk.len() {
                        h[(row, c0 + j)] = y[(row, j)] as f64 * fk[j];
                    }
                }
            }
        }
        h
    }

    /// The ICLS solver(s) for `h`: the single solver (global `wm_l2_ridge`) when
    /// `with_single`, and the per-voxel anchored ridge path when configured
    /// (≥2 λ). `HᵀH` is formed once and shared across them.
    fn solvers(&self, h: DMatrix<f64>, with_single: bool) -> (Option<IclsSolver>, Vec<(f64, IclsSolver)>) {
        let ht = h.transpose();
        let gram = &ht * &h;
        let (wm_start, n_sh_wm) = self.wm_col;
        let make = |lam: f64| {
            let ridge = (lam > 0.0)
                .then(|| wm_l2_ridge_diag(lam, self.n_cols, wm_start, n_sh_wm, self.ridge_scale));
            IclsSolver::from_gram(
                h.clone(),
                ht.clone(),
                &gram,
                Arc::clone(&self.constraint),
                self.icls_cfg,
                ridge.as_deref(),
            )
        };
        let path = if self.wm_ridge_path.len() >= 2 {
            self.wm_ridge_path.iter().map(|&lam| (lam, make(lam))).collect()
        } else {
            Vec::new()
        };
        let single = with_single.then(|| make(self.wm_l2_ridge));
        (single, path)
    }
}

/// Build the per-coefficient ridge diagonal that damps only the WM l≥2 SH
/// columns (leaving csf/gm and the WM l=0 isotropic term undamped).
fn wm_l2_ridge_diag(
    lambda: f64,
    n_cols: usize,
    wm_start: usize,
    n_sh_wm: usize,
    scale: f64,
) -> Vec<f64> {
    // Constant ridge across the WM l≥2 columns (conditioning-aware: shrinks
    // low-HᵀH / ill-conditioned directions more). `scale` = 1.0 for absolute λ,
    // or the mean WM-l≥2 HᵀH diagonal for the scale-relative (dimensionless λ)
    // mode, so one λ works across acquisitions.
    let mut ridge = vec![0.0_f64; n_cols];
    for j in 1..n_sh_wm {
        ridge[wm_start + j] = lambda * scale;
    }
    ridge
}

impl ConshMsmtPlan {
    /// Build the plan from the gradient table + dense-b responses + config.
    pub fn build(
        gtab: &GradientTable,
        responses: &ConshResponses,
        cfg: &ConshConfig,
    ) -> Result<Self> {
        // Validate tissue response shapes.
        responses
            .gm
            .require_isotropic("gm response")
            .map_err(CsDmriError::from)?;
        responses
            .csf
            .require_isotropic("csf response")
            .map_err(CsDmriError::from)?;

        let grid = DenseBGrid::for_gtab_min_bmax(gtab, cfg.step, cfg.grid_bmax);
        for (name, r) in [
            ("wm", &responses.wm),
            ("gm", &responses.gm),
            ("csf", &responses.csf),
        ] {
            // Row k is the response at b = k·step whatever the grid's length, so
            // a response covering a wider b range (e.g. one written under
            // --graddev, or a group kernel) serves a shorter grid unchanged:
            // the rows past the grid are simply never used. Too few rows is an
            // error — the top of the grid would have no response.
            if r.n_shells() < grid.len() {
                return Err(CsDmriError::Other(format!(
                    "consh: {name} response has {} rows, need at least {} (one per dense-b grid point \
                     up to b={:.0}, step={})",
                    r.n_shells(),
                    grid.len(),
                    grid.bs.last().copied().unwrap_or(0.0),
                    cfg.step
                )));
            }
        }

        // Kernels must be on the common-reference scale: every tissue is divided
        // by the scan's single-fiber-WM median b=0 at estimation, so the WM b=0
        // coefficient sits near √4π (only approximately — the mean/median ratio
        // of the training population and group pooling both move it a few
        // percent; do NOT tighten this into an exact-anchor check). A value far
        // outside that range means a raw scanner-unit response (e.g. MRtrix
        // `dwi2response` output, O(10²–10³)) was supplied, which would silently
        // put the fODFs on an incomparable scale.
        let wm_b0 = responses
            .wm
            .coeffs
            .first()
            .and_then(|row| row.first())
            .copied()
            .unwrap_or(0.0);
        let anchor_ratio = wm_b0 / (4.0 * std::f64::consts::PI).sqrt();
        if !(0.2..=5.0).contains(&anchor_ratio) {
            return Err(CsDmriError::Other(format!(
                "consh: WM response b=0 coefficient {wm_b0:.3} is far from √4π ≈ 3.545 \
                 (ratio {anchor_ratio:.3}); the kernel is not on the common-reference \
                 (single-fiber-WM b=0) scale — raw scanner-unit responses (MRtrix \
                 dwi2response output) cannot be used directly"
            )));
        }

        let lmax_wm = cfg.lmax_wm.min(responses.wm.lmax);
        if lmax_wm % 2 != 0 {
            return Err(CsDmriError::Other(format!(
                "consh: lmax_wm must be even, got {lmax_wm}"
            )));
        }
        let n_sh_wm = ncoeffs_for_lmax(lmax_wm);
        let n_aug = gtab.n_grads();

        // Column layout: csf (1) | gm (1) | [floor (1)] | wm (n_sh_wm). Iso
        // first, matching the dipy MultiShellResponse convention; the optional
        // flat noise-floor column joins the other isotropic compartments so the
        // WM block stays contiguous at the end.
        let floor_response = flat_floor_response(grid.len());
        let mut tissues = vec![
            TissueSlot { response: &responses.csf, lmax: 0 },
            TissueSlot { response: &responses.gm, lmax: 0 },
        ];
        if cfg.fit_noise_floor {
            tissues.push(TissueSlot { response: &floor_response, lmax: 0 });
        }
        tissues.push(TissueSlot { response: &responses.wm, lmax: lmax_wm });

        let mut slots = Vec::with_capacity(tissues.len());
        let mut next_col = 0usize;
        for t in &tissues {
            slots.push(((next_col, t.n_coeffs()), t.lmax));
            next_col += t.n_coeffs();
        }
        let csf_col = slots[0].0.0;
        let gm_col = slots[1].0.0;
        let floor_col = cfg.fit_noise_floor.then(|| slots[2].0.0);
        let wm_col = slots.last().expect("wm slot is always present").0;

        let n_cols = 2 + usize::from(cfg.fit_noise_floor) + n_sh_wm;
        let constraint =
            build_msmt_constraint(csf_col, gm_col, floor_col, wm_col, n_cols, lmax_wm);
        let factors = tissues
            .iter()
            .map(|t| {
                (0..grid.len())
                    .map(|k| {
                        zonal_factors_per_coeff(t.response.coeffs.get(k).map_or(&[][..], |v| &v[..]), t.lmax)
                    })
                    .collect()
            })
            .collect();
        let mut voxel = VoxelPrecompute {
            slots,
            factors,
            n_cols,
            wm_col,
            b0_threshold: gtab.b0_threshold,
            b_interp: cfg.b_interp,
            constraint: Arc::new(constraint),
            icls_cfg: cfg.icls,
            wm_l2_ridge: cfg.wm_l2_ridge,
            wm_ridge_path: cfg.wm_ridge_path.clone(),
            ridge_scale: 1.0,
        };
        // Without interpolation the nominal H is the per-shell assembly it always
        // was; with it, the same per-row assembly the per-voxel tables use.
        let h = if cfg.b_interp {
            voxel.assemble_h(&grid, &gtab.bvals, &gtab.bvecs)
        } else {
            build_h(n_aug, &build_dense_b_plans(gtab, &grid), &tissues).h
        };
        // For the scale-relative ridge, weight λ by each column's own HᵀH diagonal
        // (squared column norm of the forward operator) so λ is dimensionless and
        // one setting works across acquisitions. Computed once, before H is moved.
        // Scale-relative ridge: normalize λ by the MEAN WM-l≥2 HᵀH diagonal (a
        // per-acquisition scalar) so λ is acquisition-invariant, while keeping the
        // constant-across-coeffs (conditioning-aware) ridge shape.
        let ridge_scale = if cfg.wm_ridge_relative {
            let wm_hth: Vec<f64> = (1..n_sh_wm)
                .map(|j| h.column(wm_col.0 + j).iter().map(|v| v * v).sum::<f64>())
                .collect();
            let mean = wm_hth.iter().sum::<f64>() / wm_hth.len().max(1) as f64;
            let (mn, mx) = wm_hth
                .iter()
                .fold((f64::INFINITY, 0.0_f64), |(a, b), &v| (a.min(v), b.max(v)));
            eprintln!(
                "[consh]   WM l>=2 HtH-diag: mean={:.3} min={:.3} max={:.3} (relative-ridge scale)",
                mean, mn, mx
            );
            mean
        } else {
            1.0
        };
        voxel.ridge_scale = ridge_scale;
        // The global solver always exists (it sizes the workspaces); the ridge
        // path only when configured.
        let (icls, ridge_path) = voxel.solvers(h, true);
        let icls = icls.expect("requested the single solver");

        Ok(Self {
            n_aug,
            lmax_wm,
            n_sh_wm,
            csf_col,
            gm_col,
            wm_col,
            floor_col,
            icls,
            ridge_path,
            anchor_slack: cfg.l2_anchor_slack,
            grid,
            voxel,
        })
    }

    /// Width of the coefficient vector this plan solves for: the two isotropic
    /// tissues, the optional noise-floor column, and the WM SH block.
    pub fn n_cols(&self) -> usize {
        2 + usize::from(self.floor_col.is_some()) + self.n_sh_wm
    }

    /// Allocate a reusable per-worker workspace.
    pub fn workspace(&self) -> ConshMsmtWorkspace {
        ConshMsmtWorkspace {
            icls: self.icls.workspace(),
            signal: DVector::<f64>::zeros(self.n_aug),
            resid: DVector::<f64>::zeros(self.n_aug),
        }
    }
}

/// Per-voxel scratch for [`ConshMsmtPlan::fit_voxel_into`].
pub struct ConshMsmtWorkspace {
    pub icls: IclsWorkspace,
    pub signal: DVector<f64>,
    /// Scratch for the data residual `H x − s` (per-voxel anchor selection).
    pub resid: DVector<f64>,
}

/// Result of one voxel's continuous-b MSMT fit.
#[derive(Debug, Clone)]
pub struct ConshVoxelResult {
    /// WM fODF SH coefficients (MRtrix basis), length `n_sh_wm`.
    pub c_wm: DVector<f64>,
    pub c_gm: f64,
    pub c_csf: f64,
    /// Flat noise-floor amplitude, `None` when the compartment is disabled.
    /// Same scale as `c_gm` / `c_csf`; divide by 0.2821 for a fraction of S0.
    pub c_floor: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
}

impl ConshMsmtPlan {
    /// Fit one voxel. `signal` length must equal `n_aug` (= n_grads), in the
    /// original gradient order.
    pub fn fit_voxel(&self, signal: &[f64]) -> ConshVoxelResult {
        let mut ws = self.workspace();
        self.fit_voxel_into(signal, &mut ws)
    }

    /// Fit one voxel into a caller-supplied workspace (zero allocation past the
    /// first call — for the rayon voxel loop).
    pub fn fit_voxel_into(&self, signal: &[f64], ws: &mut ConshMsmtWorkspace) -> ConshVoxelResult {
        self.solve_with(Some(&self.icls), &self.ridge_path, signal, ws)
    }

    /// Fit one voxel whose measurements were taken with their own gradient
    /// table — e.g. the per-voxel effective table of a gradient-deviation field
    /// ([`crate::graddev`]). `bvals` / `dirs` are in the original gradient order,
    /// `dirs` in the fit's frame; `H` and its solvers are rebuilt for this table
    /// against the plan's grid, responses, constraint and ridge settings.
    /// Every b must lie within the plan's grid (see [`ConshConfig::grid_bmax`]).
    pub fn fit_voxel_table_into(
        &self,
        bvals: &[f64],
        dirs: &[[f64; 3]],
        signal: &[f64],
        ws: &mut ConshMsmtWorkspace,
    ) -> ConshVoxelResult {
        assert_eq!(bvals.len(), self.n_aug, "table length must equal n_aug");
        let h = self.voxel.assemble_h(&self.grid, bvals, dirs);
        let (single, path) = self.voxel.solvers(h, self.ridge_path.is_empty());
        self.solve_with(single.as_ref(), &path, signal, ws)
    }

    fn solve_with(
        &self,
        single: Option<&IclsSolver>,
        ridge_path: &[(f64, IclsSolver)],
        signal: &[f64],
        ws: &mut ConshMsmtWorkspace,
    ) -> ConshVoxelResult {
        assert_eq!(signal.len(), self.n_aug, "signal length must equal n_aug");
        let ConshMsmtWorkspace { icls: icls_ws, signal: sig, resid } = ws;
        for (dst, &s) in sig.iter_mut().zip(signal.iter()) {
            *dst = s;
        }
        let (x, diag) = if ridge_path.len() >= 2 {
            // Per-voxel L2-anchored ridge: pick the largest λ whose data residual
            // stays within (1 + slack) of the reference (λ = path[0]) fit. All
            // solvers share the same forward H, so residuals are comparable.
            let mut ref_rss = f64::INFINITY;
            let mut best: Option<(DVector<f64>, IclsDiagnostics)> = None;
            for (i, (_lam, solver)) in ridge_path.iter().enumerate() {
                let (xk, dk) = solver.solve_into(sig, icls_ws);
                resid.gemv(1.0, solver.h(), &xk, 0.0); // resid = H·xk
                *resid -= &*sig;                        // resid = H·xk − s
                let rss = resid.norm_squared();
                if i == 0 {
                    ref_rss = rss;
                }
                if rss <= (1.0 + self.anchor_slack) * ref_rss {
                    best = Some((xk, dk)); // ascending λ ⇒ ends at largest acceptable
                }
            }
            best.unwrap_or_else(|| ridge_path[0].1.solve_into(sig, icls_ws))
        } else {
            single.expect("single solver present without a ridge path").solve_into(sig, icls_ws)
        };
        let mut c_wm = DVector::<f64>::zeros(self.n_sh_wm);
        c_wm.copy_from_slice(&x.as_slice()[self.wm_col.0..self.wm_col.0 + self.wm_col.1]);
        ConshVoxelResult {
            c_wm,
            c_gm: x[self.gm_col],
            c_csf: x[self.csf_col],
            c_floor: self.floor_col.map(|c| x[c]),
            iterations: diag.iterations,
            converged: diag.converged,
        }
    }
}

/// Per-volume continuous-b MSMT outputs.
pub struct ConshVolumeResult {
    /// WM fODF SH coefficient field, shape (X, Y, Z, n_sh_wm).
    pub wm: Array4<f32>,
    /// GM compartment, shape (X, Y, Z, 1).
    pub gm: Array4<f32>,
    /// CSF compartment, shape (X, Y, Z, 1).
    pub csf: Array4<f32>,
    /// Flat noise-floor compartment, shape (X, Y, Z, 1). `None` when
    /// [`ConshConfig::fit_noise_floor`] is false.
    pub floor: Option<Array4<f32>>,
    /// Per-voxel "ICLS converged" flag (only when diagnostics requested).
    pub converged: Option<Array3<u8>>,
    /// Per-voxel ICLS iteration count (only when diagnostics requested).
    pub iterations: Option<Array3<u32>>,
}

/// Fit continuous-b MSMT to every masked voxel of `dwi`. Signals are
/// S0-normalized per voxel (mean b=0) before the fit, matching the S0=1-scale
/// response from [`crate::multitissue::response_estimation::estimate_consh_responses`].
/// With [`ConshConfig::quantitative`] the fitted coefficients are multiplied back
/// by `S0` afterwards, putting the output in absolute (AFD) signal units.
/// `on_voxel` fires once per completed voxel (progress heartbeats); it runs on
/// rayon workers, so it must be `Sync`.
pub fn fit_volume_consh<F>(
    dwi: &DwiData,
    responses: &ConshResponses,
    cfg: &ConshConfig,
    diagnostics: bool,
    on_voxel: F,
) -> Result<ConshVolumeResult>
where
    F: Fn() + Sync,
{
    fit_volume_consh_graddev(dwi, responses, cfg, None, diagnostics, on_voxel)
}

/// [`fit_volume_consh`] with an optional gradient-deviation field: every voxel
/// whose `T` is not the identity is fit with its own effective table
/// (`Tᵀg` directions and `b·|Tᵀg|²` b-values, see [`crate::graddev`]), so the
/// correction acts on the signal model rather than on the fitted fODF. The
/// grid must reach the largest effective b (set [`ConshConfig::grid_bmax`]
/// from [`VoxelGradDev::max_effective_b`]) and should interpolate in b
/// ([`ConshConfig::b_interp`]); identity voxels reuse the nominal solver.
pub fn fit_volume_consh_graddev<F>(
    dwi: &DwiData,
    responses: &ConshResponses,
    cfg: &ConshConfig,
    graddev: Option<&VoxelGradDev>,
    diagnostics: bool,
    on_voxel: F,
) -> Result<ConshVolumeResult>
where
    F: Fn() + Sync,
{
    let plan = ConshMsmtPlan::build(&dwi.gtab, responses, cfg)?;
    let s = dwi.data.shape();
    let (nx, ny, nz, nt) = (s[0], s[1], s[2], s[3]);
    if nt != plan.n_aug {
        return Err(CsDmriError::Dimension(format!(
            "consh: DWI has {nt} volumes but plan expects {}",
            plan.n_aug
        )));
    }
    let b0: Vec<usize> = (0..nt).filter(|&i| dwi.gtab.bvals[i] <= dwi.gtab.b0_threshold).collect();
    let n_sh_wm = plan.n_sh_wm;

    struct Out {
        wm: DVector<f64>,
        gm: f64,
        csf: f64,
        floor: Option<f64>,
        converged: u8,
        iters: u32,
    }

    let results = voxel_loop::run_init(
        &dwi.mask,
        on_voxel,
        || (plan.workspace(), Vec::<f64>::with_capacity(nt), Vec::<f64>::new(), Vec::<[f64; 3]>::new()),
        |(ws, buf, bvals, dirs), x, y, z| {
            let view = dwi.data.slice(ndarray::s![x, y, z, ..]);
            let s0 = if b0.is_empty() {
                0.0
            } else {
                b0.iter().map(|&i| view[i] as f64).sum::<f64>() / b0.len() as f64
            };
            buf.clear();
            if s0 > 0.0 {
                buf.extend(view.iter().map(|&v| v as f64 / s0));
            } else {
                buf.extend(std::iter::repeat_n(0.0, nt));
            }
            let r = match graddev {
                Some(gd) if !gd.is_identity(x, y, z) => {
                    gd.effective_table_into(dwi, x, y, z, bvals, dirs);
                    plan.fit_voxel_table_into(bvals, dirs, buf, ws)
                }
                _ => plan.fit_voxel_into(buf, ws),
            };
            // Quantitative (AFD) mode: undo the per-voxel S0 normalization so the
            // coefficients are in absolute signal units. The fit solved
            // `FOD ∗ R = S/S0`, so `FOD · S0` deconvolves the absolute `S`.
            // The noise floor is scaled with the tissues, not left behind: it is
            // documented as living on the same scale as `c_gm` / `c_csf`, and it
            // is the one compartment that is genuinely constant in absolute units
            // — so in this mode its map should come out roughly spatially flat.
            let scale = if cfg.quantitative { s0 } else { 1.0 };
            Out {
                wm: r.c_wm * scale,
                gm: r.c_gm * scale,
                csf: r.c_csf * scale,
                floor: r.c_floor.map(|f| f * scale),
                converged: if r.converged { 1 } else { 0 },
                iters: r.iterations as u32,
            }
        },
    );

    let mut wm = Array4::<f32>::zeros((nx, ny, nz, n_sh_wm));
    let mut gm = Array4::<f32>::zeros((nx, ny, nz, 1));
    let mut csf = Array4::<f32>::zeros((nx, ny, nz, 1));
    let mut floor = cfg
        .fit_noise_floor
        .then(|| Array4::<f32>::zeros((nx, ny, nz, 1)));
    let mut converged = diagnostics.then(|| Array3::<u8>::zeros((nx, ny, nz)));
    let mut iters = diagnostics.then(|| Array3::<u32>::zeros((nx, ny, nz)));
    for ((x, y, z), o) in results {
        for k in 0..n_sh_wm {
            wm[(x, y, z, k)] = o.wm[k] as f32;
        }
        gm[(x, y, z, 0)] = o.gm as f32;
        csf[(x, y, z, 0)] = o.csf as f32;
        if let (Some(a), Some(v)) = (floor.as_mut(), o.floor) {
            a[(x, y, z, 0)] = v as f32;
        }
        if let Some(a) = converged.as_mut() {
            a[(x, y, z)] = o.converged;
        }
        if let Some(a) = iters.as_mut() {
            a[(x, y, z)] = o.iters;
        }
    }
    Ok(ConshVolumeResult { wm, gm, csf, floor, converged, iterations: iters })
}

/// Build the joint non-negativity constraint `A x ≥ 0`:
///   - one row selecting `c_csf ≥ 0`,
///   - one row selecting `c_gm ≥ 0`,
///   - one row selecting `c_floor ≥ 0` (when the floor compartment is enabled),
///   - one row per sphere direction: WM fODF amplitude `Y(d)·c_wm ≥ 0`.
fn build_msmt_constraint(
    csf_col: usize,
    gm_col: usize,
    floor_col: Option<usize>,
    wm_col: (usize, usize),
    n_cols: usize,
    lmax_wm: usize,
) -> DMatrix<f64> {
    let dirs = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
    let y = sh2amp_cart(dirs, lmax_wm);
    let n_dirs = dirs.len();
    let (wm0, wm_w) = wm_col;
    let n_iso = 2 + usize::from(floor_col.is_some());
    let mut a = DMatrix::<f64>::zeros(n_iso + n_dirs, n_cols);
    a[(0, csf_col)] = 1.0;
    a[(1, gm_col)] = 1.0;
    if let Some(fc) = floor_col {
        a[(2, fc)] = 1.0;
    }
    for i in 0..n_dirs {
        for j in 0..wm_w {
            a[(n_iso + i, wm0 + j)] = y[(i, j)] as f64;
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multitissue::forward::PerShellPredictor;
    use approx::assert_abs_diff_eq;

    /// A shelled gtab: `n_b0` b=0 volumes + `dwi_bs` DWI shells, each on the
    /// same `n_dir` directions strided from the dsistudio sphere.
    fn shelled_gtab(n_b0: usize, dwi_bs: &[f64], n_dir: usize) -> GradientTable {
        let sphere = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
        let step = (sphere.len() / n_dir).max(1);
        let dirs: Vec<[f64; 3]> = sphere
            .iter()
            .step_by(step)
            .take(n_dir)
            .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
            .collect();
        let mut bvals = vec![0.0; n_b0];
        let mut bvecs: Vec<[f64; 3]> = vec![[0.0; 3]; n_b0];
        for &b in dwi_bs {
            for d in &dirs {
                bvals.push(b);
                bvecs.push(*d);
            }
        }
        GradientTable::new(bvals, bvecs, Some(0.05), Some(0.012), None).unwrap()
    }

    /// WM zonal response rows for the given lmax, a stick-ish decay per b.
    fn wm_rows(bs: &[f64], lmax: usize) -> Vec<Vec<f64>> {
        let n = lmax / 2 + 1;
        bs.iter()
            .map(|&b| {
                if b <= 1e-6 {
                    let mut r = vec![0.0; n];
                    r[0] = 1.0;
                    r
                } else {
                    // arbitrary smooth-ish decaying anisotropic response
                    (0..n)
                        .map(|l| {
                            let ll = (2 * l) as f64;
                            (-b * 0.7e-3).exp() * (-(ll) * b * 1.5e-4).exp()
                                * if l == 0 { 1.0 } else { (-1.0_f64).powi(l as i32) * 0.4 / (1.0 + ll) }
                        })
                        .collect()
                }
            })
            .collect()
    }

    fn iso_rows(bs: &[f64], d: f64) -> Vec<Vec<f64>> {
        bs.iter().map(|&b| vec![(-b * d).exp()]).collect()
    }

    fn dense_responses(grid: &DenseBGrid, lmax: usize) -> ConshResponses {
        ConshResponses {
            wm: TissueResponse { coeffs: wm_rows(&grid.bs, lmax), lmax },
            gm: TissueResponse { coeffs: iso_rows(&grid.bs, 1.1e-3), lmax: 0 },
            csf: TissueResponse { coeffs: iso_rows(&grid.bs, 3.0e-3), lmax: 0 },
        }
    }

    #[test]
    fn grouping_partitions_all_gradients_and_aligns_rows() {
        let gtab = shelled_gtab(6, &[1000.0, 2000.0, 3000.0], 30);
        let grid = DenseBGrid::for_gtab(&gtab, 50.0);
        let plans = build_dense_b_plans(&gtab, &grid);
        assert_eq!(plans.len(), grid.len(), "one plan per grid point");
        // aug_indices across plans partition 0..n_grads exactly.
        let mut seen = vec![false; gtab.n_grads()];
        let mut count = 0;
        for p in &plans {
            for &i in &p.aug_indices {
                assert!(!seen[i], "index {i} appears twice");
                seen[i] = true;
                count += 1;
            }
        }
        assert_eq!(count, gtab.n_grads());
        assert!(seen.iter().all(|&s| s));
        // b=1000 gradients land at grid index 20 (1000/50), b=2000 at 40, etc.
        assert!(!plans[20].aug_indices.is_empty(), "b=1000 -> grid 20");
        assert!(!plans[40].aug_indices.is_empty(), "b=2000 -> grid 40");
        assert!(!plans[60].aug_indices.is_empty(), "b=3000 -> grid 60");
        assert_eq!(plans[0].aug_indices.len(), 6, "6 b0 volumes at grid 0");
    }

    #[test]
    fn dense_b_equals_shelled_forward() {
        // The correctness proof: on a shelled acquisition, the continuous-b
        // forward operator is IDENTICAL to the per-shell one (so continuous-b
        // MSMT == shelled MSMT — the dipy corr=1.0000 result, exactly).
        let lmax = 8;
        let gtab = shelled_gtab(6, &[1000.0, 2000.0, 3000.0], 40);
        let grid = DenseBGrid::for_gtab(&gtab, 50.0);
        let dense = dense_responses(&grid, lmax);

        // Dense-b forward.
        let dense_plans = build_dense_b_plans(&gtab, &grid);
        let dense_h = build_h(
            gtab.n_grads(),
            &dense_plans,
            &[
                TissueSlot { response: &dense.csf, lmax: 0 },
                TissueSlot { response: &dense.gm, lmax: 0 },
                TissueSlot { response: &dense.wm, lmax },
            ],
        );

        // Shelled forward: 4 shells (b0,1000,2000,3000), response rows pulled
        // from the dense grid at those exact b-values.
        let shells = gtab.shells(50.0);
        let shell_bs: Vec<f64> = shells.iter().map(|s| s.b).collect();
        let idx: Vec<usize> = shell_bs.iter().map(|&b| grid.nearest(b)).collect();
        let pick = |r: &TissueResponse| TissueResponse {
            coeffs: idx.iter().map(|&k| r.coeffs[k].clone()).collect(),
            lmax: r.lmax,
        };
        let (wm_s, gm_s, csf_s) = (pick(&dense.wm), pick(&dense.gm), pick(&dense.csf));
        let mut shell_plans = Vec::new();
        for shell in &shells {
            let dirs: Vec<[f32; 3]> = shell
                .indices
                .iter()
                .map(|&i| {
                    let v = gtab.bvecs[i];
                    [v[0] as f32, v[1] as f32, v[2] as f32]
                })
                .collect();
            shell_plans.push(ShellPlan { b: shell.b, dirs_ras: dirs, aug_indices: shell.indices.clone() });
        }
        let shell_h = build_h(
            gtab.n_grads(),
            &shell_plans,
            &[
                TissueSlot { response: &csf_s, lmax: 0 },
                TissueSlot { response: &gm_s, lmax: 0 },
                TissueSlot { response: &wm_s, lmax },
            ],
        );

        assert_eq!(dense_h.h.shape(), shell_h.h.shape());
        let diff = (&dense_h.h - &shell_h.h).amax();
        assert!(diff < 1e-9, "dense-b vs shelled forward max diff = {diff}");
    }

    #[test]
    fn recovers_planted_three_tissue_signal_multishell() {
        // Plant a 3-tissue voxel via the forward, fit, and recover the tissue
        // fractions + WM anisotropy on a well-posed multi-shell scheme.
        let lmax = 8;
        let gtab = shelled_gtab(6, &[1000.0, 2000.0, 3000.0], 46);
        let cfg = ConshConfig { lmax_wm: lmax, step: 50.0, ..Default::default() };
        let grid = DenseBGrid::for_gtab(&gtab, cfg.step);
        let resp = dense_responses(&grid, lmax);
        let plan = ConshMsmtPlan::build(&gtab, &resp, &cfg).unwrap();

        // Build a planted signal from known coefficients via the plan's own H.
        let n_cols = plan.n_cols();
        let mut coefs = DVector::<f64>::zeros(n_cols);
        coefs[plan.csf_col] = 0.15;
        coefs[plan.gm_col] = 0.25;
        // WM: isotropic part + an l=2 lobe.
        let mut c_wm = vec![0.0_f64; plan.n_sh_wm];
        c_wm[0] = 0.6 / (4.0 * std::f64::consts::PI).sqrt();
        c_wm[3] = 0.05; // (l=2, m=0)
        for (j, &c) in c_wm.iter().enumerate() {
            coefs[plan.wm_col.0 + j] = c;
        }
        let signal = plan.icls.h() * &coefs;

        let res = plan.fit_voxel(signal.as_slice());
        assert!(res.converged, "ICLS did not converge");
        assert_abs_diff_eq!(res.c_csf, 0.15, epsilon = 1e-3);
        assert_abs_diff_eq!(res.c_gm, 0.25, epsilon = 1e-3);
        assert_abs_diff_eq!(res.c_wm[0], c_wm[0], epsilon = 1e-3);
        assert_abs_diff_eq!(res.c_wm[3], c_wm[3], epsilon = 1e-3);
        // The floor compartment is on by default; on floor-free data it must
        // stay out of the way rather than skim amplitude off the tissues.
        assert_abs_diff_eq!(res.c_floor.unwrap(), 0.0, epsilon = 1e-3);
    }

    #[test]
    fn noise_floor_compartment_absorbs_flat_offset() {
        // Plant a pure-CSF voxel, then add a b-independent offset standing in for
        // the Rician floor. Without the floor column the fit has nowhere to put a
        // flat high-b tail except WM (the only compartment with amplitude left at
        // high b), so the WM fODF inflates. With it, WM stays ~0 and the offset is
        // recovered on the flat column.
        let lmax = 8;
        let gtab = shelled_gtab(6, &[1000.0, 2000.0, 3000.0], 46);
        let grid = DenseBGrid::for_gtab(&gtab, 50.0);
        let resp = dense_responses(&grid, lmax);

        let with_floor = ConshConfig { lmax_wm: lmax, step: 50.0, ..Default::default() };
        let without = ConshConfig { fit_noise_floor: false, ..with_floor.clone() };
        let plan_on = ConshMsmtPlan::build(&gtab, &resp, &with_floor).unwrap();
        let plan_off = ConshMsmtPlan::build(&gtab, &resp, &without).unwrap();
        assert!(plan_on.floor_col.is_some());
        assert!(plan_off.floor_col.is_none());
        assert_eq!(plan_on.n_cols(), plan_off.n_cols() + 1);

        // Pure CSF, plus a flat offset of 0.05 · S0 on every measurement.
        let mut coefs = DVector::<f64>::zeros(plan_off.n_cols());
        coefs[plan_off.csf_col] = 0.28;
        let offset = 0.05;
        let signal: Vec<f64> = (plan_off.icls.h() * &coefs).iter().map(|v| v + offset).collect();

        let off = plan_off.fit_voxel(&signal);
        let on = plan_on.fit_voxel(&signal);

        // Without the floor column the offset leaks into WM...
        assert!(
            off.c_wm[0] > 0.01,
            "expected the flat offset to inflate WM without the floor column, got {}",
            off.c_wm[0]
        );
        // ...and with it, WM is left essentially empty.
        assert!(
            on.c_wm[0] < 0.1 * off.c_wm[0],
            "floor column should absorb the offset: wm_on={} wm_off={}",
            on.c_wm[0],
            off.c_wm[0]
        );
        // The flat column carries the offset (H column = floor_zonal(), so the
        // fitted amplitude times that constant reproduces the planted offset).
        assert_abs_diff_eq!(
            on.c_floor.unwrap() * floor_zonal(),
            offset,
            epsilon = 5e-3
        );
        assert!(on.c_floor.unwrap() > 0.0);
        assert!(off.c_floor.is_none());
    }

    #[test]
    fn consumes_non_shelled_qspace() {
        // A genuinely NON-shelled scheme: b-values spread continuously (like a
        // CS-DSI Cartesian grid), no shell structure. The fit must still run
        // and recover a planted signal — the whole point of continuous-b.
        let sphere = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
        let mut bvals = vec![0.0; 6];
        let mut bvecs: Vec<[f64; 3]> = vec![[0.0; 3]; 6];
        // 100 samples with continuously varying b in [200, 3000].
        for k in 0..100 {
            let b = 200.0 + (2800.0 * k as f64 / 99.0); // 200 … 3000, no shells
            let v = sphere[(k * 7) % sphere.len()];
            bvals.push(b);
            bvecs.push([v[0] as f64, v[1] as f64, v[2] as f64]);
        }
        let gtab = GradientTable::new(bvals, bvecs, Some(0.05), Some(0.012), None).unwrap();
        // Confirm it really is non-shelled (many distinct b-values).
        let n_shells = gtab.shells(50.0).len();
        assert!(n_shells > 10, "expected non-shelled, got {n_shells} pseudo-shells");

        let lmax = 8;
        let cfg = ConshConfig { lmax_wm: lmax, step: 50.0, ..Default::default() };
        let grid = DenseBGrid::for_gtab(&gtab, cfg.step);
        let resp = dense_responses(&grid, lmax);
        let plan = ConshMsmtPlan::build(&gtab, &resp, &cfg).unwrap();

        // Plant via the forward and recover.
        let n_cols = plan.n_cols();
        let mut coefs = DVector::<f64>::zeros(n_cols);
        coefs[plan.csf_col] = 0.1;
        coefs[plan.gm_col] = 0.2;
        coefs[plan.wm_col.0] = 0.7 / (4.0 * std::f64::consts::PI).sqrt();
        coefs[plan.wm_col.0 + 3] = 0.04;
        let signal = plan.icls.h() * &coefs;
        let res = plan.fit_voxel(signal.as_slice());
        assert!(res.converged);
        assert_abs_diff_eq!(res.c_csf, 0.1, epsilon = 5e-3);
        assert_abs_diff_eq!(res.c_gm, 0.2, epsilon = 5e-3);
        assert_abs_diff_eq!(res.c_wm[0], 0.7 / (4.0 * std::f64::consts::PI).sqrt(), epsilon = 5e-3);

        // Sanity: a pure-CSF signal recovers csf-dominant, wm≈0.
        let mut pure = DVector::<f64>::zeros(n_cols);
        pure[plan.csf_col] = 0.5;
        let sig_csf = plan.icls.h() * &pure;
        let r2 = plan.fit_voxel(sig_csf.as_slice());
        assert!(r2.c_csf > 0.4, "c_csf = {}", r2.c_csf);
        assert!(r2.c_wm[0].abs() < 0.05, "wm leaked: {}", r2.c_wm[0]);
        let _ = PerShellPredictor::new(plan.n_aug, &build_dense_b_plans(&gtab, &grid),
            &TissueSlot { response: &resp.wm, lmax }); // predictor builds on dense plans too
    }

    #[test]
    fn quantitative_mode_rescales_every_compartment_by_s0() {
        // Two voxels with IDENTICAL tissue composition but a 3.33x difference in
        // S0 — the situation that per-voxel S/S0 normalization collapses and that
        // AFD quantification needs back. Checks three things:
        //   1. Without --quantitative the two voxels are indistinguishable (the
        //      S/S0 scale really does discard S0).
        //   2. With it, each voxel's coefficients are EXACTLY its S/S0-scale
        //      solution times S0 — the positive-homogeneity property the mode
        //      rests on (scaling the data scales the ICLS solution, because the
        //      objective is homogeneous and the non-negativity cone is invariant).
        //   3. The noise-floor compartment is rescaled with the tissues, not left
        //      behind on the old scale.
        let lmax = 8;
        let gtab = shelled_gtab(6, &[1000.0, 2000.0, 3000.0], 46);
        let cfg_plain = ConshConfig { lmax_wm: lmax, step: 50.0, ..Default::default() };
        let cfg_quant = ConshConfig { quantitative: true, ..cfg_plain.clone() };
        let grid = DenseBGrid::for_gtab(&gtab, cfg_plain.step);
        let resp = dense_responses(&grid, lmax);
        let plan = ConshMsmtPlan::build(&gtab, &resp, &cfg_plain).unwrap();

        let mut coefs = DVector::<f64>::zeros(plan.n_cols());
        coefs[plan.csf_col] = 0.15;
        coefs[plan.gm_col] = 0.25;
        coefs[plan.wm_col.0] = 0.60;
        coefs[plan.wm_col.0 + 3] = 0.03; // l=2, m=0 — contributes nothing at b=0
        let mut unit_signal = plan.icls.h() * &coefs;
        // Renormalize so the planted b=0 mean is exactly 1.0. Then feeding
        // `unit_signal * S0` makes the fit's own `mean_b0` equal S0 exactly, which
        // is what makes `quant == plain * S0` a clean assertion below. Done from
        // the signal rather than from the coefficients so the test does not bake
        // in an assumption about the l=0 SH normalization convention.
        let b0_idx: Vec<usize> = (0..gtab.n_grads())
            .filter(|&i| gtab.bvals[i] <= gtab.b0_threshold)
            .collect();
        let b0_mean =
            b0_idx.iter().map(|&i| unit_signal[i]).sum::<f64>() / b0_idx.len() as f64;
        assert!(b0_mean > 0.0);
        unit_signal /= b0_mean;

        let s0s = [1000.0_f32, 300.0_f32];
        let n_grads = gtab.n_grads();
        let mut data = Array4::<f32>::zeros((2, 1, 1, n_grads));
        for (v, &s0) in s0s.iter().enumerate() {
            for i in 0..n_grads {
                data[(v, 0, 0, i)] = unit_signal[i] as f32 * s0;
            }
        }
        let dwi = DwiData::from_table(
            data,
            Array3::from_elem((2, 1, 1), true),
            gtab,
            crate::qspace::BvecFrame::WorldRas,
            std::path::PathBuf::from("synthetic"),
        );

        let plain = fit_volume_consh(&dwi, &resp, &cfg_plain, false, || {}).unwrap();
        let quant = fit_volume_consh(&dwi, &resp, &cfg_quant, false, || {}).unwrap();

        // (1) S/S0 scale is blind to S0: both voxels fit identically. Tolerance is
        // 1e-4 against a ~0.28-scale signal (the two voxels' S/S0 differ in the
        // last f32 bits, since the volume is stored as f32 at different S0).
        for j in 0..plan.n_sh_wm {
            assert_abs_diff_eq!(plain.wm[(0, 0, 0, j)], plain.wm[(1, 0, 0, j)], epsilon = 1e-4);
        }
        assert_abs_diff_eq!(plain.gm[(0, 0, 0, 0)], plain.gm[(1, 0, 0, 0)], epsilon = 1e-4);

        // (2) + (3) Quantitative == plain * S0, for every compartment.
        for (v, &s0) in s0s.iter().enumerate() {
            for j in 0..plan.n_sh_wm {
                assert_abs_diff_eq!(
                    quant.wm[(v, 0, 0, j)],
                    plain.wm[(v, 0, 0, j)] * s0,
                    epsilon = 1e-3 * s0
                );
            }
            assert_abs_diff_eq!(
                quant.gm[(v, 0, 0, 0)], plain.gm[(v, 0, 0, 0)] * s0, epsilon = 1e-3 * s0);
            assert_abs_diff_eq!(
                quant.csf[(v, 0, 0, 0)], plain.csf[(v, 0, 0, 0)] * s0, epsilon = 1e-3 * s0);
            let (qf, pf) = (quant.floor.as_ref().unwrap(), plain.floor.as_ref().unwrap());
            assert_abs_diff_eq!(qf[(v, 0, 0, 0)], pf[(v, 0, 0, 0)] * s0, epsilon = 1e-3 * s0);
        }

        // And the quantitative outputs now genuinely differ between the two
        // voxels, in proportion to S0 — the contrast AFD needs.
        let ratio = quant.wm[(0, 0, 0, 0)] / quant.wm[(1, 0, 0, 0)];
        assert_abs_diff_eq!(ratio, s0s[0] / s0s[1], epsilon = 1e-3);
    }

    // ---- gradient deviation (graddev) ----

    const ID_AFFINE: [[f64; 4]; 4] =
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

    /// Rotation by `deg` about `axis` composed with a stretch: a deviation with
    /// both a direction change and a direction-dependent b change.
    fn deviation(deg: f64, axis: [f64; 3], stretch: [f64; 3]) -> nalgebra::Matrix3<f64> {
        let ax = nalgebra::Unit::new_normalize(nalgebra::Vector3::new(axis[0], axis[1], axis[2]));
        let r = nalgebra::Rotation3::from_axis_angle(&ax, deg.to_radians()).into_inner();
        r * nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(stretch[0], stretch[1], stretch[2]))
    }

    fn uniform_field(dims: [usize; 3], t: &nalgebra::Matrix3<f64>) -> odx_rs::GradDevField {
        let n = dims[0] * dims[1] * dims[2];
        let row_major: Vec<f32> = (0..9).map(|k| t[(k / 3, k % 3)] as f32).collect();
        let data = (0..n).flat_map(|_| row_major.iter().copied()).collect();
        odx_rs::GradDevField::from_parts(dims, ID_AFFINE, data, odx_rs::IdentityPolicy::Included).unwrap()
    }

    /// WM coefficients of `a + b·(v·u)²` (non-negative, l ≤ 2, axis `u`) in the
    /// MRtrix basis, padded to `n_sh` — a feasible fODF with a known axis.
    fn axial_wm(u: [f64; 3], a: f64, b: f64, n_sh: usize) -> Vec<f64> {
        let dirs = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
        let y = sh2amp_cart(&dirs, 2);
        let un = nalgebra::Vector3::new(u[0], u[1], u[2]).normalize();
        let f = DVector::from_iterator(
            dirs.len(),
            dirs.iter().map(|d| {
                let c = d[0] as f64 * un[0] + d[1] as f64 * un[1] + d[2] as f64 * un[2];
                a + b * c * c
            }),
        );
        let y = DMatrix::from_fn(y.nrows(), y.ncols(), |i, j| y[(i, j)] as f64);
        let c = y.svd(true, true).solve(&f, 1e-12).unwrap();
        let mut out = vec![0.0; n_sh];
        out[..c.len()].copy_from_slice(c.as_slice());
        out
    }

    #[test]
    fn b_interp_is_nearest_on_grid_points_and_linear_between() {
        let lmax = 8;
        let gtab = shelled_gtab(4, &[1000.0, 2000.0], 30);
        let grid = DenseBGrid::for_gtab(&gtab, 50.0);
        let resp = dense_responses(&grid, lmax);
        let nearest = ConshMsmtPlan::build(&gtab, &resp, &ConshConfig::default()).unwrap();
        let interp =
            ConshMsmtPlan::build(&gtab, &resp, &ConshConfig { b_interp: true, ..Default::default() }).unwrap();
        assert_eq!(nearest.icls.h(), interp.icls.h(), "grid-aligned shells must not change");

        // Off-grid: b = 1020 is 40% of the way from row 20 (1000) to 21 (1050).
        let d = [[0.36, 0.48, 0.8]];
        let row = |b: f64| interp.voxel.assemble_h(&interp.grid, &[b], &d);
        let mid = row(1020.0);
        let want = row(1000.0) * 0.6 + row(1050.0) * 0.4;
        assert!((mid - want).amax() < 1e-12);
    }

    #[test]
    fn identity_graddev_reproduces_the_nominal_fit_exactly() {
        let lmax = 8;
        let gtab = shelled_gtab(4, &[1000.0, 2000.0, 3000.0], 30);
        let cfg = ConshConfig { b_interp: true, ..Default::default() };
        let grid = DenseBGrid::for_gtab(&gtab, cfg.step);
        let resp = dense_responses(&grid, lmax);
        let plan = ConshMsmtPlan::build(&gtab, &resp, &cfg).unwrap();
        let mut x = DVector::<f64>::zeros(plan.n_cols());
        x[plan.csf_col] = 0.1;
        x[plan.gm_col] = 0.2;
        for (j, c) in axial_wm([0.2, 0.5, 0.84], 0.05, 0.3, plan.n_sh_wm).into_iter().enumerate() {
            x[plan.wm_col.0 + j] = c;
        }
        let sig = plan.icls.h() * &x;
        let n = gtab.n_grads();
        let data = Array4::from_shape_fn((2, 1, 1, n), |(v, _, _, i)| (sig[i] * (1.0 + v as f64)) as f32);
        let dwi = DwiData::from_table(
            data,
            Array3::from_elem((2, 1, 1), true),
            gtab,
            crate::qspace::BvecFrame::WorldRas,
            "synthetic".into(),
        );
        let gd = VoxelGradDev::new(uniform_field([2, 1, 1], &nalgebra::Matrix3::identity()), &dwi, &ID_AFFINE)
            .unwrap();
        let plain = fit_volume_consh(&dwi, &resp, &cfg, false, || {}).unwrap();
        let with = fit_volume_consh_graddev(&dwi, &resp, &cfg, Some(&gd), false, || {}).unwrap();
        assert_eq!(plain.wm, with.wm);
        assert_eq!(plain.gm, with.gm);
        assert_eq!(plain.csf, with.csf);
    }

    #[test]
    fn graddev_fit_recovers_a_planted_voxel_the_nominal_fit_cannot() {
        // The voxel's signal is generated with its true (deviated) gradients:
        // 6° rotation plus a few-percent stretch. Fitting with the effective
        // table recovers the planted coefficients; the nominal table does not.
        let lmax = 8;
        let gtab = shelled_gtab(4, &[1000.0, 2000.0, 3000.0], 40);
        let n = gtab.n_grads();
        let t = deviation(6.0, [0.3, -0.5, 0.8], [1.06, 0.97, 1.02]);
        let dwi = DwiData::from_table(
            Array4::zeros((1, 1, 1, n)),
            Array3::from_elem((1, 1, 1), true),
            gtab.clone(),
            crate::qspace::BvecFrame::WorldRas,
            "synthetic".into(),
        );
        let gd = VoxelGradDev::new(uniform_field([1, 1, 1], &t), &dwi, &ID_AFFINE).unwrap();
        let bmax_eff = gd.max_effective_b(&dwi, &dwi.mask);
        assert!(bmax_eff > 3000.0 * 1.05, "stretch must raise the effective bmax");

        let cfg = ConshConfig { b_interp: true, grid_bmax: Some(bmax_eff), ..Default::default() };
        let grid = DenseBGrid::for_gtab_min_bmax(&gtab, cfg.step, cfg.grid_bmax);
        let resp = dense_responses(&grid, lmax);
        let plan = ConshMsmtPlan::build(&gtab, &resp, &cfg).unwrap();

        let mut x = DVector::<f64>::zeros(plan.n_cols());
        x[plan.csf_col] = 0.05;
        x[plan.gm_col] = 0.1;
        for (j, c) in axial_wm([0.6, 0.0, 0.8], 0.02, 0.4, plan.n_sh_wm).into_iter().enumerate() {
            x[plan.wm_col.0 + j] = c;
        }
        let (mut bvals, mut dirs) = (Vec::new(), Vec::new());
        gd.effective_table_into(&dwi, 0, 0, 0, &mut bvals, &mut dirs);
        let signal = plan.voxel.assemble_h(&plan.grid, &bvals, &dirs) * &x;

        let wm_err = |r: &ConshVoxelResult| {
            let want = x.rows(plan.wm_col.0, plan.n_sh_wm);
            (&r.c_wm - want).norm() / want.norm()
        };
        let mut ws = plan.workspace();
        let corrected = plan.fit_voxel_table_into(&bvals, &dirs, signal.as_slice(), &mut ws);
        let nominal = plan.fit_voxel_into(signal.as_slice(), &mut ws);
        assert!(wm_err(&corrected) < 1e-4, "graddev fit WM rel. error {}", wm_err(&corrected));
        assert_abs_diff_eq!(corrected.c_gm, 0.1, epsilon = 1e-5);
        assert_abs_diff_eq!(corrected.c_csf, 0.05, epsilon = 1e-5);
        assert!(wm_err(&nominal) > 0.05, "nominal fit should be visibly wrong, got {}", wm_err(&nominal));

        // The volume path takes the same per-voxel route.
        let mut dwi = dwi;
        for (i, v) in signal.iter().enumerate() {
            dwi.data[(0, 0, 0, i)] = *v as f32;
        }
        let s0 = signal.rows(0, 4).mean();
        let vol = fit_volume_consh_graddev(&dwi, &resp, &cfg, Some(&gd), false, || {}).unwrap();
        for j in 0..plan.n_sh_wm {
            // f32 DWI storage and the S0 rescale perturb the solve at ~1e-5.
            assert_abs_diff_eq!(vol.wm[(0, 0, 0, j)] as f64, corrected.c_wm[j] / s0, epsilon = 1e-4);
        }
    }

    #[test]
    fn ridge_path_runs_per_voxel_with_a_deviated_table() {
        let lmax = 8;
        let gtab = shelled_gtab(4, &[1000.0, 2000.0], 30);
        let t = deviation(4.0, [1.0, 0.0, 0.0], [1.03, 1.0, 0.98]);
        let dwi = DwiData::from_table(
            Array4::zeros((1, 1, 1, gtab.n_grads())),
            Array3::from_elem((1, 1, 1), true),
            gtab.clone(),
            crate::qspace::BvecFrame::WorldRas,
            "synthetic".into(),
        );
        let gd = VoxelGradDev::new(uniform_field([1, 1, 1], &t), &dwi, &ID_AFFINE).unwrap();
        let cfg = ConshConfig {
            b_interp: true,
            grid_bmax: Some(gd.max_effective_b(&dwi, &dwi.mask)),
            wm_ridge_path: vec![0.0, 0.5, 1.0],
            l2_anchor_slack: 0.05,
            ..Default::default()
        };
        let grid = DenseBGrid::for_gtab_min_bmax(&gtab, cfg.step, cfg.grid_bmax);
        let plan = ConshMsmtPlan::build(&gtab, &dense_responses(&grid, lmax), &cfg).unwrap();
        let (mut bvals, mut dirs) = (Vec::new(), Vec::new());
        gd.effective_table_into(&dwi, 0, 0, 0, &mut bvals, &mut dirs);
        let mut x = DVector::<f64>::zeros(plan.n_cols());
        x[plan.gm_col] = 0.2;
        for (j, c) in axial_wm([0.0, 1.0, 0.0], 0.02, 0.3, plan.n_sh_wm).into_iter().enumerate() {
            x[plan.wm_col.0 + j] = c;
        }
        let signal = plan.voxel.assemble_h(&plan.grid, &bvals, &dirs) * &x;
        let r = plan.fit_voxel_table_into(&bvals, &dirs, signal.as_slice(), &mut plan.workspace());
        assert!(r.converged);
        assert!(r.c_wm.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn responses_longer_than_the_grid_are_used_as_is_and_shorter_ones_rejected() {
        // Row k is b = k·step regardless of grid length, so a wider-range
        // response (e.g. written under --graddev) fits a nominal grid unchanged.
        let lmax = 8;
        let gtab = shelled_gtab(4, &[1000.0, 2000.0], 30);
        let cfg = ConshConfig::default();
        let grid = DenseBGrid::for_gtab(&gtab, cfg.step);
        let exact = dense_responses(&grid, lmax);
        let longer = dense_responses(&DenseBGrid::new(3500.0, cfg.step, 2.0 * cfg.step), lmax);
        assert!(longer.wm.n_shells() > grid.len());
        let a = ConshMsmtPlan::build(&gtab, &exact, &cfg).unwrap();
        let b = ConshMsmtPlan::build(&gtab, &longer, &cfg).unwrap();
        assert_eq!(a.icls.h(), b.icls.h());

        let short_cfg = ConshConfig { grid_bmax: Some(3500.0), ..Default::default() };
        let err = ConshMsmtPlan::build(&gtab, &exact, &short_cfg).err().expect("too few rows must fail");
        assert!(err.to_string().contains("need at least"), "{err}");
    }
}
