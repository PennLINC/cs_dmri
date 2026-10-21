// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dhollander 2016 unsupervised three-tissue response function estimation.
//!
//! Implements the algorithm from
//! Dhollander, Raffelt & Connelly, *"Unsupervised 3-tissue response function
//! estimation from single-shell or multi-shell diffusion MR data without a
//! co-registered T1 image"*, ISMRM Workshop 2016. Closes the SS3T loop —
//! cs_dmri can now go from raw DWI → response → SS3T → mtnormalise without
//! external MRtrix tools.
//!
//! Outputs three MRtrix-compatible `.txt` files (WM single-fiber, GM, CSF)
//! that drop straight into [`super::ss3t`].
//!
//! ## Algorithm
//!
//! 1. **DTI fit** every masked voxel (typically via [`crate::dti`]'s RESTORE).
//! 2. **Tissue selection**, one of two implementations:
//!    - Default — the staged, signal-decay-metric algorithm MRtrix actually
//!      runs, in [`super::dhollander`]. Use this for anything compared against
//!      MRtrix.
//!    - Legacy (`legacy_selection = true`) — the original threshold triple:
//!      CSF = top `md_csf_pct` of MD, WM single-fiber = FA above
//!      `fa_wm_threshold` with principal-eigenvalue dominance
//!      `λ₁ / ((λ₂ + λ₃)/2)` above `fiber_dominance_ratio`, GM = the rest.
//!      Kept so existing results stay reproducible; its CSF class admits
//!      partial-volume voxels, which depresses the CSF response amplitude.
//! 3. **Per-shell SH averaging**:
//!    - GM/CSF (isotropic, lmax=0): mean b=0 signal and mean DWI signal.
//!    - WM (anisotropic, lmax up to `lmax_wm`): fit zonal SH per voxel
//!      relative to each voxel's principal direction, then average across
//!      voxels. The b=0 row uses only `r₀` (isotropic at b=0).
//!
//! ## License posture
//!
//! The averaging in this file is cs_dmri's own, written from the Dhollander
//! 2016 ISMRM abstract and sanity-checked against dipy's response estimators
//! (BSD-3, permissive). The staged voxel selection it now calls IS a port of
//! MRtrix's MPL-2.0 `dwi2response dhollander`, and is quarantined in
//! [`super::dhollander`] — MPL-2.0 is file-scoped, so that file carries the
//! MPL notice and this one does not become MPL-covered by calling it. Keep it
//! that way: do not inline MRtrix-derived logic here.

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use ndarray::Array3;

use crate::dti::DtiVolumeResult;
use crate::io::dwi::DwiData;
use crate::math::assoc_legendre;
use crate::multitissue::dhollander::select_voxels;
pub use crate::multitissue::dhollander::{DhollanderSelectConfig, DhollanderStageCounts};
use crate::graddev::VoxelGradDev;
use crate::multitissue::consh::{ConshResponses, DenseBGrid};
use crate::multitissue::response::TissueResponse;
use crate::multitissue::response_dki::{DkiResponseFit, DkiResponseFits, dki_dense_rows, fit_dki_response};
use crate::multitissue::response_smooth::{SmoothResponseConfig, fit_dense_rows};
use crate::qspace::Shell;
use crate::{CsDmriError, Result};

/// Configuration for Dhollander tissue selection.
#[derive(Debug, Clone, Copy)]
pub struct DhollanderConfig {
    /// Use the original threshold-triple selection instead of the staged
    /// MRtrix one. Only `fa_wm_threshold`, `fiber_dominance_ratio` and
    /// `md_csf_pct` apply in that mode; only `stages` applies otherwise.
    pub legacy_selection: bool,
    /// Tuning for the staged (default) selection.
    pub stages: DhollanderSelectConfig,
    /// FA above this counts a voxel as WM single-fiber candidate. Legacy only.
    pub fa_wm_threshold: f64,
    /// Eigenvalue ratio `λ₁ / ((λ₂ + λ₃) / 2)`: above this counts as
    /// "single-fiber" (suppresses crossings). Set to 0 to skip the test.
    pub fiber_dominance_ratio: f64,
    /// Top-N percent of MD values are CSF candidates (default 2.5%).
    pub md_csf_pct: f64,
    /// Maximum SH order for the WM response.
    pub lmax_wm: usize,
}

impl Default for DhollanderConfig {
    fn default() -> Self {
        Self {
            legacy_selection: false,
            stages: DhollanderSelectConfig::default(),
            fa_wm_threshold: 0.7,
            fiber_dominance_ratio: 2.0,
            md_csf_pct: 2.5,
            lmax_wm: 8,
        }
    }
}

/// Per-tissue voxel selection counts (diagnostic).
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct ResponseEstimationDiagnostics {
    pub n_brain_voxels: usize,
    pub n_wm_voxels: usize,
    pub n_gm_voxels: usize,
    pub n_csf_voxels: usize,
    /// MD threshold (top N% cutoff) actually used. `NaN` under the staged
    /// selection, which does not threshold on MD at all.
    pub md_csf_threshold: f64,
    /// Per-stage voxel counts, when the staged selection ran.
    pub stages: Option<DhollanderStageCounts>,
    /// Median b=0 signal over the WM single-fiber voxels the response was fit
    /// from — the natural per-scan `DWI_ref` for AFD quantification.
    ///
    /// The WM single-fiber population is the best available "same tissue in
    /// every subject" reference: it is the same voxels whose S0 the response was
    /// normalized by, so expressing fiber density relative to it is what makes
    /// one unit of AFD mean the same thing across scans (Smith et al. 2022,
    /// "global intensity normalization"). 0.0 if no WM voxels were selected.
    pub wm_b0_median: f64,
    /// Fitted parameters and goodness of fit of the parametric responses, when
    /// [`ResponseEstimator::Dki`] produced them.
    pub dki: Option<DkiResponseFits>,
}

/// The three estimated responses + selection diagnostics.
#[derive(Debug, Clone)]
pub struct Ss3tResponseEstimate {
    pub wm: TissueResponse,
    pub gm: TissueResponse,
    pub csf: TissueResponse,
    pub diagnostics: ResponseEstimationDiagnostics,
    /// Optional per-tissue selection masks (one per tissue, true where the
    /// voxel was used to fit that tissue's response).
    pub wm_mask: Array3<u8>,
    pub gm_mask: Array3<u8>,
    pub csf_mask: Array3<u8>,
}

/// Estimate WM/GM/CSF responses from DWI + DTI fit + brain mask.
pub fn estimate_responses(
    dwi: &DwiData,
    dti: &DtiVolumeResult,
    cfg: &DhollanderConfig,
) -> Result<Ss3tResponseEstimate> {
    let s = dwi.data.shape();
    let (nx, ny, nz, _) = (s[0], s[1], s[2], s[3]);

    let shells = dwi.gtab.shells(50.0);
    if shells.len() != 2 || shells[0].b > dwi.gtab.b0_threshold || shells[1].b <= dwi.gtab.b0_threshold {
        return Err(CsDmriError::Other(format!(
            "response estimation: need exactly one b=0 + one DWI shell, got bvals {:?}",
            shells.iter().map(|s| s.b).collect::<Vec<_>>()
        )));
    }
    let b0_shell = &shells[0];
    let dwi_shell = &shells[1];

    // -------- Steps 1-2: collect brain voxels + Dhollander tissue selection. --------
    let brain = collect_brain_voxels(dwi, dti);
    let n_brain = brain.len();
    if n_brain == 0 {
        return Err(CsDmriError::Other(
            "response estimation: no valid brain voxels".into(),
        ));
    }
    let TissueSelection {
        wm: wm_voxels,
        gm: gm_voxels,
        csf: csf_voxels,
        wm_mask,
        gm_mask,
        csf_mask,
        md_threshold: md_csf_threshold,
        stages,
    } = select_tissues(dwi, dti, &brain, (nx, ny, nz), cfg)?;

    // -------- Step 3: per-tissue per-shell averaging. --------
    // GM and CSF are isotropic (lmax=0): just average b=0 and DWI signals.
    let gm_response = average_isotropic_response(dwi, &gm_voxels, b0_shell, dwi_shell);
    let csf_response = average_isotropic_response(dwi, &csf_voxels, b0_shell, dwi_shell);
    // WM is anisotropic on the DWI shell: fit zonal SH per voxel along the
    // principal eigenvector, average across voxels.
    let wm_response = average_wm_response(dwi, &wm_voxels, b0_shell, dwi_shell, cfg.lmax_wm);

    let diagnostics = ResponseEstimationDiagnostics {
        n_brain_voxels: n_brain,
        n_wm_voxels: wm_voxels.len(),
        n_gm_voxels: gm_voxels.len(),
        n_csf_voxels: csf_voxels.len(),
        md_csf_threshold,
        stages,
        wm_b0_median: median_b0(dwi, &wm_voxels, dwi.gtab.b0_threshold),
        dki: None,
    };

    Ok(Ss3tResponseEstimate {
        wm: wm_response,
        gm: gm_response,
        csf: csf_response,
        diagnostics,
        wm_mask,
        gm_mask,
        csf_mask,
    })
}

/// `λ₁ / mean(λ₂, λ₃)`. Returns 0 if `λ₂ + λ₃ ≤ 0` (degenerate tensor).
fn fiber_dominance(lambdas: &[f64; 3]) -> f64 {
    let denom = (lambdas[1] + lambdas[2]) * 0.5;
    if denom <= 0.0 { 0.0 } else { lambdas[0] / denom }
}

/// Averaged signal at b=0 and the DWI shell across the supplied voxel set.
/// Returns a `TissueResponse` with `lmax = 0` (single zonal coefficient per
/// shell). MRtrix convention stores the b=0 row's coefficient and the DWI
/// row's coefficient on separate lines.
fn average_isotropic_response(
    dwi: &DwiData,
    voxels: &[&Voxel],
    b0_shell: &Shell,
    dwi_shell: &Shell,
) -> TissueResponse {
    let n = voxels.len() as f64;
    let mut b0_sum = 0.0_f64;
    let mut dwi_sum = 0.0_f64;
    for v in voxels {
        let (x, y, z) = v.idx;
        let view = dwi.data.slice(ndarray::s![x, y, z, ..]);
        for &i in &b0_shell.indices {
            b0_sum += view[i] as f64;
        }
        for &i in &dwi_shell.indices {
            dwi_sum += view[i] as f64;
        }
    }
    let b0_mean = b0_sum / (n * b0_shell.indices.len() as f64);
    let dwi_mean = dwi_sum / (n * dwi_shell.indices.len() as f64);
    // Convert mean signal to zonal SH coefficient: r_0 = mean_signal · sqrt(4π).
    // (Y_00 = 1/sqrt(4π); the SH expansion of a constant on the sphere is
    // r_0 · Y_00. For a constant value c, r_0 = c · sqrt(4π).)
    let factor = (4.0 * PI).sqrt();
    TissueResponse {
        coeffs: vec![vec![b0_mean * factor], vec![dwi_mean * factor]],
        lmax: 0,
    }
}

/// Anisotropic WM response: per-voxel zonal SH along the principal
/// eigenvector, averaged across voxels.
fn average_wm_response(
    dwi: &DwiData,
    voxels: &[&Voxel],
    b0_shell: &Shell,
    dwi_shell: &Shell,
    lmax: usize,
) -> TissueResponse {
    let n_zonal = lmax / 2 + 1;

    // b=0 is isotropic — single coefficient per voxel, averaged.
    let mut b0_sum = 0.0_f64;
    let mut dwi_sum_zonal = vec![0.0_f64; n_zonal];

    let factor = (4.0 * PI).sqrt();

    for v in voxels {
        let (x, y, z) = v.idx;
        let view = dwi.data.slice(ndarray::s![x, y, z, ..]);
        // b=0 mean for this voxel.
        let mut b0_voxel = 0.0_f64;
        for &i in &b0_shell.indices {
            b0_voxel += view[i] as f64;
        }
        b0_voxel /= b0_shell.indices.len() as f64;
        b0_sum += b0_voxel;

        // DWI shell: build cos_angles[i] = g_i · principal_dir, fit zonal SH.
        let pd = v.principal_dir;
        let pd_norm = (pd[0] * pd[0] + pd[1] * pd[1] + pd[2] * pd[2]).sqrt().max(1e-12);
        let pd_unit = [pd[0] / pd_norm, pd[1] / pd_norm, pd[2] / pd_norm];

        let mut cos_angles = Vec::with_capacity(dwi_shell.indices.len());
        let mut amplitudes = Vec::with_capacity(dwi_shell.indices.len());
        for &i in &dwi_shell.indices {
            let g = dwi.gtab.bvecs[i];
            let dot = g[0] * pd_unit[0] + g[1] * pd_unit[1] + g[2] * pd_unit[2];
            cos_angles.push(dot);
            amplitudes.push(view[i] as f64);
        }
        let voxel_zonal = fit_zonal_sh(&cos_angles, &amplitudes, lmax);
        for j in 0..n_zonal {
            dwi_sum_zonal[j] += voxel_zonal[j];
        }
    }

    let n = voxels.len() as f64;
    let b0_mean = b0_sum / n;
    let mut dwi_mean_zonal = vec![0.0_f64; n_zonal];
    for j in 0..n_zonal {
        dwi_mean_zonal[j] = dwi_sum_zonal[j] / n;
    }

    // b=0 row: only r_0 nonzero, others zero (response file convention).
    let mut b0_row = vec![0.0_f64; n_zonal];
    b0_row[0] = b0_mean * factor;

    TissueResponse {
        coeffs: vec![b0_row, dwi_mean_zonal],
        lmax,
    }
}

/// Fit zonal SH coefficients `[r₀, r₂, r₄, …]` (m=0 only) to amplitudes
/// sampled at directions whose polar angle from a reference axis has
/// cosine `cos_angles[i]`.
///
/// Uses the MRtrix even-real-SH convention:
/// `Y_l_0(θ) = sqrt((2l+1)/(4π)) · P_l(cos θ)`.
///
/// Returns vec of length `lmax/2 + 1`.
fn fit_zonal_sh(cos_angles: &[f64], amplitudes: &[f64], lmax: usize) -> Vec<f64> {
    use nalgebra::{DMatrix, DVector};
    let n = cos_angles.len();
    let n_zonal = lmax / 2 + 1;
    let mut x = DMatrix::<f64>::zeros(n, n_zonal);
    for i in 0..n {
        let c = cos_angles[i];
        for j in 0..n_zonal {
            let l = 2 * j;
            let p_l = legendre(l, c);
            x[(i, j)] = ((2.0 * l as f64 + 1.0) / (4.0 * PI)).sqrt() * p_l;
        }
    }
    let y = DVector::<f64>::from_row_slice(amplitudes);
    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;
    let chol = nalgebra::Cholesky::new(xtx).expect(
        "zonal SH design matrix must be PD — check that cos_angles span enough range",
    );
    let r = chol.solve(&xty);
    (0..n_zonal).map(|j| r[j]).collect()
}

/// Legendre polynomial P_l(x) (unnormalized) for non-negative l.
fn legendre(l: usize, x: f64) -> f64 {
    // assoc_legendre(l, m=0, x) returns P_l^0(x) = P_l(x) without the
    // Condon-Shortley phase since m=0. Reuse the existing helper.
    assoc_legendre(l as u32, 0, x)
}

/// Per-voxel features used by Dhollander tissue selection.
#[derive(Debug, Clone, Copy)]
struct Voxel {
    idx: (usize, usize, usize),
    fa: f64,
    md: f64,
    lambdas: [f64; 3],
    principal_dir: [f64; 3],
}

/// Result of tissue selection: per-tissue voxel lists + masks + the CSF MD cut.
struct TissueSelection<'a> {
    wm: Vec<&'a Voxel>,
    gm: Vec<&'a Voxel>,
    csf: Vec<&'a Voxel>,
    wm_mask: Array3<u8>,
    gm_mask: Array3<u8>,
    csf_mask: Array3<u8>,
    md_threshold: f64,
    stages: Option<DhollanderStageCounts>,
}

/// Collect DTI-derived features for every masked brain voxel (sorted eigenvalues
/// + principal direction). Shared by the single-shell and continuous-b paths.
fn collect_brain_voxels(dwi: &DwiData, dti: &DtiVolumeResult) -> Vec<Voxel> {
    let s = dwi.data.shape();
    let (nx, ny, nz) = (s[0], s[1], s[2]);
    let mut brain = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if !dwi.mask[(x, y, z)] {
                    continue;
                }
                let fa = dti.fa[(x, y, z)] as f64;
                let md = dti.md[(x, y, z)] as f64;
                if dti.s0[(x, y, z)] == 0.0 && fa == 0.0 && md == 0.0 {
                    continue;
                }
                let d = nalgebra::Matrix3::new(
                    dti.tensor[(x, y, z, 0)] as f64,
                    dti.tensor[(x, y, z, 1)] as f64,
                    dti.tensor[(x, y, z, 2)] as f64,
                    dti.tensor[(x, y, z, 1)] as f64,
                    dti.tensor[(x, y, z, 3)] as f64,
                    dti.tensor[(x, y, z, 4)] as f64,
                    dti.tensor[(x, y, z, 2)] as f64,
                    dti.tensor[(x, y, z, 4)] as f64,
                    dti.tensor[(x, y, z, 5)] as f64,
                );
                let eigen = d.symmetric_eigen();
                let mut evals = [eigen.eigenvalues[0], eigen.eigenvalues[1], eigen.eigenvalues[2]];
                evals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                brain.push(Voxel {
                    idx: (x, y, z),
                    fa,
                    md,
                    lambdas: evals,
                    principal_dir: [
                        dti.principal_dir[(x, y, z, 0)] as f64,
                        dti.principal_dir[(x, y, z, 1)] as f64,
                        dti.principal_dir[(x, y, z, 2)] as f64,
                    ],
                });
            }
        }
    }
    brain
}

/// Pick the WM single-fiber / GM / CSF training populations.
///
/// Dispatches to the staged MRtrix algorithm ([`super::dhollander`]) unless
/// `cfg.legacy_selection` is set.
fn select_tissues<'a>(
    dwi: &DwiData,
    dti: &DtiVolumeResult,
    brain: &'a [Voxel],
    shape: (usize, usize, usize),
    cfg: &DhollanderConfig,
) -> Result<TissueSelection<'a>> {
    if cfg.legacy_selection {
        return select_tissues_legacy(brain, shape, cfg);
    }
    let selection = select_voxels(dwi, dti, &cfg.stages)?;
    // The staged selection works on the brain mask directly; map its masks back
    // onto the `Voxel` records (which carry the tensor features the WM response
    // fit needs). A selected voxel with no record was dropped by
    // `collect_brain_voxels` as degenerate — rare, and harmless to skip.
    let (nx, ny, nz) = shape;
    let mut by_index: std::collections::HashMap<(usize, usize, usize), &'a Voxel> =
        std::collections::HashMap::with_capacity(brain.len());
    for v in brain {
        by_index.insert(v.idx, v);
    }
    let gather = |mask: &Array3<bool>| -> (Vec<&'a Voxel>, Array3<u8>) {
        let mut out = Vec::new();
        let mut out_mask = Array3::<u8>::zeros((nx, ny, nz));
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    if mask[(x, y, z)] {
                        if let Some(v) = by_index.get(&(x, y, z)) {
                            out.push(*v);
                            out_mask[(x, y, z)] = 1;
                        }
                    }
                }
            }
        }
        (out, out_mask)
    };
    let (wm, wm_mask) = gather(&selection.sfwm);
    let (gm, gm_mask) = gather(&selection.gm);
    let (csf, csf_mask) = gather(&selection.csf);
    if wm.is_empty() || gm.is_empty() || csf.is_empty() {
        return Err(CsDmriError::Other(format!(
            "response estimation: staged tissue selection produced an empty class \
             (WM={}, GM={}, CSF={}) — stage counts {:?}",
            wm.len(),
            gm.len(),
            csf.len(),
            selection.counts
        )));
    }
    Ok(TissueSelection {
        wm,
        gm,
        csf,
        wm_mask,
        gm_mask,
        csf_mask,
        md_threshold: f64::NAN,
        stages: Some(selection.counts),
    })
}

/// The original threshold-triple selection: CSF = top-`md_csf_pct` MD;
/// WM single-fiber = high FA + eigenvalue dominance; GM = the rest.
fn select_tissues_legacy<'a>(
    brain: &'a [Voxel],
    shape: (usize, usize, usize),
    cfg: &DhollanderConfig,
) -> Result<TissueSelection<'a>> {
    let (nx, ny, nz) = shape;
    let n_brain = brain.len();
    let mut sorted_md: Vec<f64> = brain.iter().map(|v| v.md).collect();
    sorted_md.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let csf_quantile = (1.0 - cfg.md_csf_pct / 100.0).clamp(0.0, 1.0);
    let csf_idx = ((n_brain as f64 - 1.0) * csf_quantile).round() as usize;
    let md_threshold = sorted_md[csf_idx.min(n_brain - 1)];

    let mut wm_mask = Array3::<u8>::zeros((nx, ny, nz));
    let mut gm_mask = Array3::<u8>::zeros((nx, ny, nz));
    let mut csf_mask = Array3::<u8>::zeros((nx, ny, nz));
    let (mut wm, mut gm, mut csf) = (Vec::new(), Vec::new(), Vec::new());
    for v in brain {
        if v.md > md_threshold {
            csf_mask[v.idx] = 1;
            csf.push(v);
        } else if v.fa > cfg.fa_wm_threshold
            && fiber_dominance(&v.lambdas) > cfg.fiber_dominance_ratio
        {
            wm_mask[v.idx] = 1;
            wm.push(v);
        } else {
            gm_mask[v.idx] = 1;
            gm.push(v);
        }
    }
    if wm.is_empty() {
        return Err(CsDmriError::Other(format!(
            "response estimation: no WM single-fiber voxels — lower --fa-wm-threshold ({}) or --fiber-dominance-ratio ({})",
            cfg.fa_wm_threshold, cfg.fiber_dominance_ratio
        )));
    }
    if gm.is_empty() || csf.is_empty() {
        return Err(CsDmriError::Other(
            "response estimation: GM or CSF tissue class empty — check thresholds and brain mask".into(),
        ));
    }
    Ok(TissueSelection {
        wm,
        gm,
        csf,
        wm_mask,
        gm_mask,
        csf_mask,
        md_threshold,
        stages: None,
    })
}

// ================= Continuous-b response estimation =================
//
// Generalizes the per-shell Dhollander averaging above to a smooth-in-b
// (moving-shell) estimate on a dense b-grid, so it feeds
// [`crate::multitissue::consh`] directly for non-shelled CS-DSI. The tissue
// SELECTION is identical to the single-shell path (shared via
// `collect_brain_voxels` / `select_tissues`); only the per-tissue aggregation
// changes: instead of one mean per shell, a Gaussian-in-b weighted zonal-SH fit
// at every grid point.

/// Gaussian-in-b weighted zonal-SH least squares (weighted variant of
/// [`fit_zonal_sh`]). Weights fold in as `√w` on both design rows and target.
fn fit_zonal_sh_weighted(cos_angles: &[f64], amps: &[f64], w: &[f64], lmax: usize) -> Vec<f64> {
    use nalgebra::{DMatrix, DVector};
    let n = cos_angles.len();
    let n_zonal = lmax / 2 + 1;
    let mut x = DMatrix::<f64>::zeros(n, n_zonal);
    let mut y = DVector::<f64>::zeros(n);
    for i in 0..n {
        let sw = w[i].max(0.0).sqrt();
        let c = cos_angles[i];
        for j in 0..n_zonal {
            let l = 2 * j;
            x[(i, j)] = ((2.0 * l as f64 + 1.0) / (4.0 * PI)).sqrt() * legendre(l, c) * sw;
        }
        y[i] = amps[i] * sw;
    }
    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;
    match nalgebra::Cholesky::new(xtx) {
        Some(ch) => {
            let r = ch.solve(&xty);
            (0..n_zonal).map(|j| r[j]).collect()
        }
        None => vec![0.0; n_zonal], // ill-posed grid point (too little b-support)
    }
}

/// Configuration for continuous-b response estimation.
#[derive(Debug, Clone, Copy)]
pub struct ConshResponseConfig {
    /// Tissue-selection thresholds (shared with the single-shell estimator).
    pub dhollander: DhollanderConfig,
    /// Dense b-grid step in s/mm² (must match the deconvolution's `step`).
    pub step: f64,
    /// Gaussian kernel width in b for the moving-shell fit.
    pub sigma_b: f64,
    /// Grid points whose total sample weight is below this fraction of the
    /// per-point average carry the previous point's coefficients forward.
    pub min_weight_frac: f64,
    /// Extend the dense-b grid to at least this b (must match the
    /// deconvolution's [`crate::multitissue::consh::ConshConfig::grid_bmax`]).
    pub grid_bmax: Option<f64>,
    /// How the pooled samples become dense-b rows. Default
    /// [`ResponseEstimator::Kernel`], which matches MRtrix's per-shell responses
    /// on shelled data.
    pub estimator: ResponseEstimator,
    /// Settings of [`ResponseEstimator::Smooth`].
    pub smooth: SmoothResponseConfig,
}

/// Dense-b row estimator for [`estimate_consh_responses`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResponseEstimator {
    /// Each row fit on its own from a Gaussian-in-b (`sigma_b`) window of the
    /// samples; thin rows carry the previous row forward. Faithful on densely
    /// b-sampled data; a staircase on shelled data and noisy or carried-forward
    /// between the b-values of sparse non-shelled schemes.
    #[default]
    Kernel,
    /// All rows fit jointly: linear-in-b interpolation of the data term,
    /// second-difference smoothness across rows, and constraints exact at any
    /// lmax (spherical mean ≥ 0 and non-increasing in b, WM zonal sign pattern).
    /// See [`crate::multitissue::response_smooth`]. Robust to sparse non-shelled
    /// (CS-DSI) and gradient-deviated sampling.
    Smooth,
    /// Parametric: one axially symmetric DKI + offset response per tissue
    /// (7 parameters WM, 4 GM/CSF) fit to all samples at once, non-increasing
    /// in b on the whole grid — Morez et al. 2021. Rows are its zonal
    /// projection at each grid b. See [`crate::multitissue::response_dki`].
    Dki,
}

impl Default for ConshResponseConfig {
    fn default() -> Self {
        Self {
            dhollander: DhollanderConfig::default(),
            step: 50.0,
            sigma_b: 150.0,
            min_weight_frac: 0.05,
            grid_bmax: None,
            estimator: ResponseEstimator::Kernel,
            smooth: SmoothResponseConfig::default(),
        }
    }
}

/// Estimate dense-b (continuous-in-b) WM/GM/CSF responses from raw DWI, ready
/// for [`crate::multitissue::consh::ConshMsmtPlan`].
///
/// All three tissues are scaled by ONE common per-scan reference — the
/// Dhollander single-fiber-WM median b=0, the same value recorded as
/// `ReferenceB0Signal` in the ODX sidecar — so the relative amplitudes BETWEEN
/// tissues are preserved (CSF b=0 ≈ 2.6× WM, the T2/PD contrast), exactly as
/// MRtrix's `amp2response`/`dwi2response dhollander` responses preserve them.
/// The result is `amp2response`'s raw-signal response divided by one per-scan
/// constant: identical to MRtrix up to a global scale (which the final
/// `mtnormalise` pins), and gain-free (receiver gain cancels in the ratio).
///
/// Consequence: the WM b=0 coefficient is only ≈ `√4π` (mean/median ratio of
/// the single-fiber b=0 population), not exactly `√4π`. Do not assert
/// exact anchoring downstream; per-scan magnitude differences between response
/// files are real signal (PD·T2 and shape), compensated when pooling
/// (`consh-response mean`, matching `responsemean`'s default) and by
/// `mtnormalise` at fit time.
pub fn estimate_consh_responses(
    dwi: &DwiData,
    dti: &DtiVolumeResult,
    cfg: &ConshResponseConfig,
) -> Result<(ConshResponses, ResponseEstimationDiagnostics)> {
    estimate_consh_responses_graddev(dwi, dti, cfg, None)
}

/// [`estimate_consh_responses`] with an optional gradient-deviation field: each
/// selected voxel's samples are pooled at its effective b-values and gradient
/// directions (see [`crate::graddev`]), so the responses describe the tissue
/// rather than the scanner's deviation, and the grid's upper rows (b above the
/// nominal bmax, reached where `|Tᵀg| > 1`) are estimated from real samples.
/// `dti` should come from a fit with the same effective tables, so the
/// principal directions the WM samples are measured against share their frame.
pub fn estimate_consh_responses_graddev(
    dwi: &DwiData,
    dti: &DtiVolumeResult,
    cfg: &ConshResponseConfig,
    graddev: Option<&VoxelGradDev>,
) -> Result<(ConshResponses, ResponseEstimationDiagnostics)> {
    let factor = (4.0 * PI).sqrt();
    let s = dwi.data.shape();
    let shape = (s[0], s[1], s[2]);
    let brain = collect_brain_voxels(dwi, dti);
    if brain.is_empty() {
        return Err(CsDmriError::Other("response estimation: no valid brain voxels".into()));
    }
    let sel = select_tissues(dwi, dti, &brain, shape, &cfg.dhollander)?;

    let grid = DenseBGrid::for_gtab_min_bmax(&dwi.gtab, cfg.step, cfg.grid_bmax);
    let ng = grid.len();
    let (mut tb, mut td) = (Vec::new(), Vec::new());
    let lmax = cfg.dhollander.lmax_wm;
    let n_zonal = lmax / 2 + 1;
    let b0_thr = dwi.gtab.b0_threshold;

    // The common per-scan reference every tissue is divided by. Median (not
    // mean) so one bright outlier voxel cannot shift the whole scan's scale.
    let ref_b0 = median_b0(dwi, &sel.wm, b0_thr);
    if ref_b0 <= 0.0 {
        return Err(CsDmriError::Other(
            "response estimation: WM single-fiber median b=0 is not positive — cannot scale responses".into(),
        ));
    }

    // DW gradient indices shared across voxels. The parametric fit also takes
    // the b=0 volumes: they fix its S0 + C, where the other estimators pin the
    // b=0 row to the measured mean instead.
    let dw: Vec<usize> = (0..dwi.gtab.n_grads()).filter(|&i| dwi.gtab.bvals[i] > b0_thr).collect();
    let all: Vec<usize> = (0..dwi.gtab.n_grads()).collect();
    let pooled = if cfg.estimator == ResponseEstimator::Dki { &all } else { &dw };
    let grid_top = grid.bs.last().copied().unwrap_or(0.0);

    // ---- WM: pool (b, cos, S/ref) then weighted zonal fit per grid point ----
    let mut pool_b = Vec::new();
    let mut pool_cos = Vec::new();
    let mut pool_y = Vec::new();
    let mut pool_s0 = Vec::new();
    let mut wm_b0_sum = 0.0_f64;
    let mut wm_b0_n = 0.0_f64;
    for v in &sel.wm {
        let (x, y, z) = v.idx;
        let view = dwi.data.slice(ndarray::s![x, y, z, ..]);
        let s0 = mean_b0(&view, dwi, b0_thr);
        if s0 <= 0.0 {
            // Validity guard only (dead or degenerate voxel): the signal is
            // scaled by the common `ref_b0`, not by s0.
            continue;
        }
        wm_b0_sum += s0;
        wm_b0_n += 1.0;
        let pd = unit(v.principal_dir);
        voxel_table(dwi, graddev, v.idx, &mut tb, &mut td);
        for &i in pooled {
            let g = td[i];
            pool_b.push(tb[i]);
            pool_cos.push((g[0] * pd[0] + g[1] * pd[1] + g[2] * pd[2]).abs()); // axial symmetry
            pool_y.push(view[i] as f64 / ref_b0);
            pool_s0.push(s0);
        }
    }
    // Measured WM b=0 amplitude on the common-reference scale: mean b=0 of the
    // training voxels over the median (≈ 1, exactly 1 only when mean == median).
    let wm_b0_coeff = if wm_b0_n > 0.0 { wm_b0_sum / wm_b0_n / ref_b0 * factor } else { factor };
    let mut wm_dki = None;
    let wm_rows = match cfg.estimator {
        ResponseEstimator::Dki => {
            let (y, w) = per_voxel_shape(&pool_y, &pool_s0, ref_b0, wm_b0_coeff / factor);
            let fit = fit_dki_response(&pool_b, Some(&pool_cos), &y, Some(&w), grid_top)?;
            let rows = dki_dense_rows(&fit.params, &grid, lmax);
            wm_dki = Some(fit);
            rows
        }
        ResponseEstimator::Smooth => {
            let mut b0_row = vec![0.0; n_zonal];
            b0_row[0] = wm_b0_coeff;
            let (y, w) = per_voxel_shape(&pool_y, &pool_s0, ref_b0, wm_b0_coeff / factor);
            fit_dense_rows(&grid, b0_thr, &b0_row, lmax, &pool_b, Some(&pool_cos), &y, Some(&w), &cfg.smooth)?
        }
        ResponseEstimator::Kernel => {
            let mut wm_rows: Vec<Vec<f64>> = Vec::with_capacity(ng);
            let mut w_buf = vec![0.0_f64; pool_b.len()];
            let avg_weight_target = pool_b.len() as f64 / ng.max(1) as f64;
            for k in 0..ng {
                if grid.bs[k] <= b0_thr {
                    let mut row = vec![0.0; n_zonal];
                    row[0] = wm_b0_coeff; // WM b=0 isotropic, measured (≈ √4π)
                    wm_rows.push(row);
                    continue;
                }
                let bg = grid.bs[k];
                let mut wsum = 0.0;
                for (i, &bi) in pool_b.iter().enumerate() {
                    let wt = (-0.5 * ((bi - bg) / cfg.sigma_b).powi(2)).exp();
                    w_buf[i] = wt;
                    wsum += wt;
                }
                if wsum < cfg.min_weight_frac * avg_weight_target {
                    wm_rows.push(wm_rows.last().cloned().unwrap_or_else(|| vec![0.0; n_zonal]));
                    continue;
                }
                wm_rows.push(fit_zonal_sh_weighted(&pool_cos, &pool_y, &w_buf, lmax));
            }
            wm_rows
        }
    };

    // ---- GM / CSF: weighted mean S/ref per grid point ----
    let (gm_rows, gm_dki) = iso_dense_rows(dwi, graddev, &sel.gm, &grid, pooled, cfg, b0_thr, factor, ref_b0)?;
    let (csf_rows, csf_dki) = iso_dense_rows(dwi, graddev, &sel.csf, &grid, pooled, cfg, b0_thr, factor, ref_b0)?;
    let dki = match (wm_dki, gm_dki, csf_dki) {
        (Some(wm), Some(gm), Some(csf)) => Some(DkiResponseFits { wm, gm, csf }),
        _ => None,
    };

    let diagnostics = ResponseEstimationDiagnostics {
        n_brain_voxels: brain.len(),
        n_wm_voxels: sel.wm.len(),
        n_gm_voxels: sel.gm.len(),
        n_csf_voxels: sel.csf.len(),
        md_csf_threshold: sel.md_threshold,
        stages: sel.stages,
        wm_b0_median: ref_b0,
        dki,
    };
    let _ = (&sel.wm_mask, &sel.gm_mask, &sel.csf_mask);
    Ok((
        ConshResponses {
            wm: TissueResponse { coeffs: wm_rows, lmax },
            gm: TissueResponse { coeffs: gm_rows, lmax: 0 },
            csf: TissueResponse { coeffs: csf_rows, lmax: 0 },
        },
        diagnostics,
    ))
}

/// Median of the per-voxel mean b=0 signal over `voxels`. Over the WM
/// single-fiber voxels this is both the `DWI_ref` recorded in
/// [`ResponseEstimationDiagnostics`] and the common per-scan reference every
/// tissue's training signal is divided by in [`estimate_consh_responses`]; the
/// median rather than the mean so one bright outlier (a vessel, a mis-selected
/// CSF voxel) cannot shift the reference the whole scan is expressed against.
fn median_b0(dwi: &DwiData, voxels: &[&Voxel], b0_thr: f64) -> f64 {
    if voxels.is_empty() {
        return 0.0;
    }
    let mut s0s: Vec<f64> = voxels
        .iter()
        .map(|v| {
            let (x, y, z) = v.idx;
            mean_b0(&dwi.data.slice(ndarray::s![x, y, z, ..]), dwi, b0_thr)
        })
        .collect();
    let mid = s0s.len() / 2;
    s0s.select_nth_unstable_by(mid, f64::total_cmp);
    s0s[mid]
}

fn mean_b0(
    view: &ndarray::ArrayView1<f32>,
    dwi: &DwiData,
    b0_thr: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for i in 0..dwi.gtab.n_grads() {
        if dwi.gtab.bvals[i] <= b0_thr {
            sum += view[i] as f64;
            n += 1.0;
        }
    }
    if n > 0.0 { sum / n } else { 0.0 }
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Samples and weights for the smooth estimator: each voxel's own `S/S0` (the
/// shape of the decay) times one tissue-level scale (its mean `S0` over the
/// common reference — the b=0 row), weighted by `(S0/ref)²`. That is the
/// least-squares fit of `S/ref = (S0/ref) · shape`: every voxel still counts in
/// proportion to its signal, as with `S/ref` samples, but its brightness no
/// longer enters the shape. With `S/ref` samples, brightness and b-dependence
/// mix whenever rows are informed by different voxels — and the off-shell
/// b-values a gradient-deviation field creates come only from the periphery,
/// where S0 differs. The kernel estimator keeps `S/ref` (MRtrix parity).
fn per_voxel_shape(pool_y: &[f64], pool_s0: &[f64], ref_b0: f64, tissue_scale: f64) -> (Vec<f64>, Vec<f64>) {
    pool_y
        .iter()
        .zip(pool_s0)
        .map(|(&y, &s0)| (y * ref_b0 / s0 * tissue_scale, (s0 / ref_b0).powi(2)))
        .unzip()
}

/// The gradient table voxel `idx` was measured with: its graddev effective
/// table when a non-identity field is given, else the nominal one.
fn voxel_table(
    dwi: &DwiData,
    graddev: Option<&VoxelGradDev>,
    (x, y, z): (usize, usize, usize),
    bvals: &mut Vec<f64>,
    dirs: &mut Vec<[f64; 3]>,
) {
    match graddev {
        Some(gd) if !gd.is_identity(x, y, z) => gd.effective_table_into(dwi, x, y, z, bvals, dirs),
        _ => {
            bvals.clone_from(&dwi.gtab.bvals);
            dirs.clone_from(&dwi.gtab.bvecs);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn iso_dense_rows(
    dwi: &DwiData,
    graddev: Option<&VoxelGradDev>,
    voxels: &[&Voxel],
    grid: &DenseBGrid,
    dw: &[usize],
    cfg: &ConshResponseConfig,
    b0_thr: f64,
    factor: f64,
    ref_b0: f64,
) -> Result<(Vec<Vec<f64>>, Option<DkiResponseFit>)> {
    let mut pool_b = Vec::new();
    let mut pool_y = Vec::new();
    let mut pool_s0 = Vec::new();
    let mut b0_sum = 0.0_f64;
    let mut b0_n = 0.0_f64;
    let (mut tb, mut td) = (Vec::new(), Vec::new());
    for v in voxels {
        let (x, y, z) = v.idx;
        let view = dwi.data.slice(ndarray::s![x, y, z, ..]);
        let s0 = mean_b0(&view, dwi, b0_thr);
        if s0 <= 0.0 {
            // Validity guard only — the common ref_b0 sets the scale.
            continue;
        }
        b0_sum += s0;
        b0_n += 1.0;
        voxel_table(dwi, graddev, v.idx, &mut tb, &mut td);
        for &i in dw {
            pool_b.push(tb[i]);
            pool_y.push(view[i] as f64 / ref_b0);
            pool_s0.push(s0);
        }
    }
    // Measured tissue b=0 amplitude relative to the common WM reference — this
    // is where the inter-tissue T2/PD contrast (CSF ≈ 2.6× WM) survives.
    let b0_coeff = if b0_n > 0.0 { b0_sum / b0_n / ref_b0 * factor } else { factor };
    match cfg.estimator {
        ResponseEstimator::Smooth => {
            let (y, w) = per_voxel_shape(&pool_y, &pool_s0, ref_b0, b0_coeff / factor);
            let rows = fit_dense_rows(grid, b0_thr, &[b0_coeff], 0, &pool_b, None, &y, Some(&w), &cfg.smooth)?;
            return Ok((rows, None));
        }
        ResponseEstimator::Dki => {
            let (y, w) = per_voxel_shape(&pool_y, &pool_s0, ref_b0, b0_coeff / factor);
            let top = grid.bs.last().copied().unwrap_or(0.0);
            let fit = fit_dki_response(&pool_b, None, &y, Some(&w), top)?;
            return Ok((dki_dense_rows(&fit.params, grid, 0), Some(fit)));
        }
        ResponseEstimator::Kernel => {}
    }
    let mut rows = Vec::with_capacity(grid.len());
    for k in 0..grid.len() {
        if grid.bs[k] <= b0_thr {
            rows.push(vec![b0_coeff]);
            continue;
        }
        let bg = grid.bs[k];
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &bi) in pool_b.iter().enumerate() {
            let wt = (-0.5 * ((bi - bg) / cfg.sigma_b).powi(2)).exp();
            num += wt * pool_y[i];
            den += wt;
        }
        let mean = if den > 1e-12 { num / den } else { rows.last().map_or(0.0, |r: &Vec<f64>| r[0] / factor) };
        rows.push(vec![mean * factor]);
    }
    Ok((rows, None))
}

// ------------------ MRtrix .txt response output ------------------

// ------------------ MRtrix .txt response output ------------------

/// Write a `TissueResponse` to disk in MRtrix `dwi2response` `.txt` format
/// (one row per shell, whitespace-separated zonal SH coefficients).
pub fn write_response_txt(response: &TissueResponse, path: &Path) -> Result<()> {
    fs::write(path, format_response_txt(response))?;
    Ok(())
}

/// The text [`write_response_txt`] writes.
pub fn format_response_txt(response: &TissueResponse) -> String {
    let mut content = String::new();
    content.push_str(&format!(
        "# cs_dmri Dhollander-2016 response (lmax={}, {} shells)\n",
        response.lmax,
        response.n_shells()
    ));
    for row in &response.coeffs {
        // Match MRtrix's `dwi2response` formatting: 15-significant-digits,
        // shortest-by-precision rather than scientific. Rust's `{:e}` is
        // scientific; `{:.15}` is fixed; the closest analog to `%.15g`
        // is to format both and pick the shorter — but for our use case
        // the responses are O(10²)–O(10³) scale, so fixed at 15 digits
        // after decimal is fine and parseable by MRtrix.
        let words: Vec<String> = row.iter().map(|c| format!("{:.15e}", c)).collect();
        content.push_str(&words.join(" "));
        content.push('\n');
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn legendre_matches_known_values() {
        // Known values: P_0(x) = 1, P_2(x) = (3x² − 1) / 2, P_4(x) = (35x⁴ − 30x² + 3)/8
        for x in [0.0_f64, 0.3, 0.7, 1.0] {
            assert_abs_diff_eq!(legendre(0, x), 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(legendre(2, x), (3.0 * x * x - 1.0) / 2.0, epsilon = 1e-10);
            assert_abs_diff_eq!(
                legendre(4, x),
                (35.0 * x.powi(4) - 30.0 * x.powi(2) + 3.0) / 8.0,
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn zonal_sh_fit_recovers_planted_coefficients() {
        // Plant `r = [r₀, r₂, r₄]`, synthesize amplitudes at random cos_angles,
        // then re-fit; check the recovered coefficients match.
        let lmax = 4;
        let r_true = [1.5_f64, -0.8, 0.3];
        let cos_angles: Vec<f64> = (0..50)
            .map(|i| -1.0 + 2.0 * (i as f64) / 49.0)
            .collect();
        let amps: Vec<f64> = cos_angles
            .iter()
            .map(|&c| {
                let mut s = 0.0_f64;
                for j in 0..3 {
                    let l = 2 * j;
                    let p = legendre(l, c);
                    let factor = ((2.0 * l as f64 + 1.0) / (4.0 * PI)).sqrt();
                    s += r_true[j] * factor * p;
                }
                s
            })
            .collect();
        let r_fit = fit_zonal_sh(&cos_angles, &amps, lmax);
        for j in 0..3 {
            assert_abs_diff_eq!(r_fit[j], r_true[j], epsilon = 1e-9);
        }
    }

    fn weighted_zonal_fit_matches_unweighted_on_exact_data() {
        // On exact (noise-free) data, non-uniform weights must not change the
        // recovered coefficients (weighted LSQ of a consistent system).
        let lmax = 4;
        let r_true = [1.2_f64, -0.5, 0.2];
        let cos: Vec<f64> = (0..40).map(|i| -1.0 + 2.0 * i as f64 / 39.0).collect();
        let amps: Vec<f64> = cos
            .iter()
            .map(|&c| {
                (0..3)
                    .map(|j| {
                        let l = 2 * j;
                        r_true[j] * ((2.0 * l as f64 + 1.0) / (4.0 * PI)).sqrt() * legendre(l, c)
                    })
                    .sum()
            })
            .collect();
        let w: Vec<f64> = (0..40).map(|i| 0.1 + (i as f64 % 5.0)).collect();
        let r = fit_zonal_sh_weighted(&cos, &amps, &w, lmax);
        for j in 0..3 {
            assert_abs_diff_eq!(r[j], r_true[j], epsilon = 1e-9);
        }
    }
    fn consh_response_estimator_recovers_planted_wm_response() {
        use crate::io::dwi::DwiData;
        use crate::dti::DtiVolumeResult;
        use crate::qspace::{BvecFrame, GradientTable};
        use ndarray::{Array3, Array4};

        // --- gradient scheme: 6 b0 + 30 dirs at each of b=1000,2000,3000. ---
        let sphere = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
        let dirs: Vec<[f64; 3]> = sphere
            .iter()
            .step_by((sphere.len() / 30).max(1))
            .take(30)
            .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
            .collect();
        let mut bvals = vec![0.0; 6];
        let mut bvecs: Vec<[f64; 3]> = vec![[0.0; 3]; 6];
        for &b in &[1000.0, 2000.0, 3000.0] {
            for d in &dirs {
                bvals.push(b);
                bvecs.push(*d);
            }
        }
        let gtab = GradientTable::new(bvals.clone(), bvecs.clone(), Some(0.043), Some(0.011), None)
            .unwrap();
        let ng = gtab.n_grads();

        // --- planted WM zonal response r_ℓ(b) (S0=1 scale), smooth in b. ---
        let r_true = |b: f64| -> [f64; 3] {
            let e = (-0.7e-3 * b).exp();
            [e, -0.35 * e, 0.12 * e]
        };
        let ylm0 = |l: usize, c: f64| ((2.0 * l as f64 + 1.0) / (4.0 * PI)).sqrt() * legendre(l, c);

        // --- a 40-voxel volume: 8 WM (varied fiber dirs) + 6 CSF + 26 GM. ---
        let n_vox = 40;
        let mut data = Array4::<f32>::zeros((n_vox, 1, 1, ng));
        let mut mask = Array3::<bool>::from_elem((n_vox, 1, 1), true);
        let _ = &mut mask;
        let mut fa = Array3::<f32>::zeros((n_vox, 1, 1));
        let mut md = Array3::<f32>::zeros((n_vox, 1, 1));
        let mut s0v = Array3::<f32>::from_elem((n_vox, 1, 1), 1000.0);
        let mut tensor = Array4::<f32>::zeros((n_vox, 1, 1, 6));
        let mut pdir = Array4::<f32>::zeros((n_vox, 1, 1, 3));

        let wm_fibers = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.577, 0.577, 0.577],
            [0.707, 0.707, 0.0],
            [0.0, 0.707, 0.707],
            [0.707, 0.0, 0.707],
            [0.267, 0.535, 0.802],
        ];
        for vox in 0..n_vox {
            let s0 = 1000.0_f64;
            // classify: 0..8 WM, 8..14 CSF, 14..40 GM.
            let (class, dir): (u8, [f64; 3]) = if vox < 8 {
                (0u8, wm_fibers[vox])
            } else if vox < 14 {
                (1u8, [0.0, 0.0, 1.0])
            } else {
                (2u8, [0.0, 0.0, 1.0])
            };
            let d = {
                let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt().max(1e-9);
                [dir[0] / n, dir[1] / n, dir[2] / n]
            };
            // b0 volumes = s0.
            for i in 0..6 {
                data[(vox, 0, 0, i)] = s0 as f32;
            }
            for i in 6..ng {
                let b = bvals[i];
                let g = bvecs[i];
                let val = match class {
                    0 => {
                        // WM: S0 · Σ r_ℓ(b) Y_ℓ0(g·fiber)
                        let cos = g[0] * d[0] + g[1] * d[1] + g[2] * d[2];
                        let r = r_true(b);
                        s0 * (0..3).map(|j| r[j] * ylm0(2 * j, cos)).sum::<f64>()
                    }
                    1 => s0 * (-3.0e-3 * b).exp(), // CSF fast isotropic
                    _ => s0 * (-1.2e-3 * b).exp(), // GM
                };
                data[(vox, 0, 0, i)] = val as f32;
            }
            // DTI fields to make selection unambiguous.
            match class {
                0 => {
                    fa[(vox, 0, 0)] = 0.8;
                    md[(vox, 0, 0)] = 0.77e-3;
                    // eigenvalues 1.7/0.3/0.3 e-3 along the fiber → dominance ≈ 5.7.
                    // Express the tensor in the lab frame: λ1 d dᵀ + λ⊥(I − d dᵀ).
                    let (l1, lp) = (1.7e-3_f64, 0.3e-3_f64);
                    let dd = |a: usize, b: usize| d[a] * d[b];
                    let comp = [
                        lp + (l1 - lp) * dd(0, 0),
                        (l1 - lp) * dd(0, 1),
                        (l1 - lp) * dd(0, 2),
                        lp + (l1 - lp) * dd(1, 1),
                        (l1 - lp) * dd(1, 2),
                        lp + (l1 - lp) * dd(2, 2),
                    ];
                    for (k, c) in comp.iter().enumerate() {
                        tensor[(vox, 0, 0, k)] = *c as f32;
                    }
                }
                1 => {
                    fa[(vox, 0, 0)] = 0.05;
                    md[(vox, 0, 0)] = 3.0e-3;
                    for k in [0, 3, 5] {
                        tensor[(vox, 0, 0, k)] = 3.0e-3;
                    }
                }
                _ => {
                    fa[(vox, 0, 0)] = 0.15;
                    md[(vox, 0, 0)] = 1.2e-3;
                    for k in [0, 3, 5] {
                        tensor[(vox, 0, 0, k)] = 1.2e-3;
                    }
                }
            }
            pdir[(vox, 0, 0, 0)] = d[0] as f32;
            pdir[(vox, 0, 0, 1)] = d[1] as f32;
            pdir[(vox, 0, 0, 2)] = d[2] as f32;
        }

        let dwi = DwiData::from_table(
            data,
            mask,
            gtab,
            BvecFrame::WorldRas,
            std::path::PathBuf::from("synthetic.nii.gz"),
        );
        let dti = DtiVolumeResult {
            s0: s0v.clone(),
            fa,
            md,
            outlier_fraction: Array3::zeros((n_vox, 1, 1)),
            tensor,
            principal_dir: pdir,
            iterations: None,
            converged: None,
        };
        let _ = &mut s0v;

        let cfg = ConshResponseConfig {
            dhollander: DhollanderConfig {
                md_csf_pct: 15.0,
                // These fixtures plant one tissue per voxel in a 40-voxel strip,
                // which the staged selection's mask erosion would wipe out. They
                // exercise the response SCALING, so pin them to the selection the
                // planting was designed for.
                legacy_selection: true,
                ..Default::default()
            },
            step: 50.0,
            sigma_b: 80.0,
            min_weight_frac: 0.02,
            grid_bmax: None,
            ..Default::default()
        };
        let (resp, diag) = estimate_consh_responses(&dwi, &dti, &cfg).unwrap();

        assert_eq!(diag.n_wm_voxels, 8, "should select the 8 planted WM voxels");
        assert_eq!(diag.n_csf_voxels, 6);
        // WM response at grid b=1000 (index 20) recovers the planted r_ℓ(1000).
        let expect = r_true(1000.0);
        let row = &resp.wm.coeffs[20];
        for j in 0..3 {
            assert_abs_diff_eq!(row[j], expect[j], epsilon = 3e-2);
        }
        // b=0 row is isotropic; every voxel here shares b0=1000, so mean =
        // median = ref and the anchor is exactly √4π. (With heterogeneous b0
        // it is only approximate — see the common-reference tests below.)
        assert_abs_diff_eq!(resp.wm.coeffs[0][0], (4.0 * PI).sqrt(), epsilon = 1e-9);
        assert_abs_diff_eq!(resp.wm.coeffs[0][1], 0.0, epsilon = 1e-12);
        // CSF at b=1000 ≈ exp(-3e-3·1000)=0.0498 → iso coeff = that · sqrt(4π).
        assert_abs_diff_eq!(
            resp.csf.coeffs[20][0],
            (-3.0e-3 * 1000.0_f64).exp() * (4.0 * PI).sqrt(),
            epsilon = 2e-2
        );
    }
    fn synthetic_scan(
        s0_wm: &[f64; 8],
        s0_csf: f64,
        s0_gm: f64,
        gain: f64,
    ) -> (crate::io::dwi::DwiData, crate::dti::DtiVolumeResult) {
        use crate::io::dwi::DwiData;
        use crate::dti::DtiVolumeResult;
        use crate::qspace::{BvecFrame, GradientTable};
        use ndarray::{Array3, Array4};

        let sphere = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
        let dirs: Vec<[f64; 3]> = sphere
            .iter()
            .step_by((sphere.len() / 30).max(1))
            .take(30)
            .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
            .collect();
        let mut bvals = vec![0.0; 6];
        let mut bvecs: Vec<[f64; 3]> = vec![[0.0; 3]; 6];
        for &b in &[1000.0, 2000.0, 3000.0] {
            for d in &dirs {
                bvals.push(b);
                bvecs.push(*d);
            }
        }
        let gtab =
            GradientTable::new(bvals.clone(), bvecs.clone(), Some(0.043), Some(0.011), None)
                .unwrap();
        let ng = gtab.n_grads();

        let r_true = |b: f64| -> [f64; 3] {
            let e = (-0.7e-3 * b).exp();
            [e, -0.35 * e, 0.12 * e]
        };
        let ylm0 =
            |l: usize, c: f64| ((2.0 * l as f64 + 1.0) / (4.0 * PI)).sqrt() * legendre(l, c);

        let n_vox = 40;
        let mut data = Array4::<f32>::zeros((n_vox, 1, 1, ng));
        let mask = Array3::<bool>::from_elem((n_vox, 1, 1), true);
        let mut fa = Array3::<f32>::zeros((n_vox, 1, 1));
        let mut md = Array3::<f32>::zeros((n_vox, 1, 1));
        let mut tensor = Array4::<f32>::zeros((n_vox, 1, 1, 6));
        let mut pdir = Array4::<f32>::zeros((n_vox, 1, 1, 3));

        let wm_fibers = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.577, 0.577, 0.577],
            [0.707, 0.707, 0.0],
            [0.0, 0.707, 0.707],
            [0.707, 0.0, 0.707],
            [0.267, 0.535, 0.802],
        ];
        for vox in 0..n_vox {
            let (class, dir, s0): (u8, [f64; 3], f64) = if vox < 8 {
                (0u8, wm_fibers[vox], s0_wm[vox])
            } else if vox < 14 {
                (1u8, [0.0, 0.0, 1.0], s0_csf)
            } else {
                (2u8, [0.0, 0.0, 1.0], s0_gm)
            };
            let d = unit(dir);
            for i in 0..6 {
                data[(vox, 0, 0, i)] = (gain * s0) as f32;
            }
            for i in 6..ng {
                let b = bvals[i];
                let g = bvecs[i];
                let val = match class {
                    0 => {
                        let cos = g[0] * d[0] + g[1] * d[1] + g[2] * d[2];
                        let r = r_true(b);
                        s0 * (0..3).map(|j| r[j] * ylm0(2 * j, cos)).sum::<f64>()
                    }
                    1 => s0 * (-3.0e-3 * b).exp(),
                    _ => s0 * (-1.2e-3 * b).exp(),
                };
                data[(vox, 0, 0, i)] = (gain * val) as f32;
            }
            match class {
                0 => {
                    fa[(vox, 0, 0)] = 0.8;
                    md[(vox, 0, 0)] = 0.77e-3;
                    let (l1, lp) = (1.7e-3_f64, 0.3e-3_f64);
                    let dd = |a: usize, b: usize| d[a] * d[b];
                    let comp = [
                        lp + (l1 - lp) * dd(0, 0),
                        (l1 - lp) * dd(0, 1),
                        (l1 - lp) * dd(0, 2),
                        lp + (l1 - lp) * dd(1, 1),
                        (l1 - lp) * dd(1, 2),
                        lp + (l1 - lp) * dd(2, 2),
                    ];
                    for (k, c) in comp.iter().enumerate() {
                        tensor[(vox, 0, 0, k)] = *c as f32;
                    }
                }
                1 => {
                    fa[(vox, 0, 0)] = 0.05;
                    md[(vox, 0, 0)] = 3.0e-3;
                    for k in [0, 3, 5] {
                        tensor[(vox, 0, 0, k)] = 3.0e-3;
                    }
                }
                _ => {
                    fa[(vox, 0, 0)] = 0.15;
                    md[(vox, 0, 0)] = 1.2e-3;
                    for k in [0, 3, 5] {
                        tensor[(vox, 0, 0, k)] = 1.2e-3;
                    }
                }
            }
            pdir[(vox, 0, 0, 0)] = d[0] as f32;
            pdir[(vox, 0, 0, 1)] = d[1] as f32;
            pdir[(vox, 0, 0, 2)] = d[2] as f32;
        }

        let dwi = DwiData::from_table(
            data,
            mask,
            gtab,
            BvecFrame::WorldRas,
            std::path::PathBuf::from("synthetic.nii.gz"),
        );
        let dti = DtiVolumeResult {
            s0: Array3::from_elem((n_vox, 1, 1), 1000.0),
            fa,
            md,
            outlier_fraction: Array3::zeros((n_vox, 1, 1)),
            tensor,
            principal_dir: pdir,
            iterations: None,
            converged: None,
        };
        (dwi, dti)
    }
    fn test_cfg() -> ConshResponseConfig {
        ConshResponseConfig {
            dhollander: DhollanderConfig {
                md_csf_pct: 15.0,
                // These fixtures plant one tissue per voxel in a 40-voxel strip,
                // which the staged selection's mask erosion would wipe out. They
                // exercise the response SCALING, so pin them to the selection the
                // planting was designed for.
                legacy_selection: true,
                ..Default::default()
            },
            step: 50.0,
            sigma_b: 80.0,
            min_weight_frac: 0.02,
            grid_bmax: None,
            ..Default::default()
        }
    }
    fn common_reference_scaling_is_gain_free() {
        // A global receiver-gain change must cancel in S/ref exactly: the two
        // scans differ only by ×3.7 on every sample, so every response
        // coefficient must agree (up to f32 storage rounding).
        let (dwi_a, dti_a) = synthetic_scan(&[1000.0; 8], 2600.0, 800.0, 1.0);
        let (dwi_b, dti_b) = synthetic_scan(&[1000.0; 8], 2600.0, 800.0, 3.7);
        let (ra, _) = estimate_consh_responses(&dwi_a, &dti_a, &test_cfg()).unwrap();
        let (rb, _) = estimate_consh_responses(&dwi_b, &dti_b, &test_cfg()).unwrap();
        for (a, b) in [(&ra.wm, &rb.wm), (&ra.gm, &rb.gm), (&ra.csf, &rb.csf)] {
            for (row_a, row_b) in a.coeffs.iter().zip(&b.coeffs) {
                for (ca, cb) in row_a.iter().zip(row_b) {
                    assert_abs_diff_eq!(ca, cb, epsilon = 1e-5);
                }
            }
        }
    }
    fn dki_estimator_recovers_planted_isotropic_decays_and_scaling() {
        // GM/CSF are mono-exponential, so DKI ISO + offset represents them
        // exactly; the common-reference b=0 amplitudes (CSF 2.6×, GM 0.8×)
        // must survive as S0 + C. WM's planted response is not DKI-shaped, so
        // it is only checked loosely, plus the fitted-parameter diagnostics.
        let factor = (4.0 * PI).sqrt();
        let (dwi, dti) = synthetic_scan(&[1000.0; 8], 2600.0, 800.0, 1.0);
        let cfg = ConshResponseConfig { estimator: ResponseEstimator::Dki, ..test_cfg() };
        let (resp, diag) = estimate_consh_responses(&dwi, &dti, &cfg).unwrap();

        let fits = diag.dki.expect("Dki estimator reports its fits");
        assert_eq!(fits.wm.params.n_params(), 7);
        assert_eq!(fits.gm.params.n_params(), 4);
        assert!(fits.wm.converged && fits.gm.converged && fits.csf.converged);
        assert_abs_diff_eq!(fits.csf.params.d_par, 3.0, epsilon = 1e-3);
        assert_abs_diff_eq!(fits.gm.params.d_par, 1.2, epsilon = 1e-3);

        assert_abs_diff_eq!(resp.csf.coeffs[0][0], 2.6 * factor, epsilon = 1e-3);
        assert_abs_diff_eq!(resp.gm.coeffs[0][0], 0.8 * factor, epsilon = 1e-3);
        for (k, &b) in resp_grid_bs(&resp).iter().enumerate() {
            assert_abs_diff_eq!(resp.csf.coeffs[k][0], 2.6 * (-3.0e-3 * b).exp() * factor, epsilon = 1e-3);
            assert_abs_diff_eq!(resp.gm.coeffs[k][0], 0.8 * (-1.2e-3 * b).exp() * factor, epsilon = 1e-3);
        }
        // WM: b=0 row isotropic at the reference, b=1000 row near the planted one.
        assert_abs_diff_eq!(resp.wm.coeffs[0][0], factor, epsilon = 1e-2);
        let expect = [(-0.7_f64).exp(), -0.35 * (-0.7_f64).exp(), 0.12 * (-0.7_f64).exp()];
        for j in 0..3 {
            assert_abs_diff_eq!(resp.wm.coeffs[20][j], expect[j], epsilon = 5e-2);
        }
        // Monotone spherical mean on every row, including past the data.
        for k in 1..resp.wm.coeffs.len() {
            assert!(resp.wm.coeffs[k][0] <= resp.wm.coeffs[k - 1][0] + 1e-9);
        }
    }

    /// Grid b-values of a dense-b response estimated with `test_cfg()` (step 50).
    fn resp_grid_bs(resp: &ConshResponses) -> Vec<f64> {
        (0..resp.csf.coeffs.len()).map(|k| 50.0 * k as f64).collect()
    }
}
