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
use crate::multitissue::response::TissueResponse;
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
/// + principal direction).
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

/// Median of the per-voxel mean b=0 signal over `voxels`. Over the WM
/// single-fiber voxels this is the `DWI_ref` recorded in
/// [`ResponseEstimationDiagnostics`]; the median rather than the mean so one bright outlier (a vessel, a mis-selected
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
}
