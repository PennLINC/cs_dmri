// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end continuous-b multi-tissue CSD (ConSH): dense-b responses
//! (estimated or given) → per-voxel fit → optional mtnormalise. The `consh`
//! tool and the Python `consh_pipeline` both run this.

use ndarray::Array3;

use crate::dti::{DtiFitConfig, DtiVolumeResult, RestoreConfig, fit_volume_restore_graddev};
use crate::graddev::VoxelGradDev;
use crate::io::dwi::DwiData;
use crate::io::odx_out::ReferenceB0Source;
use crate::multitissue::consh::{
    ConshConfig, ConshResponses, ConshVolumeResult, DenseBGrid, fit_volume_consh_graddev,
};
use crate::multitissue::mtnormalise::{MtnormaliseConfig, MtnormaliseDiagnostics, mtnormalise};
use crate::multitissue::pipeline::{PipelineObserver, PipelineStage};
use crate::multitissue::response_estimation::{
    ConshResponseConfig, ResponseEstimationDiagnostics, ResponseEstimator,
    estimate_consh_responses_graddev,
};
use crate::{CsDmriError, Result};

/// Where the pipeline's dense-b responses come from.
#[derive(Debug, Clone)]
pub enum ConshResponseSource {
    /// Use these responses as-is, e.g. a group average. Their rows must cover
    /// the series' dense-b grid.
    Provided(ConshResponses),
    /// RESTORE tensor fit, then Dhollander tissue selection and dense-b
    /// estimation. `responses.estimator` is used unless
    /// [`ConshPipelineConfig::estimator`] is `None`.
    Estimate {
        restore: RestoreConfig,
        responses: ConshResponseConfig,
    },
}

/// Configuration for [`consh_pipeline`].
#[derive(Debug, Clone)]
pub struct ConshPipelineConfig {
    pub responses: ConshResponseSource,
    /// Response estimator for [`ConshResponseSource::Estimate`]. `None`
    /// chooses one with [`auto_response_estimator`].
    pub estimator: Option<ResponseEstimator>,
    /// Deconvolution settings. With a gradient-deviation field, `grid_bmax`
    /// is set to the largest effective b and `b_interp` is turned on.
    pub consh: ConshConfig,
    /// `None` skips intensity normalization.
    pub normalize: Option<MtnormaliseConfig>,
    pub compute_diagnostics: bool,
    /// White-matter mask for the reference b=0 recorded with the result
    /// (see [`reference_b0`]).
    pub reference_mask: Option<Array3<bool>>,
}

/// Output of [`consh_pipeline`].
pub struct ConshPipelineResult {
    /// Tissue maps (normalized when `normalize` was set) and diagnostics.
    pub fit: ConshVolumeResult,
    /// The responses the fit used.
    pub responses: ConshResponses,
    /// The estimator that produced `responses`, when they were estimated.
    pub estimator: Option<ResponseEstimator>,
    /// Estimation details and the tensor fit, when responses were estimated.
    pub estimate: Option<(ResponseEstimationDiagnostics, DtiVolumeResult)>,
    pub normalization: Option<MtnormaliseDiagnostics>,
    /// Upper end of the dense-b grid when a gradient-deviation field extended it.
    pub grid_bmax: Option<f64>,
    /// Whether the forward model interpolated the response in b.
    pub b_interp: bool,
    /// 95th percentile over the mask of the shift the gradient-deviation field
    /// applies to b-values, when there is a field.
    pub b_shift_p95: Option<f64>,
    /// Median b=0 in white matter, the `DWI_ref` recorded in ODX sidecars.
    pub reference_b0: Option<(f64, ReferenceB0Source)>,
    /// Conditions worth reporting that did not stop the run.
    pub warnings: Vec<String>,
}

/// The response estimator `consh --response-estimator auto` uses. With a
/// gradient-deviation field: `Smooth` when the field moves b-values by more
/// than `sigma_b` (95th percentile `b_shift_p95` over the mask), else `Kernel`.
/// Without a field: `Smooth` when the forward model interpolates in b, else
/// `Kernel`. The parametric `Dki` estimator is never chosen automatically.
pub fn auto_response_estimator(b_shift_p95: Option<f64>, sigma_b: f64, b_interp: bool) -> ResponseEstimator {
    match b_shift_p95 {
        Some(shift) if shift > sigma_b => ResponseEstimator::Smooth,
        Some(_) => ResponseEstimator::Kernel,
        None if b_interp => ResponseEstimator::Smooth,
        None => ResponseEstimator::Kernel,
    }
}

/// Output of [`estimate_dense_b_responses`].
pub struct DenseBEstimate {
    pub responses: ConshResponses,
    /// The estimator used (`estimator` argument, or the automatic choice).
    pub estimator: ResponseEstimator,
    pub diagnostics: ResponseEstimationDiagnostics,
    /// The tensor fit tissue selection used.
    pub dti: DtiVolumeResult,
    /// Upper end of the dense-b grid when a gradient-deviation field extended it.
    pub grid_bmax: Option<f64>,
}

/// Dense-b responses from `dwi`, as [`consh_pipeline`] estimates them: RESTORE
/// tensor fit (with per-voxel gradient tables from `graddev`), Dhollander tissue
/// selection, dense-b rows. With a field the grid reaches the largest effective
/// b. `estimator` `None` chooses with [`auto_response_estimator`], given
/// whether the fit will interpolate in b (`b_interp`, implied by a field).
pub fn estimate_dense_b_responses(
    dwi: &DwiData,
    graddev: Option<&VoxelGradDev>,
    restore: &RestoreConfig,
    responses: &ConshResponseConfig,
    estimator: Option<ResponseEstimator>,
    b_interp: bool,
    observer: &impl PipelineObserver,
) -> Result<DenseBEstimate> {
    check_consh_sampling(dwi)?;
    let grid_bmax = graddev.map(|gd| gd.max_effective_b(dwi, &dwi.mask));
    let b_shift_p95 = graddev.map(|gd| gd.b_shift_quantile(dwi, &dwi.mask, 0.95));
    let mut rcfg = *responses;
    rcfg.estimator = estimator.unwrap_or_else(|| {
        auto_response_estimator(b_shift_p95, rcfg.sigma_b, b_interp || graddev.is_some())
    });
    rcfg.grid_bmax = grid_bmax;
    observer.stage(PipelineStage::Dti);
    let dti = fit_volume_restore_graddev(
        dwi,
        restore,
        DtiFitConfig { compute_diagnostics: false },
        graddev,
        || observer.voxel(),
    )?;
    observer.stage(PipelineStage::ResponseEstimation);
    let (responses, diagnostics) = estimate_consh_responses_graddev(dwi, &dti, &rcfg, graddev)?;
    Ok(DenseBEstimate { responses, estimator: rcfg.estimator, diagnostics, dti, grid_bmax })
}

/// Distinct b-values, b=0 included, ConSH needs: as for MRtrix3's
/// `msmt_csd`, at least as many as the tissues it separates (WM, GM, CSF).
pub const MIN_CONSH_B_VALUES: usize = 3;

/// Refuse acquisitions with too few distinct b-values to separate three
/// tissues. Single-shell data (b=0 and one b-value) belong to SS3T-CSD.
pub fn check_consh_sampling(dwi: &DwiData) -> Result<()> {
    let shells = crate::fit::dwi_shells(&dwi.gtab);
    let has_b0 = dwi.gtab.bvals.iter().any(|&b| b <= dwi.gtab.b0_threshold);
    let n = shells.len() + usize::from(has_b0);
    if n < MIN_CONSH_B_VALUES {
        let listed: Vec<String> = shells.iter().map(|b| format!("{b:.0}")).collect();
        return Err(CsDmriError::Other(format!(
            "ConSH separates three tissues, which needs diffusion data at {MIN_CONSH_B_VALUES} or \
             more distinct b-values including b=0 (as MRtrix's msmt_csd needs as many shells as \
             tissues); this series has {n} ({}b = {}). For single-shell data use single-shell \
             three-tissue CSD (cs-ss3t-full, cs_dmri.ss3t_pipeline).",
            if has_b0 { "b=0 and " } else { "" },
            listed.join(", ")
        )));
    }
    Ok(())
}

/// Run responses → ConSH → mtnormalise on `dwi`, with per-voxel gradient
/// tables from `graddev` when given.
pub fn consh_pipeline(
    dwi: &DwiData,
    graddev: Option<&VoxelGradDev>,
    cfg: &ConshPipelineConfig,
    observer: &impl PipelineObserver,
) -> Result<ConshPipelineResult> {
    check_consh_sampling(dwi)?;
    let mut consh = cfg.consh.clone();
    let grid_bmax = graddev.map(|gd| gd.max_effective_b(dwi, &dwi.mask));
    let b_shift_p95 = graddev.map(|gd| gd.b_shift_quantile(dwi, &dwi.mask, 0.95));
    if graddev.is_some() {
        consh.grid_bmax = grid_bmax;
        consh.b_interp = true;
    }
    let mut warnings = Vec::new();

    let (responses, estimator, estimate) = match &cfg.responses {
        ConshResponseSource::Provided(r) => {
            if graddev.is_some() {
                let need = DenseBGrid::for_gtab_min_bmax(&dwi.gtab, consh.step, grid_bmax).len();
                let have = [r.wm.n_shells(), r.gm.n_shells(), r.csf.n_shells()];
                if have.iter().any(|&n| n < need) {
                    return Err(CsDmriError::Other(format!(
                        "the gradient-deviation field raises the effective maximum b to {:.0}, so \
                         the dense-b grid needs {need} rows, but the given responses have \
                         {}/{}/{} (WM/GM/CSF). Responses must cover the effective b range: \
                         estimate them with the same field.",
                        grid_bmax.unwrap_or(0.0),
                        have[0],
                        have[1],
                        have[2]
                    )));
                }
            }
            (r.clone(), None, None)
        }
        ConshResponseSource::Estimate { restore, responses } => {
            let mut rcfg = *responses;
            rcfg.step = consh.step;
            let est = estimate_dense_b_responses(
                dwi,
                graddev,
                restore,
                &rcfg,
                cfg.estimator,
                consh.b_interp,
                observer,
            )?;
            (est.responses, Some(est.estimator), Some((est.diagnostics, est.dti)))
        }
    };

    if consh.quantitative && !consh.fit_noise_floor {
        warnings.push(
            "absolute-scale output without the noise-floor compartment: scaling by S0 limits \
             the magnitude of spurious fODFs in low-S0 voxels but not their orientation \
             content, which the noise-floor compartment corrects."
                .to_string(),
        );
    }

    observer.stage(PipelineStage::Consh);
    let mut fit = fit_volume_consh_graddev(
        dwi,
        &responses,
        &consh,
        graddev,
        cfg.compute_diagnostics,
        || observer.voxel(),
    )?;

    let normalization = match &cfg.normalize {
        Some(mtcfg) => {
            observer.stage(PipelineStage::Normalize);
            let diag = mtnormalise(&mut fit.wm, &mut fit.gm, &mut fit.csf, &dwi.mask, mtcfg)?;
            // The noise floor is not a tissue and stays out of the tissue sum,
            // but it is divided by the same bias field so that it stays on the
            // scale of the normalized GM and CSF amplitudes.
            if let Some(floor) = fit.floor.as_mut() {
                for ((x, y, z), &inside) in dwi.mask.indexed_iter() {
                    let f = diag.bias_field[(x, y, z)];
                    if inside && f > 0.0 {
                        floor[(x, y, z, 0)] /= f;
                    }
                }
            }
            Some(diag)
        }
        None => None,
    };

    let wm_single_fiber_median = estimate
        .as_ref()
        .map(|(d, _)| d.wm_b0_median)
        .filter(|&v| v > 0.0);
    let reference_b0 = reference_b0(dwi, cfg.reference_mask.as_ref(), wm_single_fiber_median, &fit);
    if reference_b0.is_none() {
        warnings.push("no white-matter voxels found; the reference b=0 is not recorded".to_string());
    }

    Ok(ConshPipelineResult {
        fit,
        responses,
        estimator,
        estimate,
        normalization,
        grid_bmax,
        b_interp: consh.b_interp,
        b_shift_p95,
        reference_b0,
        warnings,
    })
}

/// Fit voxels whose WM share of the l=0 sum exceeds this define white matter
/// for [`reference_b0`] when neither a mask nor estimated responses are
/// available.
pub const REFERENCE_WM_FRACTION: f64 = 0.7;

/// Median b=0 signal in white matter: the `DWI_ref` of quantitative
/// fiber-density workflows, recorded in ODX sidecars. Like MRtrix3's
/// `dwinormalise`, it is the median (not the mean) of the b=0 signal over a
/// white-matter mask, which resists bright outliers at vessels and edges.
///
/// The mask is, in order of preference: `reference_mask`; the single-fiber WM
/// voxels the responses were estimated from (their median b=0,
/// `wm_single_fiber_median`); fit voxels whose WM fraction exceeds
/// [`REFERENCE_WM_FRACTION`]. `None` when no white matter is found.
pub fn reference_b0(
    dwi: &DwiData,
    reference_mask: Option<&Array3<bool>>,
    wm_single_fiber_median: Option<f64>,
    fit: &ConshVolumeResult,
) -> Option<(f64, ReferenceB0Source)> {
    let b0: Vec<usize> = (0..dwi.gtab.n_grads())
        .filter(|&i| dwi.gtab.bvals[i] <= dwi.gtab.b0_threshold)
        .collect();
    if b0.is_empty() {
        return None;
    }
    let mean_b0 = |(x, y, z): (usize, usize, usize)| -> f64 {
        b0.iter().map(|&i| dwi.data[(x, y, z, i)] as f64).sum::<f64>() / b0.len() as f64
    };
    let median = |mut v: Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        let mid = v.len() / 2;
        v.select_nth_unstable_by(mid, f64::total_cmp);
        Some(v[mid])
    };
    if let Some(m) = reference_mask {
        let vals: Vec<f64> = m
            .indexed_iter()
            .filter(|&(_, &inside)| inside)
            .map(|(idx, _)| mean_b0(idx))
            .filter(|&v| v > 0.0)
            .collect();
        if let Some(v) = median(vals) {
            return Some((v, ReferenceB0Source::UserMask));
        }
    }
    if let Some(v) = wm_single_fiber_median {
        return Some((v, ReferenceB0Source::WhiteMatterSingleFibre));
    }
    let vals: Vec<f64> = dwi
        .mask
        .indexed_iter()
        .filter(|&(idx, &inside)| {
            let (wm, gm, csf) = (fit.wm[(idx.0, idx.1, idx.2, 0)] as f64, fit.gm[(idx.0, idx.1, idx.2, 0)] as f64, fit.csf[(idx.0, idx.1, idx.2, 0)] as f64);
            let sum = wm + gm + csf;
            inside && sum > 1e-9 && wm / sum >= REFERENCE_WM_FRACTION
        })
        .map(|(idx, _)| mean_b0(idx))
        .filter(|&v| v > 0.0)
        .collect();
    median(vals).map(|v| (v, ReferenceB0Source::WhiteMatterFraction))
}
