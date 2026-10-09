// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end single-shell three-tissue pipeline: responses (estimated or
//! given) → SS3T → optional mtnormalise. `cs-ss3t-full` and the Python
//! `ss3t_pipeline` both run this.

use crate::dti::{DtiFitConfig, DtiVolumeResult, RestoreConfig, fit_volume_restore_reporting};
use crate::io::dwi::DwiData;
use crate::multitissue::mtnormalise::{MtnormaliseConfig, MtnormaliseDiagnostics, mtnormalise};
use crate::multitissue::response_estimation::{
    DhollanderConfig, Ss3tResponseEstimate, estimate_responses,
};
use crate::multitissue::ss3t::{Ss3tConfig, Ss3tResponses};
use crate::multitissue::volume::{Ss3tFitConfig, Ss3tVolumeResult, fit_volume_ss3t_reporting};
use crate::Result;

/// Where the pipeline's tissue responses come from.
#[derive(Debug, Clone)]
pub enum ResponseSource {
    /// Use these responses as-is.
    Provided(Ss3tResponses),
    /// RESTORE DTI, then Dhollander tissue selection and averaging.
    Estimate {
        restore: RestoreConfig,
        dhollander: DhollanderConfig,
    },
}

/// Configuration for [`ss3t_pipeline`].
#[derive(Debug, Clone)]
pub struct Ss3tPipelineConfig {
    pub responses: ResponseSource,
    pub ss3t: Ss3tConfig,
    pub compute_diagnostics: bool,
    /// `None` skips intensity normalisation.
    pub normalize: Option<MtnormaliseConfig>,
}

/// Pipeline stage, reported to [`PipelineObserver::stage`] as it starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// RESTORE DTI fit feeding response estimation (per-voxel progress).
    Dti,
    /// Tissue selection and response averaging.
    ResponseEstimation,
    /// SS3T decomposition (per-voxel progress).
    Ss3t,
    /// mtnormalise.
    Normalize,
}

/// Progress hooks for [`ss3t_pipeline`]. Every method defaults to a no-op.
pub trait PipelineObserver: Sync {
    fn stage(&self, _stage: PipelineStage) {}
    /// Called once per fitted voxel during the `Dti` and `Ss3t` stages, on
    /// rayon workers.
    fn voxel(&self) {}
}

/// Observer that ignores everything.
pub struct NoProgress;
impl PipelineObserver for NoProgress {}

/// Output of [`ss3t_pipeline`].
pub struct Ss3tPipelineResult {
    /// Tissue maps (normalised when `normalize` was set) and diagnostics.
    pub fit: Ss3tVolumeResult,
    /// The responses the fit used.
    pub responses: Ss3tResponses,
    /// Estimation details and the DTI fit, when responses were estimated.
    pub estimate: Option<(Ss3tResponseEstimate, DtiVolumeResult)>,
    pub normalization: Option<MtnormaliseDiagnostics>,
    /// Non-fatal conditions worth surfacing (e.g. a clamped WM lmax).
    pub warnings: Vec<String>,
}

/// Run responses → SS3T → mtnormalise on `dwi`.
pub fn ss3t_pipeline(
    dwi: &DwiData,
    cfg: &Ss3tPipelineConfig,
    observer: &impl PipelineObserver,
) -> Result<Ss3tPipelineResult> {
    let (responses, estimate) = match &cfg.responses {
        ResponseSource::Provided(r) => (r.clone(), None),
        ResponseSource::Estimate {
            restore,
            dhollander,
        } => {
            observer.stage(PipelineStage::Dti);
            let dti = fit_volume_restore_reporting(
                dwi,
                restore,
                DtiFitConfig {
                    compute_diagnostics: false,
                },
                || observer.voxel(),
            )?;
            observer.stage(PipelineStage::ResponseEstimation);
            let est = estimate_responses(dwi, &dti, dhollander)?;
            let r = Ss3tResponses {
                wm: est.wm.clone(),
                gm: est.gm.clone(),
                csf: est.csf.clone(),
            };
            (r, Some((est, dti)))
        }
    };

    observer.stage(PipelineStage::Ss3t);
    let mut fit = fit_volume_ss3t_reporting(
        dwi,
        &responses,
        &cfg.ss3t,
        Ss3tFitConfig {
            compute_diagnostics: cfg.compute_diagnostics,
        },
        || observer.voxel(),
    )?;
    let warnings = fit.plan.lmax_clamp_warnings(responses.wm.lmax);

    let normalization = match &cfg.normalize {
        Some(mtcfg) => {
            observer.stage(PipelineStage::Normalize);
            Some(mtnormalise(
                &mut fit.wm,
                &mut fit.gm,
                &mut fit.csf,
                &dwi.mask,
                mtcfg,
            )?)
        }
        None => None,
    };

    Ok(Ss3tPipelineResult {
        fit,
        responses,
        estimate,
        normalization,
        warnings,
    })
}
