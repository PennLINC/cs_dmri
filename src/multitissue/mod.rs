// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multi-tissue spherical-deconvolution reconstructions.
//!
//! Houses the SS3T (Single-Shell 3-Tissue) algorithm — WM FOD + GM + CSF fit
//! to a single-shell DWI by an alternating two-tissue fixed point — together
//! with response estimation and multi-tissue intensity normalisation.

pub mod dhollander;
pub mod forward;
pub mod mtnormalise;
pub mod pipeline;
pub mod response;
pub mod response_estimation;
pub mod sidecar;
pub mod ss3t;
pub mod volume;

pub use dhollander::{
    DhollanderSelectConfig, DhollanderSelection, DhollanderStageCounts, select_voxels,
};
pub use forward::{ForwardOperator, PerShellPredictor, ShellPlan};
pub use mtnormalise::{MtnormaliseConfig, MtnormaliseDiagnostics, mtnormalise, target_sum_sqrt_4pi};
pub use response::{ResponseError, TissueResponse};
pub use response_estimation::{
    DhollanderConfig, ResponseEstimationDiagnostics, Ss3tResponseEstimate, estimate_responses,
    format_response_txt, write_response_txt,
};
pub use sidecar::{Ss3tSidecar, Ss3tSolverParams};
pub use ss3t::{
    LmaxWmStrategy, Ss3tConfig, Ss3tPlan, Ss3tResponses, Ss3tVolumePlan, Ss3tVoxelDiagnostics,
    Ss3tVoxelResult, Ss3tVoxelWorkspace, fit_voxel, fit_voxel_into, fit_voxel_path_bic_into,
};
pub use volume::{Ss3tVolumeResult, fit_volume_ss3t, fit_volume_ss3t_reporting};
