// SPDX-License-Identifier: MIT OR Apache-2.0
//! JSON sidecar describing an SS3T fit. One sidecar lives next to the WM FOD
//! NIfTI and records the inputs (response file paths), algorithm parameters,
//! the inner ICLS config, and provenance.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::io::provenance::Provenance;
use crate::qspace::BvecFrame;

/// Inner ICLS solver knobs flattened into the sidecar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ss3tSolverParams {
    pub max_iter: usize,
    pub tol: f64,
    pub epsilon: f64,
}

/// Sidecar mirroring `cs-fit`'s `SidecarMetadata` shape but specific to SS3T.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ss3tSidecar {
    /// One of `"ss3t"`. Locks the sidecar shape so older `cs-fit` parsers
    /// reject it loudly.
    pub method: String,
    /// SS3T outer-iteration count.
    pub niter: u32,
    /// b=0 contribution percentage used to derive `bzero_sw`.
    pub bzero_pct: f64,
    /// Maximum WM lmax used (after clamping to the WM response file). Equal
    /// to the single value for `Fixed`, the largest candidate for `PathBic`.
    pub lmax_wm: usize,
    /// Candidate WM lmaxes considered. Length 1 for `Fixed`, ≥ 2 for
    /// `PathBic`. Records the strategy implicitly (length tells you which).
    pub lmax_wm_candidates: Vec<usize>,
    /// `sqrt(n_dwi · bzero_pct / (n_b0 · 100))` — the b=0 scaling factor.
    pub bzero_sw: f64,
    /// Inner ICLS (Goldfarb-Idnani active-set) parameters.
    pub icls: Ss3tSolverParams,
    /// Number of sphere directions used for the WM non-negativity constraint.
    pub n_sphere_dirs: usize,
    /// Identifier of the sphere used for non-negativity (e.g. `"dsistudio_odf8"`).
    pub sphere_id: String,
    /// Source file paths for the three response functions. Hashing is not
    /// performed (would require an additional dependency); the `provenance`
    /// block's run-level fingerprinting is the canonical reproducibility
    /// signal.
    pub responses: ResponseProvenance,
    /// Per-output coefficient counts.
    pub n_coefficients: NCoefficients,
    /// Frame the bvecs (and therefore the fitted SH coefficients) live in.
    pub bvec_frame: BvecFrame,
    /// Reproducibility provenance. Optional — `--provenance none` skips it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseProvenance {
    pub wm_path: PathBuf,
    pub gm_path: PathBuf,
    pub csf_path: PathBuf,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NCoefficients {
    pub wm: usize,
    pub gm: usize,
    pub csf: usize,
}
