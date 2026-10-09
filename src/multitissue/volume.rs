// SPDX-License-Identifier: MIT OR Apache-2.0
//! SS3T volume driver: build the per-volume plan, then fit every masked voxel.
//!
//! Mirrors the rayon pattern in [`fit::fit_volume_reporting`] but produces
//! three coupled output buffers (WM FOD, GM, CSF) instead of one. Supports
//! both `LmaxWmStrategy::Fixed` (one inner solve sweep per voxel) and
//! `LmaxWmStrategy::PathBic` (sweep candidate lmaxes per voxel, pick min-BIC).

use ndarray::{Array3, Array4};

use crate::io::dwi::DwiData;
use crate::multitissue::ss3t::{
    Ss3tConfig, Ss3tResponses, Ss3tVolumePlan, fit_voxel_path_bic_into,
};
use crate::voxel_loop;
use crate::Result;

/// Per-volume SS3T outputs.
pub struct Ss3tVolumeResult {
    /// WM FOD coefficient field, shape (X, Y, Z, max_n_sh_wm). For path-BIC,
    /// voxels that select a smaller lmax have their higher-order coefficients
    /// zero-padded to the volume's max width.
    pub wm: Array4<f32>,
    /// GM compartment, shape (X, Y, Z, 1).
    pub gm: Array4<f32>,
    /// CSF compartment, shape (X, Y, Z, 1).
    pub csf: Array4<f32>,
    /// Per-voxel total inner-iteration count (sum across all CSD calls).
    /// Optional — present when `compute_diagnostics` is requested.
    pub iterations: Option<Array3<u32>>,
    /// Per-voxel L2 norm of the final residual (in augmented signal units).
    pub residual_l2: Option<Array3<f32>>,
    /// Per-voxel "all CSD calls converged" flag.
    pub converged: Option<Array3<u8>>,
    /// Per-voxel selected lmax (only when path-BIC is in use; `None` for
    /// Fixed strategy where every voxel uses the same lmax).
    pub chosen_lmax: Option<Array3<u8>>,
    /// Per-voxel minimum BIC. `None` unless diagnostics are on AND path-BIC
    /// is in use (BIC is meaningless when there's nothing to choose between).
    pub bic: Option<Array3<f32>>,
    /// The volume plan, kept for inspection / sidecar metadata.
    pub plan: Ss3tVolumePlan,
}

/// Per-voxel diagnostic toggle. Mirrors [`crate::FitConfig`] but kept separate
/// because SS3T's diagnostic shape is different (no R²/sparsity/α).
#[derive(Debug, Clone, Copy, Default)]
pub struct Ss3tFitConfig {
    pub compute_diagnostics: bool,
}

/// Fit SS3T to every masked voxel of `dwi`.
pub fn fit_volume_ss3t(
    dwi: &DwiData,
    responses: &Ss3tResponses,
    cfg: &Ss3tConfig,
    diag_cfg: Ss3tFitConfig,
) -> Result<Ss3tVolumeResult> {
    fit_volume_ss3t_reporting(dwi, responses, cfg, diag_cfg, || ())
}

/// Variant that calls `on_voxel` once per completed voxel — for progress
/// heartbeats. The callback runs on rayon worker threads, so it must be `Sync`.
pub fn fit_volume_ss3t_reporting<F>(
    dwi: &DwiData,
    responses: &Ss3tResponses,
    cfg: &Ss3tConfig,
    diag_cfg: Ss3tFitConfig,
    on_voxel: F,
) -> Result<Ss3tVolumeResult>
where
    F: Fn() + Sync,
{
    let volume_plan = Ss3tVolumePlan::build(&dwi.gtab, responses, cfg)?;

    let s = dwi.data.shape();
    let (nx, ny, nz, nt) = (s[0], s[1], s[2], s[3]);
    if nt != volume_plan.n_aug {
        return Err(crate::CsDmriError::Dimension(format!(
            "ss3t: DWI has {} volumes but plan expects {}",
            nt, volume_plan.n_aug
        )));
    }
    let path_bic_in_use = volume_plan.candidates.len() > 1;

    struct VoxelOut {
        wm: nalgebra::DVector<f64>,
        gm: f64,
        csf: f64,
        iters: u32,
        residual: f32,
        converged: u8,
        chosen_lmax: u8,
        bic: f32,
    }

    // Per-rayon-worker workspaces: one workspace per candidate plan, allocated
    // once per worker. For PathBic with 5 candidates that's 5 workspaces per
    // worker thread (~16 workers typical) = 80 workspace pairs total, vs
    // millions of allocations for per-voxel scratch.
    let results = voxel_loop::run_init(
        &dwi.mask,
        on_voxel,
        || {
            (
                volume_plan.workspaces(),
                Vec::<f64>::with_capacity(volume_plan.n_aug),
            )
        },
        |(workspaces, signal_buf), x, y, z| {
            let view = dwi.data.slice(ndarray::s![x, y, z, ..]);
            signal_buf.clear();
            signal_buf.extend(view.iter().map(|&v| v as f64));
            let result = fit_voxel_path_bic_into(signal_buf, &volume_plan, workspaces);
            VoxelOut {
                wm: result.c_wm,
                gm: result.c_gm,
                csf: result.c_csf,
                iters: result.diagnostics.total_inner_iter as u32,
                residual: result.diagnostics.residual_l2 as f32,
                converged: if result.diagnostics.all_converged { 1 } else { 0 },
                chosen_lmax: result.chosen_lmax as u8,
                bic: result.min_bic as f32,
            }
        },
    );

    let mut wm = Array4::<f32>::zeros((nx, ny, nz, volume_plan.max_n_sh_wm));
    let mut gm = Array4::<f32>::zeros((nx, ny, nz, 1));
    let mut csf = Array4::<f32>::zeros((nx, ny, nz, 1));
    let mut iters_map = diag_cfg
        .compute_diagnostics
        .then(|| Array3::<u32>::zeros((nx, ny, nz)));
    let mut residual_map = diag_cfg
        .compute_diagnostics
        .then(|| Array3::<f32>::zeros((nx, ny, nz)));
    let mut converged_map = diag_cfg
        .compute_diagnostics
        .then(|| Array3::<u8>::zeros((nx, ny, nz)));
    let mut chosen_lmax_map = (diag_cfg.compute_diagnostics && path_bic_in_use)
        .then(|| Array3::<u8>::zeros((nx, ny, nz)));
    let mut bic_map = (diag_cfg.compute_diagnostics && path_bic_in_use)
        .then(|| Array3::<f32>::zeros((nx, ny, nz)));

    for ((x, y, z), v) in results {
        for k in 0..volume_plan.max_n_sh_wm {
            wm[(x, y, z, k)] = v.wm[k] as f32;
        }
        gm[(x, y, z, 0)] = v.gm as f32;
        csf[(x, y, z, 0)] = v.csf as f32;
        if let Some(arr) = iters_map.as_mut() {
            arr[(x, y, z)] = v.iters;
        }
        if let Some(arr) = residual_map.as_mut() {
            arr[(x, y, z)] = v.residual;
        }
        if let Some(arr) = converged_map.as_mut() {
            arr[(x, y, z)] = v.converged;
        }
        if let Some(arr) = chosen_lmax_map.as_mut() {
            arr[(x, y, z)] = v.chosen_lmax;
        }
        if let Some(arr) = bic_map.as_mut() {
            arr[(x, y, z)] = v.bic;
        }
    }

    Ok(Ss3tVolumeResult {
        wm,
        gm,
        csf,
        iterations: iters_map,
        residual_l2: residual_map,
        converged: converged_map,
        chosen_lmax: chosen_lmax_map,
        bic: bic_map,
        plan: volume_plan,
    })
}
