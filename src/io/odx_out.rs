// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bundled ODX output for `cs-ss3t-full` (SS3T) and `cs-fit` / `cs-odf`
//! (single-tissue SHORE).
//!
//! - `write_ss3t_odx` packs the SS3T tissue maps + responses into an ODX.
//! - `write_continuous_b_odx` is the extension point for continuous-b
//!   multi-tissue methods: the same bundle, plus dense-b responses, an
//!   optional noise-floor compartment and a BIDS-style quantitative sidecar.
//! - `build_shore_odx` projects SHORE → Tournier SH and populates an
//!   `OdxBuilder` ready for the caller to add provenance / microstructure
//!   DPVs and finalize. Shared by `cs-fit --odx-output` and `cs-odf`.
//!
//! Both share peak extraction (DSI-Studio ODF8 + Newton refinement) so any
//! ODX from this codebase looks the same to downstream peak-aware viewers.
//! Every fixel-emitting path also attaches per-fixel `dispersion` — FMLS lobe
//! integral ÷ peak amplitude, MRtrix `fod2fixel -disp` — see
//! [`FixelDispersion`].

use std::path::Path;

use anyhow::{Result, anyhow};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array3, Array4};
use rayon::prelude::*;

use odx_rs::CanonTransform;
use odx_rs::dtype::DType;
use odx_rs::fmls::{Fmls, FmlsConfig, IntegrationWeights, Lobe, hemisphere_adjacency};
use odx_rs::formats::dsistudio_odf8;
use odx_rs::mrtrix_sh::{
    ANISOTROPIC_POWER_NORM_FACTOR, RowSamplePlan, anisotropic_power, max_lmax_for_direction_count,
    ncoeffs_for_lmax, sh2amp_cart,
};
use odx_rs::peak_finder::{PeakFinderConfig, SpherePeakFinder};
use odx_rs::reference_affine::read_reference_affine;
use odx_rs::sh_basis_evaluator::ShBasisKind;
use odx_rs::stream::OdxBuilder;

use std::time::Duration;

use crate::ShoreBasis;
use crate::basis::Basis;
use crate::io::{atomic_write, atomic_write_directory};
use crate::multitissue::ss3t::Ss3tResponses;
use crate::multitissue::TissueResponse;
use crate::odf::{n_tournier_sh_coeffs, shore_to_tournier_sh_matrix};
use crate::progress::Heartbeat;

/// Peak finder defaults — matched to `cs-odf` so ODXs from either tool are
/// interchangeable for downstream peak-aware viewers.
const PEAK_NPEAKS: usize = 5;
const PEAK_REL_THRESH: f32 = 0.5;
const PEAK_MIN_SEP_DEG: f32 = 25.0;

/// Per-fixel dispersion (MRtrix `fod2fixel -disp`): FMLS lobe integral ÷ lobe
/// peak amplitude, evaluated on the same DSI-Studio ODF8 hemisphere the peak
/// finder samples — small for a tight single-fibre lobe, large for a fanning
/// one. Both terms scale linearly with the fODF, so the ratio is unchanged by
/// QA normalization or absolute-scale rescaling and means the same thing on
/// every output scale.
///
/// Fixels stay defined by `SpherePeakFinder`; FMLS runs with permissive
/// thresholds purely to *measure* each accepted peak's watershed lobe. This is
/// shared setup (adjacency + quadrature weights); per-voxel state lives in
/// [`DispersionSegmenter`], one per worker thread.
struct FixelDispersion {
    /// Antipodally-wrapped mesh adjacency. Plain hemisphere adjacency would
    /// sever a lobe straddling the rim into two, halving its integral — see
    /// `odx_rs::fmls::hemisphere_adjacency`.
    adjacency: Vec<Vec<usize>>,
    /// Spherical quadrature weights, so lobe integrals are true solid-angle
    /// integrals despite the ODF8 tessellation not being equal-area (uniform
    /// weights would bias the integral with lobe orientation). Calibrated at
    /// the largest order the 321 directions determine exactly (lmax 22, 276
    /// constraints): band-limited integrands up to that order — every fODF
    /// this crate emits — integrate to machine precision. Calibrating higher
    /// (mrtrix3 uses `LforN+2`) overdetermines the solve and lets the
    /// least-squares residual leak percent-level, axis-dependent error into
    /// the low-l constraints that actually carry the FOD.
    weights: IntegrationWeights,
}

impl FixelDispersion {
    fn for_odf8_sphere() -> Self {
        let verts = dsistudio_odf8::hemisphere_vertices_ras();
        let n = verts.len();
        let adjacency =
            hemisphere_adjacency(dsistudio_odf8::full_vertices_ras(), dsistudio_odf8::faces());
        let cal_lmax = max_lmax_for_direction_count(n);
        let design = sh2amp_cart(verts, cal_lmax);
        let weights = design
            .as_slice()
            .and_then(|flat| IntegrationWeights::new(n, ncoeffs_for_lmax(cal_lmax), flat))
            .unwrap_or_else(|| IntegrationWeights::uniform(n));
        Self { adjacency, weights }
    }

    fn segmenter(&self) -> DispersionSegmenter<'_> {
        let verts = dsistudio_odf8::hemisphere_vertices_ras();
        DispersionSegmenter {
            fmls: Fmls::new(
                verts,
                &self.adjacency,
                &self.weights,
                // Permissive on purpose: which fixels exist is the peak
                // finder's decision, so every positive lobe must survive here
                // or an accepted peak could be left without its measurement.
                FmlsConfig {
                    integral_threshold: 0.0,
                    peak_value_threshold: 0.0,
                    ..FmlsConfig::default()
                },
            ),
            vertex_lobe: vec![u32::MAX; verts.len()],
        }
    }
}

/// Per-thread FMLS state for [`FixelDispersion`].
struct DispersionSegmenter<'a> {
    fmls: Fmls<'a>,
    /// Scratch: sphere vertex → index into this voxel's lobe list.
    vertex_lobe: Vec<u32>,
}

impl DispersionSegmenter<'_> {
    /// Append one dispersion value per entry of `peaks` (a voxel's accepted
    /// `(amplitude, direction)` fixels) to `out`. `amplitudes` must be the
    /// ODF8-hemisphere samples the peaks were found on.
    fn push_voxel(&mut self, amplitudes: &[f32], peaks: &[(f32, [f32; 3])], out: &mut Vec<f32>) {
        if peaks.is_empty() {
            return;
        }
        let verts = dsistudio_odf8::hemisphere_vertices_ras();
        let lobes = self.fmls.segment(amplitudes);
        self.vertex_lobe.clear();
        self.vertex_lobe.resize(verts.len(), u32::MAX);
        for (li, lobe) in lobes.iter().enumerate() {
            for &v in &lobe.vertices {
                self.vertex_lobe[v] = li as u32;
            }
        }
        for &(_, dir) in peaks {
            out.push(dispersion_for_dir(dir, &lobes, &self.vertex_lobe, verts));
        }
    }
}

/// Match one refined peak direction to its lobe and return that lobe's
/// dispersion. The vertex nearest the direction (by |dot|, honouring antipodal
/// symmetry) carries the watershed assignment itself, so this is exact rather
/// than an angular heuristic. Falls back to the best-aligned lobe peak for the
/// rare refined direction whose nearest vertex sampled to zero amplitude (and
/// so belongs to no lobe); NaN only if the voxel segmented to no lobes at all.
fn dispersion_for_dir(
    dir: [f32; 3],
    lobes: &[Lobe],
    vertex_lobe: &[u32],
    verts: &[[f32; 3]],
) -> f32 {
    let align = |v: [f32; 3]| (v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2]).abs();
    let mut nearest = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in verts.iter().enumerate() {
        let a = align(v);
        if a > best {
            best = a;
            nearest = i;
        }
    }
    if vertex_lobe[nearest] != u32::MAX {
        return lobes[vertex_lobe[nearest] as usize].dispersion();
    }
    lobes
        .iter()
        .max_by(|a, b| align(verts[a.peak_index]).total_cmp(&align(verts[b.peak_index])))
        .map(Lobe::dispersion)
        .unwrap_or(f32::NAN)
}

/// Write an SS3T ODX. `dwi_ref` provides the affine and is the path to the
/// input DWI NIfTI. `output_path` ends in `.odx` for archive output, or any
/// other extension for a directory tree.
pub fn write_ss3t_odx(
    output_path: &Path,
    dwi_ref: &Path,
    mask: &Array3<bool>,
    wm: &Array4<f32>,
    gm: &Array4<f32>,
    csf: &Array4<f32>,
    lmax_wm: usize,
    responses: &Ss3tResponses,
    overwrite: bool,
    directory: bool,
) -> Result<()> {
    let raw_affine = read_reference_affine(dwi_ref)
        .map_err(|e| anyhow!("read affine from {:?}: {e}", dwi_ref))?;
    write_ss3t_odx_with_affine(
        output_path, raw_affine, mask, wm, gm, csf, lmax_wm, responses, overwrite, directory,
    )
}

/// [`write_ss3t_odx`] with the voxel-to-world affine given directly instead
/// of read from the DWI NIfTI.
#[allow(clippy::too_many_arguments)]
pub fn write_ss3t_odx_with_affine(
    output_path: &Path,
    raw_affine: [[f64; 4]; 4],
    mask: &Array3<bool>,
    wm: &Array4<f32>,
    gm: &Array4<f32>,
    csf: &Array4<f32>,
    lmax_wm: usize,
    responses: &Ss3tResponses,
    overwrite: bool,
    directory: bool,
) -> Result<()> {
    write_multitissue_odx(
        output_path, raw_affine, None, mask, wm, gm, csf, lmax_wm, &responses.wm, &responses.gm,
        &responses.csf, "ss3t_responses", None, None, overwrite, directory, 0.0, 0.0, None,
    )
}

/// Write a continuous-b multi-tissue CSD ODX — identical bundle to
/// [`write_ss3t_odx`] (WM SH glyphs + GM/CSF + mask + WM peaks), with the
/// dense-b responses recorded in the header under `response_key`.
///
/// An extension point for continuous-b methods, which fit non-shelled q-space
/// against responses defined on a dense b-grid rather than per acquired shell.
/// The caller chooses `response_key` so files written by an existing method
/// keep the header key its readers expect.
///
/// `b_step` is the dense b-grid step the responses were sampled on (row `k` is
/// `b = k · b_step`). It is recorded alongside the coefficients: without it a
/// reader has no way to label the b axis and must assume the 50 s/mm² default.
#[allow(clippy::too_many_arguments)]
pub fn write_continuous_b_odx(
    output_path: &Path,
    dwi_ref: &Path,
    mask: &Array3<bool>,
    wm: &Array4<f32>,
    gm: &Array4<f32>,
    csf: &Array4<f32>,
    lmax_wm: usize,
    responses: [&TissueResponse; 3],
    response_key: &str,
    b_step: f64,
    floor: Option<&Array4<f32>>,
    overwrite: bool,
    directory: bool,
    peak_min_amplitude: f32,
    peak_min_amplitude_frac: f32,
    quant_meta: Option<&ContinuousBQuantMeta>,
) -> Result<()> {
    let [wm_response, gm_response, csf_response] = responses;
    let raw_affine = read_reference_affine(dwi_ref)
        .map_err(|e| anyhow!("read affine from {:?}: {e}", dwi_ref))?;
    write_multitissue_odx(
        output_path, raw_affine, Some(dwi_ref), mask, wm, gm, csf, lmax_wm, wm_response, gm_response,
        csf_response, response_key, Some(b_step), floor, overwrite, directory,
        peak_min_amplitude, peak_min_amplitude_frac, quant_meta,
    )
}

/// Shared multi-tissue ODX writer: packs WM SH glyphs, GM/CSF compartments, the
/// brain mask, and WM peaks, recording the three tissue responses in the header
/// under `response_key`. `b_step` is `Some` only for dense-b (continuous-b) responses,
/// whose rows lie on a uniform b-grid; shelled (SS3T) responses have one row per
/// acquired shell and no uniform step to record.
#[allow(clippy::too_many_arguments)]
fn write_multitissue_odx(
    output_path: &Path,
    raw_affine: [[f64; 4]; 4],
    // The DWI the fit came from, recorded as the BIDS `Sources` entry.
    dwi_ref: Option<&Path>,
    mask: &Array3<bool>,
    wm: &Array4<f32>,
    gm: &Array4<f32>,
    csf: &Array4<f32>,
    lmax_wm: usize,
    wm_response: &TissueResponse,
    gm_response: &TissueResponse,
    csf_response: &TissueResponse,
    response_key: &str,
    b_step: Option<f64>,
    // Flat noise-floor compartment (continuous-b only), one value per voxel.
    floor: Option<&Array4<f32>>,
    overwrite: bool,
    directory: bool,
    peak_min_amplitude: f32,
    peak_min_amplitude_frac: f32,
    quant_meta: Option<&ContinuousBQuantMeta>,
) -> Result<()> {
    // Canonicalize to RAS+ so this ODX shares voxel ordering with `odx convert`
    // output (matching the MRtrix / cs-dsi-eval references). The fODF SH and
    // peaks are stored in world (RAS) space, so only the voxel grid is reindexed
    // — per-voxel SH values are unchanged, just moved to their canonical index.
    let canon = CanonTransform::from_affine(raw_affine);
    let (wm_canon, affine) = canonicalize_array4(wm, raw_affine, &canon)?;
    let (gm_canon, _) = canonicalize_array4(gm, raw_affine, &canon)?;
    let (csf_canon, _) = canonicalize_array4(csf, raw_affine, &canon)?;
    let floor_canon = floor
        .map(|f| canonicalize_array4(f, raw_affine, &canon).map(|(a, _)| a))
        .transpose()?;
    let mask_canon = canonicalize_bool_mask(mask, raw_affine, &canon)?;
    let (wm, gm, csf, mask) = (&wm_canon, &gm_canon, &csf_canon, &mask_canon);
    let floor = floor_canon.as_ref();

    let s = mask.shape();
    let (nx, ny, nz) = (s[0], s[1], s[2]);
    if wm.shape()[..3] != [nx, ny, nz]
        || gm.shape()[..3] != [nx, ny, nz]
        || csf.shape()[..3] != [nx, ny, nz]
    {
        return Err(anyhow!(
            "tissue volumes do not share spatial shape {:?}: wm={:?} gm={:?} csf={:?}",
            [nx, ny, nz],
            wm.shape(),
            gm.shape(),
            csf.shape()
        ));
    }
    let n_wm = wm.shape()[3];
    if gm.shape()[3] != 1 || csf.shape()[3] != 1 {
        return Err(anyhow!(
            "expected GM and CSF to have one channel each (l=0); got gm={} csf={}",
            gm.shape()[3],
            csf.shape()[3]
        ));
    }
    if let Some(f) = floor {
        if f.shape()[..3] != [nx, ny, nz] || f.shape()[3] != 1 {
            return Err(anyhow!(
                "noise-floor volume must be {:?} with one channel; got {:?}",
                [nx, ny, nz],
                f.shape()
            ));
        }
    }

    // Walk the mask in C order; this is what odx-rs's per-voxel arrays expect.
    let masked: Vec<(usize, usize, usize)> = (0..nx)
        .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
        .filter(|&(i, j, k)| mask[(i, j, k)])
        .collect();
    let n_voxels = masked.len();

    let mask_bytes: Vec<u8> = (0..nx)
        .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
        .map(|(i, j, k)| if mask[(i, j, k)] { 1u8 } else { 0u8 })
        .collect();

    let mut wm_floats: Vec<f32> = Vec::with_capacity(n_voxels * n_wm);
    let mut gm_floats: Vec<f32> = Vec::with_capacity(n_voxels);
    let mut csf_floats: Vec<f32> = Vec::with_capacity(n_voxels);
    for &(i, j, k) in &masked {
        for c in 0..n_wm {
            wm_floats.push(wm[(i, j, k, c)]);
        }
        gm_floats.push(gm[(i, j, k, 0)]);
        csf_floats.push(csf[(i, j, k, 0)]);
    }

    let wm_bytes = floats_to_le_bytes(&wm_floats);
    let gm_bytes = floats_to_le_bytes(&gm_floats);
    let csf_bytes = floats_to_le_bytes(&csf_floats);

    let dimensions = [nx as u64, ny as u64, nz as u64];
    let mut builder = OdxBuilder::new(affine, dimensions, mask_bytes);
    builder.set_sh_info(lmax_wm as u64, "tournier07".to_string());
    builder.set_sh_data("coefficients", wm_bytes, n_wm, DType::Float32);
    // GM and CSF are isotropic — one l=0 amplitude per voxel, not an lmax_wm SH
    // expansion. They belong in `dpv` alongside the other per-voxel scalars
    // (`gfa`, `anisotropic_power`): every ODX SH consumer reads the fODF from
    // `sh["coefficients"]`, and odx-rs validates that *all* `sh` arrays carry
    // the header's `sh_order` column count, so filing them under `sh` fails
    // validation with "SH array 'gm' has 1 columns, expected 45 for order 8".
    builder.set_dpv_data("gm", gm_bytes, 1, DType::Float32);
    builder.set_dpv_data("csf", csf_bytes, 1, DType::Float32);
    // Flat Rician-noise-floor compartment, on the same amplitude scale as gm/csf
    // (≈0.2821 for a compartment filling the voxel). Not a tissue: it is the part
    // of the signal that does not decay with b, so it doubles as a per-voxel SNR
    // flag — large values mark voxels where the magnitude data is floor-dominated.
    if let Some(f) = floor {
        let floats: Vec<f32> = masked.iter().map(|&(i, j, k)| f[(i, j, k, 0)]).collect();
        builder.set_dpv_data("noise_floor", floats_to_le_bytes(&floats), 1, DType::Float32);
    }

    builder.set_extra_value(
        response_key,
        responses_json(wm_response, gm_response, csf_response, b_step),
    );
    if let Some(meta) = quant_meta {
        builder.set_extra_value("bids", bids_json(meta, dwi_ref, &raw_affine));
    }

    // Peaks from WM SH, mirroring cs-odf's defaults so trxviz / odx tools
    // see the same fixel structure either binary would produce.
    let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig {
        npeaks: PEAK_NPEAKS,
        relative_peak_threshold: PEAK_REL_THRESH,
        min_separation_angle_deg: PEAK_MIN_SEP_DEG,
    });
    let plan = odx_rs::mrtrix_sh::RowSamplePlan::for_sh_rows_nonnegative(
        finder.vertices(),
        n_wm,
    )
    .map_err(|e| anyhow!("sphere sampling plan: {e}"))?;
    let n_dirs = plan.ndir();

    let mut amp_per_fixel: Vec<f32> = Vec::new();
    let mut disp_per_fixel: Vec<f32> = Vec::new();
    let mut odf_scratch = vec![0.0f32; n_dirs];
    let basis = ShBasisKind::MrtrixTournier { lmax: lmax_wm };
    let dispersion = FixelDispersion::for_odf8_sphere();
    let mut disp_seg = dispersion.segmenter();
    let mut disp_scratch: Vec<f32> = Vec::with_capacity(PEAK_NPEAKS);

    // Pass 1: find peaks for every voxel, buffered flat (`counts` slices it), and
    // record each voxel's dominant-peak amplitude. The relative-threshold and
    // separation gates already ran inside the finder; the amplitude floor below
    // is the additional cut that removes shallow/spurious lobes — the
    // over-detection that grows under aggressive ridge damping.
    let mut found_flat: Vec<(f32, [f32; 3], f32)> = Vec::with_capacity(n_voxels);
    let mut counts: Vec<u32> = Vec::with_capacity(n_voxels);
    let mut vox_max: Vec<f32> = Vec::with_capacity(n_voxels);
    for v in 0..n_voxels {
        let row = &wm_floats[v * n_wm..(v + 1) * n_wm];
        plan.apply_row_into(row, &mut odf_scratch);
        let found = finder.find_peaks_with_sh(&odf_scratch, row, basis);
        disp_scratch.clear();
        disp_seg.push_voxel(&odf_scratch, &found, &mut disp_scratch);
        debug_assert_eq!(disp_scratch.len(), found.len());
        counts.push(found.len() as u32);
        let mut vmax = 0.0f32;
        for ((amp, dir), disp) in found.into_iter().zip(&disp_scratch) {
            vmax = vmax.max(amp);
            found_flat.push((amp, dir, *disp));
        }
        if vmax > 0.0 {
            vox_max.push(vmax);
        }
    }

    // Resolve the floor. The absolute form is used as given; the fractional form
    // is taken against the median dominant-peak amplitude, which rescales with
    // the fODFs — so one fraction means the same thing on either fODF scale
    // (notably with and without absolute-scale output).
    let peak_floor = if peak_min_amplitude_frac > 0.0 {
        if vox_max.is_empty() {
            0.0
        } else {
            let mid = vox_max.len() / 2;
            vox_max.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            peak_min_amplitude_frac * vox_max[mid]
        }
    } else {
        peak_min_amplitude
    };

    // Pass 2: apply the floor and emit.
    let mut cursor = 0usize;
    let mut peaks: Vec<[f32; 3]> = Vec::with_capacity(PEAK_NPEAKS);
    for &n in &counts {
        peaks.clear();
        for &(amp, dir, disp) in &found_flat[cursor..cursor + n as usize] {
            if amp < peak_floor {
                continue;
            }
            peaks.push(dir);
            amp_per_fixel.push(amp);
            disp_per_fixel.push(disp);
        }
        cursor += n as usize;
        builder.push_voxel_peaks(&peaks);
    }
    let amp_bytes = floats_to_le_bytes(&amp_per_fixel);
    builder.set_dpf_data("amplitude", amp_bytes, 1, DType::Float32);
    let disp_bytes = floats_to_le_bytes(&disp_per_fixel);
    builder.set_dpf_data("dispersion", disp_bytes, 1, DType::Float32);

    let dataset = builder
        .finalize()
        .map_err(|e| anyhow!("ODX validation failed: {e}"))?;

    if directory {
        atomic_write_directory(output_path, overwrite, |tmp_dir| {
            dataset
                .save_directory(tmp_dir)
                .map_err(|e| crate::CsDmriError::Other(format!("ODX directory write: {e}")))
        })?;
    } else {
        atomic_write(output_path, overwrite, |tmp_path| {
            dataset
                .save_archive(tmp_path)
                .map_err(|e| crate::CsDmriError::Other(format!("ODX archive write: {e}")))
        })?;
    }
    Ok(())
}

fn floats_to_le_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * std::mem::size_of::<f32>());
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// The b=0 intensity mrtrix3's `dwinormalise` scales the reference to
/// (`DEFAULT_TARGET_INTENSITY` in `lib/mrtrix3/dwinormalise/individual.py`).
///
/// A continuous-b method need not rescale to it — with absolute-scale output
/// the stored fODFs are already pinned by a final `mtnormalise`, which
/// supersedes this scalar for cross-subject comparability. The recipe "multiply by
/// `REFERENCE_B0_TARGET_INTENSITY / ReferenceB0Signal`" is therefore only valid
/// for quantities still in raw scanner units (the DWI itself, or an
/// unnormalised fit); applying it to a post-`mtnormalise` fODF would
/// double-normalise.
pub const REFERENCE_B0_TARGET_INTENSITY: f64 = 1000.0;

/// How the white-matter mask used for `ReferenceB0Signal` was defined.
///
/// mrtrix3's `dwinormalise` offers two: a group FA template thresholded at 0.4,
/// or a user-supplied mask. These are the continuous-b analogues,
/// most-preferred first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceB0Source {
    /// Mask supplied by the caller — the direct analogue of
    /// `dwinormalise individual`, and the closest match to mrtrix3 because the
    /// mask definition is identical by construction.
    UserMask,
    /// The Dhollander single-fibre WM voxels the response was estimated from.
    /// Stricter than mrtrix3's FA > 0.4: the Dhollander selection additionally gates on an
    /// eigenvalue-ratio single-fibre criterion, so this samples a purer WM
    /// population. Available only when responses are estimated, not read.
    WhiteMatterSingleFibre,
    /// Voxels where the fitted WM compartment dominates the tissue sum.
    /// The fallback when responses are read from disk, so DTI and tissue
    /// selection are skipped and no FA is available. Analogous to mrtrix3's
    /// FA-thresholded mask in intent, but derived from the multi-tissue fit
    /// rather than from anisotropy, so the two are not interchangeable.
    WhiteMatterFraction,
}

impl ReferenceB0Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserMask => "UserMaskMedian",
            Self::WhiteMatterSingleFibre => "WhiteMatterSingleFibreMedian",
            Self::WhiteMatterFraction => "WhiteMatterFractionMedian",
        }
    }
}

/// Quantitative provenance for a continuous-b ODX, serialized into the bundle as a
/// BIDS-style JSON sidecar object (see [`bids_json`]).
///
/// QST Eq. (4) only needs `AFD_ref / DWI_ref` to be *identical across
/// subjects*, and the pipeline delivers that by construction: the shared
/// group response fixes `AFD_ref`, and the final `mtnormalise` re-pins the fODF
/// scale, playing the role of `DWI_ref` equalisation. These fields are
/// therefore provenance and QC — they record what the scan's raw scale *was*
/// (e.g. to track receiver-gain drift across sessions) — not inputs any
/// downstream computation requires. Nothing downstream can reconstruct them
/// from the fODF alone, so they travel with the data.
#[derive(Debug, Clone)]
pub struct ContinuousBQuantMeta {
    /// The method that produced the ODX, recorded as the BIDS `GeneratedBy`
    /// entry.
    pub generated_by: GeneratedBy,
    /// True if the fit was rescaled to absolute signal units.
    pub quantitative: bool,
    /// The per-scan `DWI_ref`: median b=0 within a white-matter mask, paired
    /// with a label describing how that mask was defined.
    ///
    /// The statistic mirrors mrtrix3's `dwinormalise`, which normalises "the
    /// median b=0 white matter value" — `mrstats -output median` over the
    /// mean-b=0 image within a WM mask (`lib/mrtrix3/dwinormalise/`). mrtrix3
    /// derives that mask from an FA template thresholded at 0.4 (`group`) or
    /// takes it from the user (`individual`); the label records which analogue
    /// was used here. See [`REFERENCE_B0_TARGET_INTENSITY`] for the value
    /// mrtrix3 scales this to.
    ///
    /// `None` only if no white-matter voxels could be identified at all.
    pub reference_b0: Option<(f64, ReferenceB0Source)>,
    /// `mtnormalise` target sum, or `None` if normalization was skipped.
    pub intensity_normalization_target: Option<f64>,
    /// Global-gain summary of the multiplier `mtnormalise` actually applied:
    /// `1 / lognorm_scale`, i.e. `exp(-mean(log f))` over the final inlier
    /// voxels (the bias field's spatially-varying part averages out). The
    /// reciprocal of mrtrix3's `lognorm_scale` header entry, and the analogue
    /// of the `-scale` output of mrtrix3 dev's `dwinormalise mtnorm`. Together with
    /// `ReferenceB0Signal` this makes the full raw→stored scale chain
    /// auditable from the sidecar alone: stored ≈ raw × this value in a
    /// typical voxel. `None` if normalization was skipped.
    pub intensity_normalization_scale: Option<f64>,
    /// True if the responses were read from disk (i.e. potentially a
    /// group-average kernel) rather than being estimated from this scan.
    pub external_responses: bool,
    /// True if the flat Rician-noise-floor compartment was fit.
    pub noise_floor: bool,
    /// Max WM SH order.
    pub sh_degree: usize,
    /// Gradient-deviation field the fit used: every voxel was
    /// deconvolved with its own effective gradient table. `None` = nominal table.
    pub graddev: Option<std::path::PathBuf>,
    /// True if the response was interpolated linearly in b between grid rows
    /// (implied by a gradient-deviation field) rather than snapped to the nearest.
    pub b_interp: bool,
    /// Name of the dense-b response estimator used for this scan; `None` when
    /// the responses were read from disk.
    pub response_estimator: Option<String>,
}

/// BIDS `GeneratedBy` entry: the pipeline/method that wrote the derivative.
#[derive(Debug, Clone)]
pub struct GeneratedBy {
    pub name: String,
    pub version: String,
    pub description: String,
}

/// Build the BIDS-style sidecar object for a continuous-b ODX.
///
/// BIDS has no schema for FOD/AFD derivatives, so this follows the *conventions*
/// as closely as the domain allows: CamelCase keys throughout, the real BIDS
/// derivatives keys (`Description`, `Sources`, `GeneratedBy`, `Units`) used with
/// their specified meanings, `Sources` written as a BIDS URI, and everything
/// domain-specific expressed as CamelCase keys with explicit units in the name
/// (`...MM`, `...MM3`) per the BIDS practice of encoding units in the field.
fn bids_json(
    meta: &ContinuousBQuantMeta,
    dwi_ref: Option<&Path>,
    affine: &[[f64; 4]; 4],
) -> serde_json::Value {
    // Voxel dimensions are the column norms of the affine's 3x3 block.
    let vox = |c: usize| {
        (affine[0][c].powi(2) + affine[1][c].powi(2) + affine[2][c].powi(2)).sqrt()
    };
    let voxel_size_mm = [vox(0), vox(1), vox(2)];
    let mut root = serde_json::json!({
        "Description": if meta.quantitative {
            "White matter fibre orientation distribution from continuous-b \
             multi-tissue constrained spherical deconvolution, in absolute \
             signal units suitable for apparent fibre density (AFD) and SIFT2 \
             quantification."
        } else {
            "White matter fibre orientation distribution from continuous-b \
             multi-tissue constrained spherical deconvolution, normalized \
             per voxel by the b=0 signal. NOT in AFD units: see FODScaling."
        },
        "Sources": dwi_ref.map(bids_uri).into_iter().collect::<Vec<_>>(),
        "GeneratedBy": [{
            "Name": meta.generated_by.name,
            "Version": meta.generated_by.version,
            "Description": meta.generated_by.description,
        }],
        "Units": "arbitrary",
        "SphericalHarmonicBasis": "MRtrix3",
        "SphericalHarmonicDegree": meta.sh_degree,
        // The distinction that decides whether these fODFs can be compared across
        // subjects at all. "absolute" => amplitude is proportional to the DWI
        // signal (and hence to intra-axonal volume at high b), which is what AFD
        // and SIFT2 require. "PerVoxelB0" => amplitude is a fraction of each
        // voxel's own b=0, which both Smith 2022 and Dhollander 2021 rule out for
        // fibre density analysis.
        "FODScaling": if meta.quantitative { "Absolute" } else { "PerVoxelB0" },
        "QuantitativeFODScaling": meta.quantitative,
        "VoxelSizeMM": voxel_size_mm.to_vec(),
        "VoxelVolumeMM3": voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2],
        "ResponseFunctionSource": if meta.external_responses { "External" } else { "Estimated" },
        "NoiseFloorCompartment": meta.noise_floor,
        "ResponseBInterpolation": if meta.b_interp { "Linear" } else { "Nearest" },
        "GradientNonlinearityCorrection": if meta.graddev.is_some() { "PerVoxelGradientTable" } else { "None" },
    });
    let obj = root.as_object_mut().expect("object literal");
    if let Some(e) = &meta.response_estimator {
        obj.insert("ResponseEstimator".into(), serde_json::json!(e));
    }
    if let Some(gd) = &meta.graddev {
        obj.insert("GradientNonlinearityField".into(), serde_json::json!(bids_uri(gd)));
    }
    // DWI_ref for QST Eq. (4). Always emitted when a WM mask could be formed,
    // including when responses were read from disk: the group-response
    // workflow is exactly the one that needs it.
    if let Some((b0, src)) = &meta.reference_b0 {
        obj.insert("ReferenceB0Signal".into(), serde_json::json!(b0));
        obj.insert("ReferenceB0Source".into(), serde_json::json!(src.as_str()));
        obj.insert("ReferenceB0Statistic".into(), serde_json::json!("median"));
        obj.insert(
            "ReferenceB0TargetIntensity".into(),
            serde_json::json!(REFERENCE_B0_TARGET_INTENSITY),
        );
    }
    match meta.intensity_normalization_target {
        Some(t) => {
            obj.insert("IntensityNormalization".into(), serde_json::json!("mtnormalise"));
            obj.insert("IntensityNormalizationTarget".into(), serde_json::json!(t));
            // The applied multiplier, not just the target: with it, a reader
            // can undo the normalization (or audit raw→stored gain) from the
            // sidecar alone.
            if let Some(s) = meta.intensity_normalization_scale {
                obj.insert("IntensityNormalizationScale".into(), serde_json::json!(s));
            }
        }
        None => {
            obj.insert("IntensityNormalization".into(), serde_json::json!("none"));
        }
    }
    root
}

/// Best-effort BIDS URI (`bids::sub-.../...`) for `path`: the path relative to
/// the enclosing BIDS dataset root, identified by walking up to the directory
/// holding `dataset_description.json`. Falls back to the bare filename when the
/// input is not inside a recognizable BIDS tree, so the field is always a
/// relative reference and never leaks an absolute local path.
fn bids_uri(path: &Path) -> String {
    let abs = path.canonicalize();
    let p = abs.as_deref().unwrap_or(path);
    for root in p.ancestors().skip(1) {
        if root.join("dataset_description.json").is_file() {
            if let Ok(rel) = p.strip_prefix(root) {
                return format!("bids::{}", rel.to_string_lossy());
            }
        }
    }
    match p.file_name() {
        Some(n) => format!("bids::{}", n.to_string_lossy()),
        None => "bids::".to_string(),
    }
}

fn responses_json(
    wm: &TissueResponse,
    gm: &TissueResponse,
    csf: &TissueResponse,
    b_step: Option<f64>,
) -> serde_json::Value {
    let mut val = serde_json::json!({
        "wm": tissue_json(wm),
        "gm": tissue_json(gm),
        "csf": tissue_json(csf),
    });
    if let (Some(step), Some(obj)) = (b_step, val.as_object_mut()) {
        obj.insert("step".to_string(), serde_json::json!(step));
    }
    val
}

fn tissue_json(t: &TissueResponse) -> serde_json::Value {
    serde_json::json!({
        "lmax": t.lmax,
        "coeffs": t.coeffs,
    })
}

/// Default DSI-Studio ODF8 peak-finder config — matches `cs-odf`'s defaults
/// so `cs-fit --odx-output` and `cs-odf` produce interchangeable peaks.
pub const SHORE_ODX_DEFAULT_NPEAKS: usize = 5;
pub const SHORE_ODX_DEFAULT_REL_THRESH: f32 = 0.5;
pub const SHORE_ODX_DEFAULT_MIN_SEP_DEG: f32 = 25.0;

/// Peak-finder configuration for the SHORE→ODX path.
#[derive(Debug, Clone, Copy)]
pub struct ShoreOdxPeakOpts {
    pub npeaks: usize,
    pub relative_threshold: f32,
    pub min_separation_deg: f32,
}

impl Default for ShoreOdxPeakOpts {
    fn default() -> Self {
        Self {
            npeaks: SHORE_ODX_DEFAULT_NPEAKS,
            relative_threshold: SHORE_ODX_DEFAULT_REL_THRESH,
            min_separation_deg: SHORE_ODX_DEFAULT_MIN_SEP_DEG,
        }
    }
}

/// Knobs `build_shore_odx` exposes to its callers. Defaults match `cs-odf`'s
/// defaults so a one-step `cs-fit --odx-output` produces the same ODX as the
/// two-step `cs-fit … && cs-odf …` pipeline.
#[derive(Debug, Clone)]
pub struct ShoreOdxOptions {
    /// Field name under `sh/` in the ODX. `cs-odf` and trxviz both expect
    /// `"coefficients"` for SH glyph rendering — leave alone unless writing
    /// a companion array.
    pub field_name: String,
    /// DSI-Studio-style brain-wide ODF normalization (per-voxel QA = max −
    /// min, divide every voxel's SH by the brain-wide max QA so the
    /// brightest peak ends at 1).
    pub global_normalize: bool,
    /// Anisotropic-power DPV (Dell'Acqua 2014). Most ODX viewers use it as
    /// the slice background; only disable if you have your own scalar map.
    pub anisotropic_power: bool,
    pub ap_norm_factor: f64,
    /// Peak (fixel) extraction. `None` skips peaks entirely (SH-only ODX).
    pub peaks: Option<ShoreOdxPeakOpts>,
    pub progress_interval_secs: u64,
    pub quiet: bool,
}

impl Default for ShoreOdxOptions {
    fn default() -> Self {
        Self {
            field_name: "coefficients".to_string(),
            global_normalize: true,
            anisotropic_power: true,
            ap_norm_factor: ANISOTROPIC_POWER_NORM_FACTOR,
            peaks: Some(ShoreOdxPeakOpts::default()),
            progress_interval_secs: 30,
            quiet: false,
        }
    }
}

/// Side-channel results from `build_shore_odx` that callers may need to
/// continue populating the builder before finalize — e.g. cs-odf's
/// microstructure step uses `peak0_dirs` for RTAP/RTPP and
/// `masked_indices` to write sibling NIfTIs.
pub struct ShoreOdxBuild {
    pub builder: OdxBuilder,
    /// Order matches the `sh/` and `dpv/` arrays — first peak per masked
    /// voxel in canonical voxel coordinates, or `None` if this voxel's ODF
    /// has no peak (always present when `opts.peaks` is `Some`; absent when
    /// peaks were disabled).
    pub peak0_dirs: Option<Vec<Option<[f32; 3]>>>,
    /// Canonical-frame voxel indices, in the same order as the masked
    /// arrays above.
    pub masked_indices: Vec<(usize, usize, usize)>,
}

/// Project SHORE coefficients to Tournier-ordered ODF SH and populate an
/// `OdxBuilder`. Caller is responsible for the final `finalize()` +
/// `save_archive` / `save_directory` so it can attach extra DPVs (e.g.
/// microstructure) or extra header values (e.g. provenance) first.
///
/// All inputs are expected in canonical RAS+ orientation. Callers reading
/// from disk should use `canonicalize_array4`, `canonicalize_bool_mask`,
/// and `canonicalize_array3_f32` first; in-memory callers (e.g. cs-fit)
/// can do the same on their fitted arrays.
pub fn build_shore_odx(
    coeffs: &Array4<f32>,
    affine: [[f64; 4]; 4],
    mask: &Array3<bool>,
    basis: &ShoreBasis,
    lmax: u32,
    diagnostic_dpvs: &[(&str, &Array3<f32>)],
    opts: &ShoreOdxOptions,
) -> Result<ShoreOdxBuild> {
    if lmax % 2 != 0 {
        return Err(anyhow!("lmax must be even, got {lmax}"));
    }
    if lmax > basis.radial_order {
        return Err(anyhow!(
            "lmax {lmax} exceeds radial_order {}; the SHORE basis has no ℓ > radial_order blocks",
            basis.radial_order
        ));
    }
    if coeffs.shape()[3] != basis.n_coeffs() {
        return Err(anyhow!(
            "coefficients have {} channels, but SHORE(radial_order={}, ζ={}) expects {}",
            coeffs.shape()[3],
            basis.radial_order,
            basis.zeta,
            basis.n_coeffs()
        ));
    }

    let canon_dims = [coeffs.shape()[0], coeffs.shape()[1], coeffs.shape()[2]];
    if mask.shape() != [canon_dims[0], canon_dims[1], canon_dims[2]] {
        return Err(anyhow!(
            "mask shape {:?} does not match coefficient spatial shape {:?}",
            mask.shape(),
            canon_dims
        ));
    }

    let masked_indices: Vec<(usize, usize, usize)> = mask
        .indexed_iter()
        .filter_map(|((i, j, k), &m)| if m { Some((i, j, k)) } else { None })
        .collect();
    let n_voxels = masked_indices.len();
    let n_shore = basis.n_coeffs();
    let n_sh = n_tournier_sh_coeffs(lmax);

    if !opts.quiet {
        eprintln!(
            "[shore-odx] coeffs {:?}, lmax={lmax} (n_sh={n_sh}), masked voxels: {n_voxels}",
            coeffs.shape()
        );
    }

    let proj: DMatrix<f64> = shore_to_tournier_sh_matrix(basis, lmax);
    debug_assert_eq!(proj.nrows(), n_sh);

    let interval = Duration::from_secs(opts.progress_interval_secs.max(1));

    let mut sh_floats: Vec<f32> = vec![0.0; n_voxels * n_sh];
    let proj_hb = Heartbeat::new("shore-odx project", n_voxels, interval, opts.quiet);
    sh_floats
        .par_chunks_mut(n_sh)
        .zip(masked_indices.par_iter())
        .for_each_init(
            || (DVector::<f64>::zeros(n_shore), DVector::<f64>::zeros(n_sh)),
            |(shore_vec, sh_vec), (out, &(i, j, k))| {
                for c in 0..n_shore {
                    shore_vec[c] = coeffs[(i, j, k, c)] as f64;
                }
                proj.mul_to(shore_vec, sh_vec);
                for r in 0..n_sh {
                    out[r] = sh_vec[r] as f32;
                }
                proj_hb.tick();
            },
        );
    proj_hb.finish();

    if opts.global_normalize {
        let sphere = dsistudio_odf8::hemisphere_vertices_ras();
        let sphere_plan = RowSamplePlan::for_sh_rows_nonnegative(sphere, n_sh)
            .map_err(|e| anyhow!("failed to build sphere sampling plan: {e}"))?;
        let global_qa: f32 = sh_floats
            .par_chunks(n_sh)
            .map_init(
                || vec![0.0f32; sphere.len()],
                |scratch, row| voxel_qa(row, &sphere_plan, scratch),
            )
            .reduce(|| 0.0_f32, f32::max);
        if global_qa > 0.0 && global_qa.is_finite() {
            let scale = 1.0 / global_qa;
            sh_floats.par_iter_mut().for_each(|v| *v *= scale);
            if !opts.quiet {
                eprintln!(
                    "[shore-odx] global QA normalization: max(QA)={global_qa:.4e}, scale=1/{global_qa:.4e}"
                );
            }
        } else if !opts.quiet {
            eprintln!("[shore-odx] global QA is zero or non-finite; skipping normalization");
        }
    }

    let mut sh_bytes: Vec<u8> = Vec::with_capacity(sh_floats.len() * std::mem::size_of::<f32>());
    for v in &sh_floats {
        sh_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let mask_bytes: Vec<u8> = mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
    let dimensions = [canon_dims[0] as u64, canon_dims[1] as u64, canon_dims[2] as u64];

    let mut builder = OdxBuilder::new(affine, dimensions, mask_bytes);
    // `tournier07` is the canonical identifier — odx-rs's
    // ShBasisEvaluator::from_header rejects bare `"tournier"` and would
    // silently disable SH glyph rendering downstream.
    builder.set_sh_info(lmax as u64, "tournier07".to_string());
    builder.set_sh_data(&opts.field_name, sh_bytes, n_sh, DType::Float32);

    for (name, dpv) in diagnostic_dpvs {
        if dpv.shape() != mask.shape() {
            return Err(anyhow!(
                "diagnostic DPV {name:?} shape {:?} does not match coefficient spatial shape {:?}",
                dpv.shape(),
                mask.shape()
            ));
        }
        let mut bytes: Vec<u8> = Vec::with_capacity(n_voxels * std::mem::size_of::<f32>());
        for &(i, j, k) in &masked_indices {
            bytes.extend_from_slice(&dpv[(i, j, k)].to_le_bytes());
        }
        builder.set_dpv_data(name, bytes, 1, DType::Float32);
    }

    if opts.anisotropic_power {
        let mut ap_bytes: Vec<u8> = Vec::with_capacity(n_voxels * std::mem::size_of::<f32>());
        for v in 0..n_voxels {
            let row = &sh_floats[v * n_sh..(v + 1) * n_sh];
            let ap = anisotropic_power(row, lmax as usize, opts.ap_norm_factor);
            ap_bytes.extend_from_slice(&ap.to_le_bytes());
        }
        builder.set_dpv_data("anisotropic_power", ap_bytes, 1, DType::Float32);
    }

    let mut peak0_dirs: Option<Vec<Option<[f32; 3]>>> = None;

    if let Some(peak_opts) = opts.peaks.as_ref() {
        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig {
            npeaks: peak_opts.npeaks,
            relative_peak_threshold: peak_opts.relative_threshold,
            min_separation_angle_deg: peak_opts.min_separation_deg,
        });
        let plan = RowSamplePlan::for_sh_rows_nonnegative(finder.vertices(), n_sh)
            .map_err(|e| anyhow!("sphere sampling plan: {e}"))?;
        let n_dirs = plan.ndir();

        struct VoxelOut {
            peaks: Vec<[f32; 3]>,
            qa: Vec<f32>,
            raw_amp: Vec<f32>,
            disp: Vec<f32>,
            gfa: f32,
        }
        let dispersion = FixelDispersion::for_odf8_sphere();
        let peak_hb = Heartbeat::new("shore-odx peaks", n_voxels, interval, opts.quiet);
        let voxel_results: Vec<VoxelOut> = (0..n_voxels)
            .into_par_iter()
            .map_init(
                || (vec![0.0f32; n_dirs], dispersion.segmenter()),
                |(odf, disp_seg), v| {
                    let src = &sh_floats[v * n_sh..(v + 1) * n_sh];
                    plan.apply_row_into(src, odf);

                    let n = n_dirs as f32;
                    let mut m1 = 0.0f32;
                    let mut m2 = 0.0f32;
                    let mut min_v = f32::INFINITY;
                    for &x in odf.iter() {
                        m1 += x;
                        m2 += x * x;
                        if x < min_v {
                            min_v = x;
                        }
                    }
                    let gfa = if m2 > 0.0 && n > 1.0 {
                        let var = m2 - (m1 * m1) / n;
                        ((n / (n - 1.0)) * var / m2).max(0.0).sqrt()
                    } else {
                        0.0
                    };

                    let basis = ShBasisKind::MrtrixTournier { lmax: lmax as usize };
                    let found = finder.find_peaks_with_sh(odf, src, basis);
                    let mut disp = Vec::with_capacity(found.len());
                    disp_seg.push_voxel(odf, &found, &mut disp);
                    let mut peaks = Vec::with_capacity(found.len());
                    let mut qa = Vec::with_capacity(found.len());
                    let mut raw = Vec::with_capacity(found.len());
                    for (amp, dir) in found {
                        peaks.push(dir);
                        raw.push(amp);
                        qa.push((amp - min_v).max(0.0));
                    }
                    peak_hb.tick();
                    VoxelOut { peaks, qa, raw_amp: raw, disp, gfa }
                },
            )
            .collect();
        peak_hb.finish();

        peak0_dirs = Some(
            voxel_results
                .iter()
                .map(|r| r.peaks.first().copied())
                .collect(),
        );

        let mut directions: Vec<[f32; 3]> = Vec::new();
        let mut qa_per_fixel: Vec<f32> = Vec::new();
        let mut raw_amp_per_fixel: Vec<f32> = Vec::new();
        let mut disp_per_fixel: Vec<f32> = Vec::new();
        let mut gfa_per_voxel: Vec<f32> = Vec::with_capacity(n_voxels);
        for r in voxel_results {
            builder.push_voxel_peaks(&r.peaks);
            directions.extend(r.peaks);
            qa_per_fixel.extend(r.qa);
            raw_amp_per_fixel.extend(r.raw_amp);
            disp_per_fixel.extend(r.disp);
            gfa_per_voxel.push(r.gfa);
        }

        let max_qa = qa_per_fixel.iter().copied().fold(0.0f32, f32::max);
        if max_qa > 0.0 && max_qa.is_finite() {
            let inv = 1.0 / max_qa;
            for q in qa_per_fixel.iter_mut() {
                *q *= inv;
            }
        }

        let qa_bytes: Vec<u8> = qa_per_fixel.iter().flat_map(|q| q.to_le_bytes()).collect();
        builder.set_dpf_data("qa", qa_bytes, 1, DType::Float32);

        let raw_bytes: Vec<u8> = raw_amp_per_fixel
            .iter()
            .flat_map(|a| a.to_le_bytes())
            .collect();
        builder.set_dpf_data("amplitude", raw_bytes, 1, DType::Float32);

        let disp_bytes: Vec<u8> = disp_per_fixel
            .iter()
            .flat_map(|d| d.to_le_bytes())
            .collect();
        builder.set_dpf_data("dispersion", disp_bytes, 1, DType::Float32);

        let gfa_bytes: Vec<u8> = gfa_per_voxel.iter().flat_map(|g| g.to_le_bytes()).collect();
        builder.set_dpv_data("gfa", gfa_bytes, 1, DType::Float32);

        if !opts.quiet {
            eprintln!(
                "[shore-odx] extracted {} peaks across {n_voxels} voxels (npeaks≤{}, rel-thr={}, sep≥{}°); max QA={max_qa:.4e}",
                directions.len(),
                peak_opts.npeaks,
                peak_opts.relative_threshold,
                peak_opts.min_separation_deg,
            );
        }
    } else {
        for _ in 0..n_voxels {
            builder.push_voxel_peaks(&[]);
        }
    }

    Ok(ShoreOdxBuild {
        builder,
        peak0_dirs,
        masked_indices,
    })
}

/// Finalize the dataset and write it atomically. Thin convenience wrapper
/// shared by cs-fit and cs-odf. Set `directory` to write a directory tree
/// instead of a `.odx` zip archive.
pub fn finalize_and_write_odx(
    builder: OdxBuilder,
    output_path: &Path,
    overwrite: bool,
    directory: bool,
) -> Result<()> {
    let dataset = builder
        .finalize()
        .map_err(|e| anyhow!("ODX validation failed: {e}"))?;
    if directory {
        atomic_write_directory(output_path, overwrite, |tmp_dir| {
            dataset
                .save_directory(tmp_dir)
                .map_err(|e| crate::CsDmriError::Other(format!("ODX directory write: {e}")))
        })?;
    } else {
        atomic_write(output_path, overwrite, |tmp_path| {
            dataset
                .save_archive(tmp_path)
                .map_err(|e| crate::CsDmriError::Other(format!("ODX archive write: {e}")))
        })?;
    }
    Ok(())
}

/// Auto-detect a brain mask from a 4-D coefficient volume: any voxel whose
/// 4th-axis has at least one nonzero coefficient is in. Mirrors `cs-odf`'s
/// fallback when no `--mask` is supplied.
pub fn auto_mask_from_coeffs(coeffs: &Array4<f32>) -> Array3<bool> {
    let s = coeffs.shape();
    Array3::from_shape_fn((s[0], s[1], s[2]), |(i, j, k)| {
        (0..s[3]).any(|c| coeffs[(i, j, k, c)] != 0.0)
    })
}

/// Options for [`shore_coeffs_to_odx`].
#[derive(Debug, Clone)]
pub struct ShoreToOdxOptions {
    /// ODF SH order; `None` = largest even ≤ radial order.
    pub lmax: Option<u32>,
    pub odx: ShoreOdxOptions,
    /// Also embed RTOP/RTAP/RTPP/MSD/QIV/NG DPVs; `None` skips them.
    pub microstructure: Option<crate::io::microstructure::MicrostructureOptions>,
}

/// Output of [`shore_coeffs_to_odx`]: an unfinalized builder (so the caller
/// can attach provenance) plus what the caller needs to write siblings.
pub struct ShoreToOdx {
    pub build: ShoreOdxBuild,
    /// Canonical (RAS+) affine and spatial shape the ODX was built on.
    pub affine: [[f64; 4]; 4],
    pub spatial: [usize; 3],
    pub microstructure: Option<crate::io::microstructure::MicrostructureScalars>,
}

/// SHORE coefficients on their acquisition grid → canonical RAS+ ODX builder.
///
/// Canonicalizes coefficients, mask and per-voxel diagnostic DPVs with
/// `raw_affine`; uses `raw_mask` if given (it must match the coefficient grid)
/// and otherwise every voxel with a nonzero coefficient; validates `lmax`; then
/// runs [`build_shore_odx`] and, if requested, embeds microstructure DPVs. The
/// shared tail of `cs-fit --odx-output` and `cs-odf`.
pub fn shore_coeffs_to_odx(
    raw_coeffs: &Array4<f32>,
    raw_affine: [[f64; 4]; 4],
    raw_mask: Option<&Array3<bool>>,
    basis: &ShoreBasis,
    raw_diag_dpvs: &[(&str, &Array3<f32>)],
    opts: &ShoreToOdxOptions,
) -> Result<ShoreToOdx> {
    use crate::io::microstructure::{compute_microstructure, embed_microstructure_dpvs};

    let canon = CanonTransform::from_affine(raw_affine);
    let (coeffs, affine) = canonicalize_array4(raw_coeffs, raw_affine, &canon)?;
    let spatial = [coeffs.shape()[0], coeffs.shape()[1], coeffs.shape()[2]];

    let radial_order = basis.radial_order;
    let lmax = opts
        .lmax
        .unwrap_or_else(|| crate::odf::default_lmax(basis.radial_order));
    if lmax % 2 != 0 {
        return Err(anyhow!("lmax must be even, got {lmax}"));
    }
    if lmax > radial_order {
        return Err(anyhow!(
            "lmax {lmax} exceeds radial_order {radial_order}; the SHORE basis has no ℓ > radial_order blocks"
        ));
    }

    let mask = match raw_mask {
        Some(m) => {
            let canon_mask = canonicalize_bool_mask(m, raw_affine, &canon)?;
            if canon_mask.shape() != spatial {
                return Err(anyhow!(
                    "mask canonical shape {:?} does not match coefficient spatial shape {:?}",
                    canon_mask.shape(),
                    spatial
                ));
            }
            canon_mask
        }
        None => auto_mask_from_coeffs(&coeffs),
    };

    let canon_dpvs: Vec<(&str, Array3<f32>)> = raw_diag_dpvs
        .iter()
        .map(|(name, arr)| Ok((*name, canonicalize_array3_f32(arr, raw_affine, &canon)?)))
        .collect::<Result<_>>()?;
    let dpv_refs: Vec<(&str, &Array3<f32>)> = canon_dpvs.iter().map(|(n, a)| (*n, a)).collect();

    let mut build = build_shore_odx(&coeffs, affine, &mask, basis, lmax, &dpv_refs, &opts.odx)?;

    let microstructure = opts.microstructure.as_ref().map(|mopts| {
        let scalars = compute_microstructure(
            basis,
            &coeffs,
            &build.masked_indices,
            build.peak0_dirs.as_deref(),
            mopts,
        );
        embed_microstructure_dpvs(&mut build.builder, &scalars);
        scalars
    });

    Ok(ShoreToOdx {
        build,
        affine,
        spatial,
        microstructure,
    })
}

/// Reorient a 4-D float volume to canonical RAS+ alongside its affine.
pub fn canonicalize_array4(
    raw: &Array4<f32>,
    raw_affine: [[f64; 4]; 4],
    canon: &CanonTransform,
) -> Result<(Array4<f32>, [[f64; 4]; 4])> {
    let raw_dims = raw.shape().to_vec();
    // A borrowed view when `raw` is already C-order; one copy otherwise.
    let standard = raw.as_standard_layout();
    let raw_flat = standard.as_slice().expect("standard layout is contiguous");
    let (canon_dims, canon_aff, canon_flat) = canon.apply(&raw_dims, raw_affine, raw_flat);
    drop(standard);
    let arr = Array4::from_shape_vec(
        (canon_dims[0], canon_dims[1], canon_dims[2], canon_dims[3]),
        canon_flat,
    )
    .map_err(|e| anyhow!("coeffs reshape after canonicalize: {e}"))?;
    Ok((arr, canon_aff))
}

/// Reorient a 3-D float volume to canonical RAS+ under the given transform.
pub fn canonicalize_array3_f32(
    raw: &Array3<f32>,
    raw_affine: [[f64; 4]; 4],
    canon: &CanonTransform,
) -> Result<Array3<f32>> {
    let raw_dims = raw.shape().to_vec();
    let standard = raw.as_standard_layout();
    let raw_flat = standard.as_slice().expect("standard layout is contiguous");
    let (canon_dims, _, canon_flat) = canon.apply(&raw_dims, raw_affine, raw_flat);
    drop(standard);
    Array3::from_shape_vec((canon_dims[0], canon_dims[1], canon_dims[2]), canon_flat)
        .map_err(|e| anyhow!("3-D reshape after canonicalize: {e}"))
}

/// Reorient a boolean mask to canonical RAS+ under the given transform.
pub fn canonicalize_bool_mask(
    raw: &Array3<bool>,
    raw_affine: [[f64; 4]; 4],
    canon: &CanonTransform,
) -> Result<Array3<bool>> {
    let raw_dims = raw.shape().to_vec();
    let standard = raw.as_standard_layout();
    let raw_flat = standard.as_slice().expect("standard layout is contiguous");
    let (canon_dims, _, canon_flat) = canon.apply(&raw_dims, raw_affine, raw_flat);
    drop(standard);
    Array3::from_shape_vec((canon_dims[0], canon_dims[1], canon_dims[2]), canon_flat)
        .map_err(|e| anyhow!("mask reshape after canonicalize: {e}"))
}

/// Read the affine of the input DWI (the SH/peak voxel grid) directly. Thin
/// re-export of `odx_rs::reference_affine::read_reference_affine` so callers
/// don't need to import odx-rs.
pub fn read_reference_affine_for_shore(path: &Path) -> Result<[[f64; 4]; 4]> {
    read_reference_affine(path).map_err(|e| anyhow!("read affine from {:?}: {e}", path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(coeffs: Vec<Vec<f64>>, lmax: usize) -> TissueResponse {
        TissueResponse { coeffs, lmax }
    }

    fn generated_by() -> GeneratedBy {
        GeneratedBy {
            name: "continuous-b-test".into(),
            version: "0.0.0".into(),
            description: "Continuous-b multi-tissue CSD for non-shelled q-space.".into(),
        }
    }

    fn quant_meta() -> ContinuousBQuantMeta {
        ContinuousBQuantMeta {
            generated_by: generated_by(),
            quantitative: true,
            reference_b0: Some((1234.5, ReferenceB0Source::WhiteMatterSingleFibre)),
            intensity_normalization_target: Some(0.2821),
            intensity_normalization_scale: Some(0.0042),
            external_responses: true,
            noise_floor: true,
            sh_degree: 8,
            graddev: None,
            b_interp: false,
            response_estimator: None,
        }
    }

    /// 2 x 2 x 2.5 mm voxels on an RAS-ish affine.
    fn affine_2mm() -> [[f64; 4]; 4] {
        [
            [2.0, 0.0, 0.0, -90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.5, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// The sidecar carries the terms QST Eq. (4) needs and that nothing
    /// downstream can recover from the fODF alone: the per-scan DWI_ref, the
    /// voxel volume, and — above all — whether the fODF is on the absolute scale
    /// that makes it AFD at all.
    #[test]
    fn bids_json_records_the_quantitative_scaling_terms() {
        let j = bids_json(&quant_meta(), Some(Path::new("/data/sub-01_dwi.nii.gz")), &affine_2mm());

        assert_eq!(j["FODScaling"], serde_json::json!("Absolute"));
        assert_eq!(j["QuantitativeFODScaling"], serde_json::json!(true));
        assert_eq!(j["ReferenceB0Signal"], serde_json::json!(1234.5));
        assert_eq!(j["ReferenceB0Source"], serde_json::json!("WhiteMatterSingleFibreMedian"));
        // Statistic and target intensity make the value reproducible against
        // mrtrix3's `dwinormalise` without the reader having to guess.
        assert_eq!(j["ReferenceB0Statistic"], serde_json::json!("median"));
        assert_eq!(j["ReferenceB0TargetIntensity"], serde_json::json!(1000.0));
        assert_eq!(j["VoxelSizeMM"], serde_json::json!([2.0, 2.0, 2.5]));
        assert_eq!(j["VoxelVolumeMM3"], serde_json::json!(10.0));
        assert_eq!(j["IntensityNormalizationTarget"], serde_json::json!(0.2821));
        assert_eq!(j["ResponseFunctionSource"], serde_json::json!("External"));
        assert_eq!(j["SphericalHarmonicBasis"], serde_json::json!("MRtrix3"));
        assert_eq!(j["SphericalHarmonicDegree"], serde_json::json!(8));
        // Real BIDS derivative keys, used with their specified meanings.
        assert!(j["Description"].as_str().unwrap().contains("absolute"));
        assert_eq!(j["GeneratedBy"][0]["Name"], serde_json::json!("continuous-b-test"));
        assert!(j["Sources"][0].as_str().unwrap().starts_with("bids::"));

        // Every key must be CamelCase — the one BIDS convention that applies
        // uniformly to fields the spec does not itself define.
        for k in j.as_object().unwrap().keys() {
            let c = k.chars().next().unwrap();
            assert!(c.is_ascii_uppercase(), "key {k:?} is not CamelCase");
            assert!(!k.contains('_') && !k.contains('-'), "key {k:?} is not CamelCase");
        }
    }

    /// Non-quantitative output must say so unambiguously: a consumer that treats
    /// per-voxel-b0-normalized fODFs as AFD gets silently wrong fibre density.
    #[test]
    fn bids_json_flags_non_quantitative_output_and_omits_absent_terms() {
        let meta = ContinuousBQuantMeta {
            generated_by: generated_by(),
            quantitative: false,
            reference_b0: None,
            intensity_normalization_target: None,
            intensity_normalization_scale: None,
            external_responses: false,
            ..quant_meta()
        };
        let j = bids_json(&meta, Some(Path::new("/data/sub-01_dwi.nii.gz")), &affine_2mm());

        assert_eq!(j["FODScaling"], serde_json::json!("PerVoxelB0"));
        assert_eq!(j["QuantitativeFODScaling"], serde_json::json!(false));
        assert!(j["Description"].as_str().unwrap().contains("NOT in AFD units"));
        assert_eq!(j["IntensityNormalization"], serde_json::json!("none"));
        assert_eq!(j["ResponseFunctionSource"], serde_json::json!("Estimated"));
        // Absent rather than zero/null: a group-average kernel has no per-scan
        // WM b=0, and a fabricated 0.0 would be read as a real reference.
        // Absent only when NO white-matter voxels could be found at all. With
        // responses read from disk a WM-fraction fallback supplies it, so the
        // group-response workflow is not missing DWI_ref.
        assert!(j.get("ReferenceB0Signal").is_none(), "{j}");
        assert!(j.get("ReferenceB0Source").is_none(), "{j}");
        assert!(j.get("ReferenceB0TargetIntensity").is_none(), "{j}");
        assert!(j.get("IntensityNormalizationTarget").is_none(), "{j}");
        assert!(j.get("IntensityNormalizationScale").is_none(), "{j}");
    }

    /// The sidecar must record the scaling mtnormalise *applied*, not just the
    /// target it aimed for: with ReferenceB0Signal (raw WM b=0) and
    /// IntensityNormalizationScale (raw→stored multiplier) both present, the
    /// full scale chain is auditable — and invertible — from the sidecar alone.
    #[test]
    fn bids_json_records_the_applied_normalization_scale() {
        let j = bids_json(&quant_meta(), Some(Path::new("/data/sub-01_dwi.nii.gz")), &affine_2mm());
        assert_eq!(j["IntensityNormalization"], serde_json::json!("mtnormalise"));
        assert_eq!(j["IntensityNormalizationScale"], serde_json::json!(0.0042));
        // The chain: raw b=0 (ReferenceB0Signal) × Scale ≈ stored-space b=0,
        // pinned near Target. Both ends must be in the same sidecar.
        assert_eq!(j["ReferenceB0Signal"], serde_json::json!(1234.5));
        assert_eq!(j["IntensityNormalizationTarget"], serde_json::json!(0.2821));

        // A normalised bundle whose scale is unknown is not fully auditable —
        // the target-only case still says "mtnormalise" but omits the scale.
        let meta = ContinuousBQuantMeta {
            intensity_normalization_scale: None,
            ..quant_meta()
        };
        let j = bids_json(&meta, Some(Path::new("/data/sub-01_dwi.nii.gz")), &affine_2mm());
        assert_eq!(j["IntensityNormalization"], serde_json::json!("mtnormalise"));
        assert!(j.get("IntensityNormalizationScale").is_none(), "{j}");
    }

    /// A path outside any BIDS tree must still yield a relative URI — never an
    /// absolute local path, which would leak the filesystem layout into a file
    /// that gets shared.
    #[test]
    fn bids_uri_falls_back_to_filename_outside_a_bids_tree() {
        let u = bids_uri(Path::new("/some/scratch/dir/sub-01_dwi.nii.gz"));
        assert_eq!(u, "bids::sub-01_dwi.nii.gz");
        assert!(!u.contains("/some/scratch"), "leaked absolute path: {u}");
    }

    /// A lobe `|d·axis|^p` with `axis` an exact sphere vertex has unit peak
    /// and integral `4π/(p+1)`, so per-fixel dispersion is known analytically.
    /// Sweeping the axis over every hemisphere vertex is the load-bearing
    /// part: a lobe centred near the antipodal rim gets severed (integral
    /// halved → dispersion halved) if the adjacency fails to wrap, and an
    /// orientation-dependent quadrature bias would surface as outliers at
    /// specific axes. Neither failure is visible at any single "nice" axis.
    #[test]
    fn fixel_dispersion_matches_analytic_lobes_at_every_axis() {
        let verts = dsistudio_odf8::hemisphere_vertices_ras();
        let ctx = FixelDispersion::for_odf8_sphere();
        let mut seg = ctx.segmenter();
        for p in [4i32, 16] {
            let expect = 4.0 * std::f32::consts::PI / (p as f32 + 1.0);
            for k in (0..verts.len()).step_by(3) {
                let axis = verts[k];
                let amps: Vec<f32> = verts
                    .iter()
                    .map(|d| (d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2]).abs().powi(p))
                    .collect();
                let mut out = Vec::new();
                seg.push_voxel(&amps, &[(1.0, axis)], &mut out);
                assert_eq!(out.len(), 1);
                assert!(
                    (out[0] - expect).abs() < 0.02 * expect,
                    "axis {k} p={p}: disp {} vs analytic {expect}",
                    out[0]
                );
            }
        }
    }

    /// Two crossing lobes of different widths: each fixel must pick up its
    /// *own* lobe's dispersion (broad ≠ sharp), and a flipped peak direction
    /// must match the same lobe — fixel directions are axial, so the sign
    /// carries no information.
    #[test]
    fn fixel_dispersion_matches_peaks_to_their_own_lobes() {
        let verts = dsistudio_odf8::hemisphere_vertices_ras();
        // Two near-perpendicular vertex axes, so each lobe peaks exactly on a
        // vertex and the analytic single-lobe values still approximately hold.
        let a1 = verts[0];
        let a2 = *verts
            .iter()
            .min_by(|u, v| {
                let d = |w: &[f32; 3]| (w[0] * a1[0] + w[1] * a1[1] + w[2] * a1[2]).abs();
                d(u).total_cmp(&d(v))
            })
            .unwrap();
        let amps: Vec<f32> = verts
            .iter()
            .map(|d| {
                let c1 = (d[0] * a1[0] + d[1] * a1[1] + d[2] * a1[2]).abs();
                let c2 = (d[0] * a2[0] + d[1] * a2[1] + d[2] * a2[2]).abs();
                c1.powi(16) + c2.powi(4)
            })
            .collect();
        let ctx = FixelDispersion::for_odf8_sphere();
        let mut seg = ctx.segmenter();
        let peaks = [(1.0, a1), (1.0, a2), (1.0, [-a2[0], -a2[1], -a2[2]])];
        let mut out = Vec::new();
        seg.push_voxel(&amps, &peaks, &mut out);
        assert_eq!(out.len(), 3, "one dispersion per fixel, in fixel order");
        let (sharp, broad, broad_flipped) = (out[0], out[1], out[2]);
        assert!(
            broad > sharp,
            "broad lobe must disperse more: broad={broad} sharp={sharp}"
        );
        assert_eq!(
            broad.to_bits(),
            broad_flipped.to_bits(),
            "antipodal peak directions are the same fixel axis"
        );
        // Watershed splits the crossing region between the lobes, so the
        // single-lobe analytic values hold only loosely — but each fixel must
        // still land near its own lobe's value, not its neighbour's.
        let e_sharp = 4.0 * std::f32::consts::PI / 17.0;
        let e_broad = 4.0 * std::f32::consts::PI / 5.0;
        assert!((sharp - e_sharp).abs() < 0.25 * e_sharp, "sharp {sharp} vs {e_sharp}");
        assert!((broad - e_broad).abs() < 0.25 * e_broad, "broad {broad} vs {e_broad}");
    }

    /// A voxel with no accepted peaks contributes no dispersion entries — the
    /// dpf array must stay in lockstep with the fixel list.
    #[test]
    fn fixel_dispersion_emits_nothing_for_peakless_voxels() {
        let verts = dsistudio_odf8::hemisphere_vertices_ras();
        let ctx = FixelDispersion::for_odf8_sphere();
        let mut seg = ctx.segmenter();
        let amps = vec![1.0f32; verts.len()];
        let mut out = Vec::new();
        seg.push_voxel(&amps, &[], &mut out);
        assert!(out.is_empty());
    }

    /// End-to-end plumbing through the real `cs-odf` writer: SHORE coeffs →
    /// SH projection → peak finding → dispersion → `finalize()` → read back.
    /// The `dispersion` dpf must exist alongside `amplitude`/`qa`, carry one
    /// row per fixel (so `finalize`'s dpf-row validation passes), and hold only
    /// finite, positive ratios. The two voxels are the sign extremes of the
    /// `(n=2, l=2, m=0)` term: a negative (equatorial "pancake") lobe, broad →
    /// large dispersion, and a positive (axial) lobe, tight → small dispersion.
    #[test]
    fn build_shore_odx_emits_dispersion_dpf_alongside_amplitude() {
        let basis = ShoreBasis::new(6, 700.0);
        let n = basis.n_coeffs();
        let affine = [
            [2.0, 0.0, 0.0, -90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // Index 5 is (n=2, l=2, m=0) — the z-symmetric anisotropy. Both signs
        // are verified to yield ≥1 peak through this path.
        let mut coeffs = Array4::<f32>::zeros((2, 1, 1, n));
        for v in 0..2 {
            coeffs[(v, 0, 0, 0)] = 3.0;
        }
        coeffs[(0, 0, 0, 5)] = -3.0;
        coeffs[(1, 0, 0, 5)] = 2.0;
        let mask = Array3::from_elem((2, 1, 1), true);

        let opts = ShoreOdxOptions { quiet: true, ..Default::default() };
        let built = build_shore_odx(&coeffs, affine, &mask, &basis, 6, &[], &opts).unwrap();
        let ds = built.builder.finalize().expect("finalize validates dpf rows");

        assert!(
            ds.dpf_names().contains(&"dispersion"),
            "dpf arrays: {:?}",
            ds.dpf_names()
        );
        let amp = ds.dpf::<f32>("amplitude").unwrap();
        let disp = ds.dpf::<f32>("dispersion").unwrap();
        assert_eq!(disp.nrows(), amp.nrows(), "one dispersion per fixel");
        assert!(disp.nrows() > 0, "test voxels must produce fixels");
        for r in 0..disp.nrows() {
            let d = disp.row(r)[0];
            assert!(d.is_finite() && d > 0.0, "fixel {r} dispersion {d} not positive-finite");
        }
    }

    /// The response block is the contract with trxviz's continuous-b response
    /// panel: it keys off `step` to label the b axis, and falls back to the 50
    /// s/mm² default when absent. Dense-b (continuous-b) responses must carry it.
    #[test]
    fn responses_json_records_b_step_only_for_dense_b() {
        let wm = resp(vec![vec![3.5, 0.0], vec![2.0, -0.4]], 2);
        let gm = resp(vec![vec![3.5], vec![2.5]], 0);
        let csf = resp(vec![vec![3.5], vec![1.0]], 0);

        let dense = responses_json(&wm, &gm, &csf, Some(25.0));
        assert_eq!(dense["step"], serde_json::json!(25.0));
        assert_eq!(dense["wm"]["lmax"], serde_json::json!(2));
        assert_eq!(dense["wm"]["coeffs"][1][1], serde_json::json!(-0.4));

        // Shelled (SS3T) rows are per-acquired-shell — no uniform step exists.
        let ss3t = responses_json(&wm, &gm, &csf, None);
        assert!(ss3t.get("step").is_none(), "{ss3t}");
    }
}

fn voxel_qa(sh_row: &[f32], sphere_plan: &RowSamplePlan, scratch: &mut [f32]) -> f32 {
    sphere_plan.apply_row_into(sh_row, scratch);
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &amp in scratch.iter() {
        if amp < lo {
            lo = amp;
        }
        if amp > hi {
            hi = amp;
        }
    }
    if hi.is_finite() && lo.is_finite() {
        (hi - lo).max(0.0)
    } else {
        0.0
    }
}
