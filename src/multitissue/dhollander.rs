// SPDX-License-Identifier: MPL-2.0
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file is a Rust port of the voxel-selection stages of MRtrix3's
 * `dwi2response dhollander` (python/mrtrix3/commands/dwi2response/dhollander.py)
 * together with the supporting `mrthreshold` / `Filter::OptimalThreshold` /
 * `Filter::Erode` / `Math::median` primitives it invokes
 * (cpp/cmd/mrthreshold.cpp, cpp/core/filter/optimal_threshold.h,
 * cpp/core/filter/erode.h, cpp/core/math/median.h).
 *
 * Copyright (c) 2008-2026 the MRtrix3 contributors (original work).
 * Copyright (c) 2026 the PennLINC developers team (port).
 *
 * Covered Software is provided under this License on an "as is" basis,
 * without warranty of any kind. See the Mozilla Public License v. 2.0.
 *
 * NOTE ON LICENSING: the rest of cs_dmri is distributed under the terms in
 * ./LICENSE. MPL-2.0 is a file-scoped copyleft, so keeping the ported logic
 * confined to THIS file lets the two coexist: this file stays MPL-2.0 and its
 * source must remain available; the files that merely call into it do not
 * become MPL-covered. Do not copy chunks of this file into other modules.
 */

//! MRtrix-faithful Dhollander tissue voxel selection.
//!
//! [`super::response_estimation`]'s original selection was a threshold triple
//! (top-N%-MD ⇒ CSF, high-FA ⇒ WM, remainder ⇒ GM) written from the 2016 ISMRM
//! abstract. That admits partial-volume voxels into CSF, which shows up
//! downstream as a CSF response amplitude well below MRtrix's on identical
//! data. This module instead reproduces the actual staged algorithm MRtrix
//! runs, whose discriminating feature is the **signal decay metric** (SDM)
//! rather than tensor-derived MD:
//!
//! ```text
//! preparation → erode mask, SDM per voxel, drop erroneous voxels
//! crude       → FA splits WM from GM+CSF; optimal threshold on SDM splits CSF from GM
//! refined     → WM sheds high-SDM outliers (which are re-offered to CSF);
//!               GM and CSF shed their partial-volume tails
//! final       → CSF = top csf% by SDM; GM = the gm% closest to the refined-GM
//!               median SDM; WM single-fibre = see [`select_sfwm_by_fa`]
//! ```
//!
//! The SDM is `mean_over_shells( log(S̄_b0 / S̄_b) )`, volume-weighted — a
//! monotone stand-in for `b·ADC` that needs no tensor fit and so is not
//! confounded by the tensor model breaking down in free water. Selecting CSF as
//! the top decile of an SDM-refined population, rather than the top 2.5% of MD,
//! is what excludes the partial-volume voxels.
//!
//! Deviations from MRtrix, all documented at their site:
//!   * FA comes from cs_dmri's existing RESTORE tensor fit rather than a
//!     freshly-run `dwi2tensor` restricted to the safe mask.
//!   * Shells are cs_dmri's b-value clusters (see [`sdm_shells`]) so the metric
//!     is also defined for non-shelled CS-DSI schemes.
//!   * Which volumes count as b=0 follows cs_dmri's `DEFAULT_B0_THRESHOLD`
//!     (50 s/mm², dipy's convention). MRtrix's `BZeroThreshold` is 22.5 on the
//!     dev branch and was 10.0 through 3.0.x. Immaterial on data acquired at
//!     exactly b=0, but on a scheme with b≈30 "b=0" volumes the three
//!     conventions build the SDM's reference from different volumes.
//!   * The single-fibre WM stage is FA-ranked ([`select_sfwm_by_fa`]),
//!     equivalent to MRtrix's `-wm_algo fa`, not the built-in 2019 CSD metric.
//!
//! ## Measured against MRtrix
//!
//! Ported from MRtrix3's `dev` branch (post-3.0.4), and the
//! numbers below hold for both dev and 3.0.x. Everything this port touches is
//! semantically identical between the two — the `dhollander.py` command
//! pipeline, `mrthreshold`'s `calculate`/`get_data`/`apply`,
//! `Filter::OptimalThreshold`, `Math::golden_section_search`, `Filter::Erode`
//! and `Math::median` differ only by clang-format, `#pragma once`,
//! `std::string_view` and equivalent casts. (The October 2025 `Math::median`
//! commit fixed `quantile()`, which `dhollander` never calls.) Confirmed by
//! running both: `dwi2response dhollander` from 3.0.8-2097-g99963980 and from
//! 3.0.4 give identical stage counts and responses agreeing to 3e-14 relative.
//! The one dev change reaching this algorithm is `BZeroThreshold` 10.0 → 22.5,
//! noted above.
//!
//! Stage counts on QST synthetic sub-0001a ses-1 (same DWI, same mask, all
//! defaults):
//!
//! | stage | MRtrix | this port |
//! |---|---|---|
//! | eroded mask | 161512 | **161512** |
//! | safe (post-SDM) | 161164 | **161164** |
//! | crude WM | 54277 | 67120 |
//! | crude GM / CSF | 90345 / 16542 | 79375 / 14669 |
//! | refined WM / GM / CSF | 48994 / 53490 / 3797 | 60342 / 49127 / 3401 |
//! | final SFWM / GM / CSF | 245 / 1070 / 380 | 302 / 983 / 340 |
//!
//! Preparation is exact — erosion, the SDM, and the erroneous-voxel rejection
//! agree to the voxel. Everything downstream inherits one difference: the crude
//! split puts 41.6% of the safe mask in WM where MRtrix puts 33.7%, because
//! cs_dmri's FA comes from RESTORE and MRtrix's from its own `dwi2tensor`.
//! **That tensor fit, not anything in this file, is the remaining lever on
//! exact stage-count parity.**
//!
//! It is the b-range, not the estimator. Measured on the same scan:
//!
//! | tensor fit | RESTORE median FA / frac > 0.2 | `dwi2tensor` | r | median Δ |
//! |---|---|---|---|---|
//! | all shells (b ≤ 3000) | 0.148 / 0.360 | 0.117 / 0.278 | 0.960 | +0.021 |
//! | b ≤ 1000 only | 0.115 / 0.260 | 0.114 / 0.259 | 0.996 | −0.000 |
//!
//! On single-shell data the two estimators are interchangeable. The divergence
//! appears only once b = 2000/3000 enter the fit, where a mono-exponential
//! tensor is misspecified and the two disagree about what to do: RESTORE
//! downweights the offending shells and sharpens the tensor, `dwi2tensor`'s
//! plain IWLS absorbs them and flattens it. Neither is right — DTI does not
//! describe b = 3000 — so *swapping estimators would not fix this*; limiting
//! the tensor's b-range would. Note that MRtrix's own dhollander fits its
//! tensor to all shells, so b ≤ 1000 lands us at 0.260 against its 0.278, close
//! but from the other side; exact stage-count parity and the better tensor are
//! not the same target. `--dh-fa 0.242` buys the parity on this scan if that
//! is what you want.
//!
//! Visible in the same measurement, and *not* a problem: RESTORE flags a median
//! 28.5% of measurements per voxel as outliers here (25.9% at b ≤ 1000), well
//! above the few per cent the method nominally assumes. The fit is unaffected —
//! on single-shell data it reproduces unweighted IWLS to r = 0.996 — so the
//! reweighting is spreading itself roughly evenly rather than distorting
//! anything, and on the noisy CS-DSI data this codebase targets a high
//! rejection rate is expected. The only real consequence is that
//! `cs-dti --output-outlier-fraction` is close to useless as a QC channel: it
//! sits at 0.25–0.33 for 95% of voxels (IQR 0.035) and never exceeds 0.5, so it
//! cannot single out a corrupted voxel. Don't read that map as one.
//!
//! The populations are nevertheless equivalent where it counts — the responses
//! they train match MRtrix's at every shell (CSF/WM b=0 amplitude 3.20 vs 3.16,
//! GM/WM 1.24 vs 1.21, and per-tissue decay shapes within a few per cent),
//! which is what the old MD-based selection got wrong (CSF/WM 1.88).

use ndarray::Array3;

use crate::dti::DtiVolumeResult;
use crate::io::dwi::DwiData;
use crate::{CsDmriError, Result};

/// Tuning for [`select_voxels`]. Defaults are MRtrix's `dwi2response
/// dhollander` defaults.
#[derive(Debug, Clone, Copy)]
pub struct DhollanderSelectConfig {
    /// Erosion passes applied to the brain mask before anything else.
    pub erode: usize,
    /// FA threshold for the crude WM vs GM-CSF split.
    pub fa: f64,
    /// Final single-fibre WM voxels, as a percentage of refined WM.
    pub sfwm_pct: f64,
    /// Final GM voxels, as a percentage of refined GM.
    pub gm_pct: f64,
    /// Final CSF voxels, as a percentage of refined CSF.
    pub csf_pct: f64,
    /// b-value clustering tolerance (s/mm²) for the SDM shells.
    pub shell_tolerance: f64,
    /// SDM shells with fewer volumes than this are merged into their nearest
    /// neighbour, so a single noisy direction cannot veto a voxel. Irrelevant
    /// for properly shelled data; load-bearing for continuous-b schemes.
    pub min_shell_volumes: usize,
}

impl Default for DhollanderSelectConfig {
    fn default() -> Self {
        Self {
            erode: 3,
            fa: 0.2,
            sfwm_pct: 0.5,
            gm_pct: 2.0,
            csf_pct: 10.0,
            shell_tolerance: 100.0,
            min_shell_volumes: 6,
        }
    }
}

/// Voxel counts surviving each stage — the same numbers MRtrix prints, and the
/// first thing to look at when a selection comes out wrong.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct DhollanderStageCounts {
    pub mask: usize,
    pub eroded: usize,
    pub safe: usize,
    pub crude_wm: usize,
    pub crude_gm: usize,
    pub crude_csf: usize,
    pub refined_wm: usize,
    pub refined_gm: usize,
    pub refined_csf: usize,
    pub sfwm: usize,
    pub gm: usize,
    pub csf: usize,
}

/// The three final voxel populations plus the intermediate state a caller may
/// want (the SDM itself, the safe mask, and refined WM for an alternative
/// single-fibre stage).
#[derive(Debug, Clone)]
pub struct DhollanderSelection {
    pub sfwm: Array3<bool>,
    pub gm: Array3<bool>,
    pub csf: Array3<bool>,
    pub refined_wm: Array3<bool>,
    pub safe_mask: Array3<bool>,
    /// Signal decay metric, zero outside [`Self::safe_mask`], capped at 10.
    pub safe_sdm: Array3<f64>,
    pub counts: DhollanderStageCounts,
}

/// Run the full staged selection.
pub fn select_voxels(
    dwi: &DwiData,
    dti: &DtiVolumeResult,
    cfg: &DhollanderSelectConfig,
) -> Result<DhollanderSelection> {
    let s = dwi.data.shape();
    let dims = (s[0], s[1], s[2]);

    // ---------------- Preparation ----------------
    let n_mask = count(&dwi.mask);
    if n_mask == 0 {
        return Err(CsDmriError::Other(
            "dhollander selection: brain mask is empty".into(),
        ));
    }
    let eroded = erode_mask(&dwi.mask, cfg.erode);
    let n_eroded = count(&eroded);
    if n_eroded == 0 {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: eroding the brain mask {} pass(es) left no voxels ({} before) — lower the erosion count",
            cfg.erode, n_mask
        )));
    }

    let (full_sdm, erroneous) = signal_decay_metric(dwi, &eroded, cfg)?;

    // safe_mask = eroded mask minus voxels where any shell mean or the SDM
    // itself came out non-finite or non-positive.
    let mut safe_mask = Array3::<bool>::default(dims);
    let mut safe_sdm = Array3::<f64>::zeros(dims);
    ndarray::Zip::from(&mut safe_mask)
        .and(&mut safe_sdm)
        .and(&eroded)
        .and(&erroneous)
        .and(&full_sdm)
        .for_each(|keep, sdm, &in_mask, &bad, &value| {
            if in_mask && !bad {
                *keep = true;
                *sdm = value.min(10.0);
            }
        });
    let n_safe = count(&safe_mask);
    if n_safe == 0 {
        return Err(CsDmriError::Other(
            "dhollander selection: no voxels survived the signal-decay-metric validity check — \
             check that the DWI has a usable b=0 and that the mask is inside the brain"
                .into(),
        ));
    }

    // ---------------- Crude segmentation ----------------
    // WM vs everything else, on FA alone.
    let mut crude_wm = Array3::<bool>::default(dims);
    let mut crude_nonwm = Array3::<bool>::default(dims);
    for x in 0..dims.0 {
        for y in 0..dims.1 {
            for z in 0..dims.2 {
                if !safe_mask[(x, y, z)] {
                    continue;
                }
                if dti.fa[(x, y, z)] as f64 > cfg.fa {
                    crude_wm[(x, y, z)] = true;
                } else {
                    crude_nonwm[(x, y, z)] = true;
                }
            }
        }
    }
    let n_crude_wm = count(&crude_wm);
    let n_crude_nonwm = count(&crude_nonwm);
    if n_crude_wm == 0 {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: no voxel exceeds the crude WM FA threshold ({}) — check the tensor fit",
            cfg.fa
        )));
    }
    if n_crude_nonwm == 0 {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: every voxel exceeds the crude WM FA threshold ({}) — no GM/CSF candidates",
            cfg.fa
        )));
    }

    // GM vs CSF, by an optimal threshold on the SDM within the non-WM voxels.
    //
    // The median subtraction is MRtrix's and is kept for exact correspondence,
    // but it does not move the answer: the Ridgway cost is shift-equivariant
    // (subtracting a constant shifts every candidate threshold and the search
    // bracket by that same constant, leaving the induced masks identical), so
    // it only buys precision headroom in MRtrix's float32 search. Ours is
    // float64. Don't "simplify" it away — the `-neg` in the refined-GM low
    // branch below is a real transformation and looks the same at a glance.
    let nonwm_median = masked_median(&safe_sdm, &crude_nonwm);
    let crude_csf = threshold_optimal(&safe_sdm, &crude_nonwm, nonwm_median, false)?;
    let crude_gm = difference(&crude_nonwm, &crude_csf);
    let n_crude_csf = count(&crude_csf);
    let n_crude_gm = count(&crude_gm);
    if n_crude_csf == 0 || n_crude_gm == 0 {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: crude GM/CSF split degenerate ({} GM, {} CSF from {} non-WM voxels)",
            n_crude_gm, n_crude_csf, n_crude_nonwm
        )));
    }

    // ---------------- Refined segmentation ----------------
    // WM: drop the high-SDM tail (2 robust sigma above the median). Those are
    // typically CSF-contaminated voxels that the FA test let through.
    let wm_median = masked_median(&safe_sdm, &crude_wm);
    let wm_mad = masked_median_of(&crude_wm, |idx| (safe_sdm[idx] - wm_median).abs());
    let wm_outlier_thresh = wm_median + 1.4826 * wm_mad * 2.0;
    let wm_outliers = select_where(&crude_wm, |idx| safe_sdm[idx] > wm_outlier_thresh);
    let refined_wm = difference(&crude_wm, &wm_outliers);
    let n_refined_wm = count(&refined_wm);
    if n_refined_wm == 0 {
        return Err(CsDmriError::Other(
            "dhollander selection: refined WM is empty after outlier rejection".into(),
        ));
    }

    // GM: split at the median, then keep the half of each side that sits
    // closest to the median — the tails are where partial volume lives.
    let gm_median = masked_median(&safe_sdm, &crude_gm);
    let gm_high = select_where(&crude_gm, |idx| safe_sdm[idx] > gm_median);
    let gm_low = difference(&crude_gm, &gm_high);
    let gm_high_keep = if count(&gm_high) > 0 {
        threshold_optimal(&safe_sdm, &gm_high, gm_median, true)?
    } else {
        gm_high.clone()
    };
    let gm_low_keep = if count(&gm_low) > 0 {
        // Negated so "far from the median" is again the high side.
        let neg = map_masked(&gm_low, |idx| -(safe_sdm[idx] - gm_median));
        threshold_optimal(&neg, &gm_low, 0.0, true)?
    } else {
        gm_low.clone()
    };
    let refined_gm = union(&gm_high_keep, &gm_low_keep);
    let n_refined_gm = count(&refined_gm);
    if n_refined_gm == 0 {
        return Err(CsDmriError::Other(
            "dhollander selection: refined GM is empty".into(),
        ));
    }

    // CSF: first recover the crude-WM outliers that decay at least as fast as
    // the slowest crude CSF voxel (mis-classified free water), then keep only
    // the fast-decaying part of that enlarged population.
    let csf_min = masked_min(&safe_sdm, &crude_csf);
    let csf_extra = union(
        &crude_csf,
        &select_where(&wm_outliers, |idx| safe_sdm[idx] > csf_min),
    );
    let refined_csf = threshold_optimal(&safe_sdm, &csf_extra, csf_min, false)?;
    let n_refined_csf = count(&refined_csf);
    if n_refined_csf == 0 {
        return Err(CsDmriError::Other(
            "dhollander selection: refined CSF is empty".into(),
        ));
    }

    // ---------------- Final voxel selection ----------------
    // CSF: the fastest-decaying csf_pct of refined CSF.
    let n_csf_target = pct_count(n_refined_csf, cfg.csf_pct, "CSF")?;
    let csf = select_top(&safe_sdm, &refined_csf, n_csf_target)?;

    // GM: the gm_pct of refined GM whose SDM sits closest to the refined-GM
    // median. The +1 offset keeps every in-mask metric value strictly positive
    // so the zero-valued background stays excluded from the ranking.
    let refined_gm_median = masked_median(&safe_sdm, &refined_gm);
    let gm_metric = map_masked(&refined_gm, |idx| {
        (safe_sdm[idx] - refined_gm_median).abs() + 1.0
    });
    let n_gm_target = pct_count(n_refined_gm, cfg.gm_pct, "GM")?;
    let gm = select_bottom(&gm_metric, &refined_gm, n_gm_target)?;

    // Single-fibre WM.
    let n_sfwm_target = pct_count(n_refined_wm, cfg.sfwm_pct, "single-fibre WM")?;
    let sfwm = select_sfwm_by_fa(dti, &refined_wm, n_sfwm_target)?;

    let counts = DhollanderStageCounts {
        mask: n_mask,
        eroded: n_eroded,
        safe: n_safe,
        crude_wm: n_crude_wm,
        crude_gm: n_crude_gm,
        crude_csf: n_crude_csf,
        refined_wm: n_refined_wm,
        refined_gm: n_refined_gm,
        refined_csf: n_refined_csf,
        sfwm: count(&sfwm),
        gm: count(&gm),
        csf: count(&csf),
    };

    Ok(DhollanderSelection {
        sfwm,
        gm,
        csf,
        refined_wm,
        safe_mask,
        safe_sdm,
        counts,
    })
}

/// Single-fibre WM = the `n` highest-FA voxels of refined WM.
///
/// This is MRtrix's `-wm_algo fa` branch, not its default. The default is the
/// Dhollander 2019 metric, which scores each refined-WM voxel by the ratio of
/// its fODF peak amplitude to total (WM + CSF) l=0 amplitude under a two-tissue
/// CSD fit — a genuine single-fibre test, where FA merely correlates with one.
/// Both are supported upstream and both feed the same `amp2response` step; FA
/// ranking is what cs_dmri can do without standing up a second CSD solve inside
/// response estimation.
pub fn select_sfwm_by_fa(
    dti: &DtiVolumeResult,
    refined_wm: &Array3<bool>,
    n: usize,
) -> Result<Array3<bool>> {
    let fa = map_masked(refined_wm, |idx| dti.fa[idx] as f64);
    select_top(&fa, refined_wm, n)
}

// ===================== signal decay metric =====================

/// b-value clusters used as SDM "shells".
///
/// MRtrix reads shells straight out of the gradient table, which presumes
/// shelled acquisition. cs_dmri also has to cope with continuous-b CS-DSI
/// schemes, so shells here are b-value clusters at
/// `tolerance`, with clusters below `min_volumes` merged into their nearest
/// neighbour. On genuinely shelled data (the QST validation set: 18/90/90/90 at
/// b = 0/1000/2000/3000) this reproduces the acquisition shells exactly.
///
/// Returns `(mean_b, volume_indices)` sorted by ascending b, b=0 first.
fn sdm_shells(dwi: &DwiData, cfg: &DhollanderSelectConfig) -> Vec<(f64, Vec<usize>)> {
    let shells = dwi.gtab.shells(cfg.shell_tolerance);
    let mut clusters: Vec<(f64, Vec<usize>)> =
        shells.into_iter().map(|s| (s.b, s.indices)).collect();

    // Merge undersized clusters into the nearest surviving neighbour by mean b.
    // The b=0 cluster (index 0) is never merged away: it is the SDM reference.
    loop {
        if clusters.len() <= 2 {
            break;
        }
        let smallest = clusters
            .iter()
            .enumerate()
            .skip(1)
            .min_by_key(|(_, (_, idx))| idx.len())
            .map(|(i, _)| i);
        let Some(i) = smallest else { break };
        if clusters[i].1.len() >= cfg.min_shell_volumes {
            break;
        }
        // Nearest other cluster, but never the b=0 cluster.
        let target = (1..clusters.len())
            .filter(|&j| j != i)
            .min_by(|&a, &b| {
                let da = (clusters[a].0 - clusters[i].0).abs();
                let db = (clusters[b].0 - clusters[i].0).abs();
                da.total_cmp(&db)
            });
        let Some(t) = target else { break };
        let (b_i, idx_i) = clusters.remove(i);
        let t = if t > i { t - 1 } else { t };
        let n_i = idx_i.len() as f64;
        let n_t = clusters[t].1.len() as f64;
        clusters[t].0 = (clusters[t].0 * n_t + b_i * n_i) / (n_t + n_i);
        clusters[t].1.extend(idx_i);
    }
    clusters.sort_by(|a, b| a.0.total_cmp(&b.0));
    clusters
}

/// Per-voxel signal decay metric, and the mask of voxels where it (or any shell
/// mean it is built from) came out invalid.
///
/// `SDM = Σ_{shells b>0} n_b · log(S̄₀ / S̄_b) / Σ_{shells b>0} n_b`, where each
/// `S̄` is the mean over that shell's volumes of the signal clamped at zero.
fn signal_decay_metric(
    dwi: &DwiData,
    within: &Array3<bool>,
    cfg: &DhollanderSelectConfig,
) -> Result<(Array3<f64>, Array3<bool>)> {
    let shells = sdm_shells(dwi, cfg);
    if shells.len() < 2 {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: need at least 2 distinct b-value shells (including b=0), found {}",
            shells.len()
        )));
    }
    if shells[0].0 > dwi.gtab.b0_threshold {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: lowest shell is b={:.0}, above the b=0 threshold {:.0} — \
             the signal decay metric needs a b=0 reference",
            shells[0].0, dwi.gtab.b0_threshold
        )));
    }

    let s = dwi.data.shape();
    let dims = (s[0], s[1], s[2]);
    let mut sdm = Array3::<f64>::zeros(dims);
    let mut erroneous = Array3::<bool>::default(dims);
    let total_dw_volumes: f64 = shells[1..].iter().map(|(_, i)| i.len() as f64).sum();

    for x in 0..dims.0 {
        for y in 0..dims.1 {
            for z in 0..dims.2 {
                if !within[(x, y, z)] {
                    continue;
                }
                let view = dwi.data.slice(ndarray::s![x, y, z, ..]);
                let shell_mean = |indices: &[usize]| -> f64 {
                    let sum: f64 = indices.iter().map(|&i| (view[i] as f64).max(0.0)).sum();
                    sum / indices.len() as f64
                };
                let s0 = shell_mean(&shells[0].1);
                if !s0.is_finite() || s0 <= 0.0 {
                    erroneous[(x, y, z)] = true;
                    continue;
                }
                let mut acc = 0.0;
                let mut bad = false;
                for (_, indices) in &shells[1..] {
                    let sb = shell_mean(indices);
                    if !sb.is_finite() || sb <= 0.0 {
                        bad = true;
                        break;
                    }
                    acc += (s0 / sb).ln() * indices.len() as f64;
                }
                let value = acc / total_dw_volumes;
                if bad || !value.is_finite() || value <= 0.0 {
                    erroneous[(x, y, z)] = true;
                } else {
                    sdm[(x, y, z)] = value;
                }
            }
        }
    }
    Ok((sdm, erroneous))
}

// ===================== mrthreshold / statistics primitives =====================

/// MRtrix's parameter-free optimal threshold (Ridgway et al., NeuroImage 2009,
/// 44(1):99-111), as `Filter::estimate_optimal_threshold`: pick the threshold
/// maximising the correlation between the image and the binary mask it induces,
/// searched by golden section over the value range.
///
/// `values` must already be restricted to the mask and finite. Returns `None`
/// if there is nothing to search over (fewer than two distinct values).
fn estimate_optimal_threshold(values: &[f64]) -> Option<f64> {
    let count = values.len();
    if count == 0 {
        return None;
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut sum_sqr = 0.0;
    for &v in values {
        min = min.min(v);
        max = max.max(v);
        sum += v;
        sum_sqr += v * v;
    }
    if max <= min {
        return None;
    }
    let n = count as f64;
    let mean = sum / n;
    let stdev = ((sum_sqr - sum * mean) / n).sqrt();
    if !stdev.is_finite() || stdev <= 0.0 {
        return None;
    }

    // Negative correlation between the image and the mask `value > threshold`;
    // golden section minimises, so the returned threshold maximises correlation.
    let cost = |threshold: f64| -> f64 {
        let mut hits = 0.0;
        let mut sum_xy = 0.0;
        for &v in values {
            if v > threshold {
                hits += 1.0;
                sum_xy += v;
            }
        }
        let mean_xy = sum_xy / n;
        let covariance = mean_xy - (hits / n) * mean;
        let mask_stdev = ((hits - hits * hits / n) / n).sqrt();
        -covariance / (stdev * mask_stdev)
    };

    Some(golden_section_search(
        cost,
        min + 0.001 * (max - min),
        0.5 * (min + max),
        max - 0.001 * (max - min),
        0.01,
    ))
}

/// `Math::golden_section_search`. Minimises `f` on `[min_bound, max_bound]`.
fn golden_section_search<F: Fn(f64) -> f64>(
    f: F,
    min_bound: f64,
    init: f64,
    max_bound: f64,
    tolerance: f64,
) -> f64 {
    const G1: f64 = 0.618_033_99;
    const G2: f64 = 1.0 - G1;
    let (mut x0, mut x3) = (min_bound, max_bound);
    let (mut x1, mut x2);
    if (max_bound - init).abs() > (init - min_bound).abs() {
        x1 = init;
        x2 = init + G2 * (max_bound - init);
    } else {
        x2 = init;
        x1 = init - G2 * (init - min_bound);
    }
    let (mut f1, mut f2) = (f(x1), f(x2));
    // Guard against a pathological cost surface keeping the bracket alive.
    let mut iterations = 0;
    while tolerance * (x1.abs() + x2.abs()) < (x3 - x0).abs() && iterations < 200 {
        if f2 < f1 {
            x0 = x1;
            x1 = x2;
            x2 = G1 * x1 + G2 * x3;
            f1 = f2;
            f2 = f(x2);
        } else {
            x3 = x2;
            x2 = x1;
            x1 = G1 * x2 + G2 * x0;
            f2 = f1;
            f1 = f(x1);
        }
        iterations += 1;
    }
    if f1 < f2 { x1 } else { x2 }
}

/// `mrthreshold` with no explicit mechanism: determine the optimal threshold
/// from `(metric − offset)` over `mask`, then select the voxels of `mask` at or
/// above it (below it, when `invert`).
fn threshold_optimal(
    metric: &Array3<f64>,
    mask: &Array3<bool>,
    offset: f64,
    invert: bool,
) -> Result<Array3<bool>> {
    let values: Vec<f64> = masked_values(metric, mask)
        .into_iter()
        .map(|v| v - offset)
        .collect();
    let Some(threshold) = estimate_optimal_threshold(&values) else {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: could not determine an optimal threshold — the signal decay \
             metric is constant over the {} candidate voxels",
            values.len()
        )));
    };
    Ok(select_where(mask, |idx| {
        let v = metric[idx] - offset;
        if invert { v < threshold } else { v >= threshold }
    }))
}

/// `mrthreshold -top n -ignorezero`: select the `n` highest-valued voxels of
/// `mask`. Ties at the cut are all kept, matching MRtrix (which thresholds at
/// the n-th value rather than truncating the sorted list).
fn select_top(metric: &Array3<f64>, mask: &Array3<bool>, n: usize) -> Result<Array3<bool>> {
    let mut values = nonzero_masked_values(metric, mask);
    if n == 0 || n > values.len() {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: asked for the top {} of {} candidate voxels",
            n,
            values.len()
        )));
    }
    let index = values.len() - n;
    values.select_nth_unstable_by(index, f64::total_cmp);
    let threshold = values[index];
    Ok(select_where(mask, |idx| metric[idx] >= threshold))
}

/// `mrthreshold -bottom n -ignorezero`: select the `n` lowest-valued voxels.
fn select_bottom(metric: &Array3<f64>, mask: &Array3<bool>, n: usize) -> Result<Array3<bool>> {
    let mut values = nonzero_masked_values(metric, mask);
    if n == 0 || n > values.len() {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: asked for the bottom {} of {} candidate voxels",
            n,
            values.len()
        )));
    }
    let index = n - 1;
    values.select_nth_unstable_by(index, f64::total_cmp);
    let threshold = values[index];
    Ok(select_where(mask, |idx| metric[idx] <= threshold))
}

/// `int(round(total * pct / 100))`, with Python's round-half-to-even.
fn pct_count(total: usize, pct: f64, label: &str) -> Result<usize> {
    let n = (total as f64 * pct / 100.0).round_ties_even() as usize;
    if n == 0 {
        return Err(CsDmriError::Other(format!(
            "dhollander selection: {}% of {} refined {} voxels rounds to zero — raise the percentage",
            pct, total, label
        )));
    }
    Ok(n)
}

// ===================== small mask/array helpers =====================

fn count(mask: &Array3<bool>) -> usize {
    mask.iter().filter(|v| **v).count()
}

fn erode_mask(mask: &Array3<bool>, npass: usize) -> Array3<bool> {
    let dims = mask.dim();
    let mut current = mask.to_owned();
    for _ in 0..npass {
        let mut next = Array3::<bool>::default(dims);
        for x in 1..dims.0.saturating_sub(1) {
            for y in 1..dims.1.saturating_sub(1) {
                for z in 1..dims.2.saturating_sub(1) {
                    next[(x, y, z)] = current[(x, y, z)]
                        && current[(x - 1, y, z)]
                        && current[(x + 1, y, z)]
                        && current[(x, y - 1, z)]
                        && current[(x, y + 1, z)]
                        && current[(x, y, z - 1)]
                        && current[(x, y, z + 1)];
                }
            }
        }
        current = next;
    }
    current
}

fn select_where<F: Fn((usize, usize, usize)) -> bool>(
    mask: &Array3<bool>,
    pred: F,
) -> Array3<bool> {
    let dims = mask.dim();
    let mut out = Array3::<bool>::default(dims);
    for x in 0..dims.0 {
        for y in 0..dims.1 {
            for z in 0..dims.2 {
                if mask[(x, y, z)] && pred((x, y, z)) {
                    out[(x, y, z)] = true;
                }
            }
        }
    }
    out
}

fn map_masked<F: Fn((usize, usize, usize)) -> f64>(mask: &Array3<bool>, f: F) -> Array3<f64> {
    let dims = mask.dim();
    let mut out = Array3::<f64>::zeros(dims);
    for x in 0..dims.0 {
        for y in 0..dims.1 {
            for z in 0..dims.2 {
                if mask[(x, y, z)] {
                    out[(x, y, z)] = f((x, y, z));
                }
            }
        }
    }
    out
}

fn union(a: &Array3<bool>, b: &Array3<bool>) -> Array3<bool> {
    let mut out = a.to_owned();
    for (o, v) in out.iter_mut().zip(b.iter()) {
        *o = *o || *v;
    }
    out
}

fn difference(a: &Array3<bool>, b: &Array3<bool>) -> Array3<bool> {
    let mut out = a.to_owned();
    for (o, v) in out.iter_mut().zip(b.iter()) {
        *o = *o && !*v;
    }
    out
}

fn masked_values(metric: &Array3<f64>, mask: &Array3<bool>) -> Vec<f64> {
    metric
        .iter()
        .zip(mask.iter())
        .filter(|(v, m)| **m && v.is_finite())
        .map(|(v, _)| *v)
        .collect()
}

/// `mrthreshold`'s `-ignorezero` data collection: masked, finite, non-zero.
fn nonzero_masked_values(metric: &Array3<f64>, mask: &Array3<bool>) -> Vec<f64> {
    metric
        .iter()
        .zip(mask.iter())
        .filter(|(v, m)| **m && v.is_finite() && **v != 0.0)
        .map(|(v, _)| *v)
        .collect()
}

/// `Math::median`: middle element, averaging the two middle ones for even
/// counts. NaN for an empty population (callers guard against that).
fn median_of_values(mut values: Vec<f64>) -> f64 {
    let num = values.len();
    if num == 0 {
        return f64::NAN;
    }
    let middle = num / 2;
    values.select_nth_unstable_by(middle, f64::total_cmp);
    let hi = values[middle];
    if num % 2 == 1 {
        return hi;
    }
    let lo = values[..middle]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    (hi + lo) / 2.0
}

fn masked_median(metric: &Array3<f64>, mask: &Array3<bool>) -> f64 {
    median_of_values(masked_values(metric, mask))
}

fn masked_median_of<F: Fn((usize, usize, usize)) -> f64>(mask: &Array3<bool>, f: F) -> f64 {
    let dims = mask.dim();
    let mut values = Vec::new();
    for x in 0..dims.0 {
        for y in 0..dims.1 {
            for z in 0..dims.2 {
                if mask[(x, y, z)] {
                    let v = f((x, y, z));
                    if v.is_finite() {
                        values.push(v);
                    }
                }
            }
        }
    }
    median_of_values(values)
}

fn masked_min(metric: &Array3<f64>, mask: &Array3<bool>) -> f64 {
    masked_values(metric, mask)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qspace::{BvecFrame, GradientTable};
    use ndarray::Array4;

    /// A 20³ phantom in three axial slabs — CSF (fast, isotropic), GM (slow,
    /// isotropic), WM (anisotropic) — with a slight in-slab diffusivity and FA
    /// gradient so the top-N selections have something to rank.
    fn slab_phantom() -> (DwiData, DtiVolumeResult, usize, usize) {
        let n = 20usize;
        let csf_end = 7usize; // z < 7
        let gm_end = 13usize; // 7 <= z < 13; WM above

        let dirs: Vec<[f64; 3]> = (0..20)
            .map(|i| {
                // Cheap quasi-uniform spiral, good enough for a powder average.
                let t = (i as f64 + 0.5) / 20.0;
                let z = 1.0 - 2.0 * t;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = 2.399_963_2 * i as f64;
                [r * phi.cos(), r * phi.sin(), z]
            })
            .collect();
        let mut bvals = vec![0.0; 6];
        let mut bvecs = vec![[0.0; 3]; 6];
        for &b in &[1000.0, 2000.0, 3000.0] {
            for d in &dirs {
                bvals.push(b);
                bvecs.push(*d);
            }
        }
        let gtab = GradientTable::new(bvals.clone(), bvecs.clone(), Some(0.043), Some(0.011), None)
            .unwrap();
        let ng = gtab.n_grads();

        let mut data = Array4::<f32>::zeros((n, n, n, ng));
        let mask = Array3::<bool>::from_elem((n, n, n), true);
        let mut fa = Array3::<f32>::zeros((n, n, n));
        let md = Array3::<f32>::zeros((n, n, n));
        let tensor = Array4::<f32>::zeros((n, n, n, 6));
        let mut pdir = Array4::<f32>::zeros((n, n, n, 3));
        let s0 = 1000.0_f64;

        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    // Breaks ties so `-top N` selects exactly N voxels. Every
                    // (x, y, z) gets a distinct value; MRtrix keeps all voxels
                    // tied at the cut, so equal values would overshoot N.
                    let jitter = 1.0 + 0.002 * (x + n * y) as f64 + 0.0001 * z as f64;
                    let (adc_par, adc_perp, this_fa) = if z < csf_end {
                        (3.0e-3 * jitter, 3.0e-3 * jitter, 0.02)
                    } else if z < gm_end {
                        (0.9e-3 * jitter, 0.9e-3 * jitter, 0.08)
                    } else {
                        (1.7e-3 * jitter, 0.2e-3 * jitter, 0.75 * jitter)
                    };
                    fa[(x, y, z)] = this_fa as f32;
                    pdir[(x, y, z, 2)] = 1.0;
                    for i in 0..ng {
                        let g = bvecs[i];
                        let adc = adc_perp + (adc_par - adc_perp) * g[2] * g[2];
                        data[(x, y, z, i)] = (s0 * (-bvals[i] * adc).exp()) as f32;
                    }
                }
            }
        }

        let dwi = DwiData::from_table(
            data,
            mask,
            gtab,
            BvecFrame::WorldRas,
            std::path::PathBuf::from("synthetic.nii.gz"),
        );
        let dti = DtiVolumeResult {
            s0: Array3::from_elem((n, n, n), s0 as f32),
            fa,
            md,
            outlier_fraction: Array3::zeros((n, n, n)),
            tensor,
            principal_dir: pdir,
            iterations: None,
            converged: None,
        };
        (dwi, dti, csf_end, gm_end)
    }

    #[test]
    fn staged_selection_assigns_every_final_voxel_to_the_right_slab() {
        let (dwi, dti, csf_end, gm_end) = slab_phantom();
        let sel = select_voxels(&dwi, &dti, &DhollanderSelectConfig::default())
            .expect("staged selection on a clean phantom");

        let slab_of = |mask: &Array3<bool>| -> Vec<usize> {
            let mut zs = Vec::new();
            for ((_, _, z), v) in mask.indexed_iter() {
                if *v {
                    zs.push(z);
                }
            }
            zs
        };
        assert!(
            slab_of(&sel.csf).iter().all(|&z| z < csf_end),
            "final CSF voxels escaped the CSF slab: {:?}",
            sel.counts
        );
        assert!(
            slab_of(&sel.gm).iter().all(|&z| (csf_end..gm_end).contains(&z)),
            "final GM voxels escaped the GM slab: {:?}",
            sel.counts
        );
        assert!(
            slab_of(&sel.sfwm).iter().all(|&z| z >= gm_end),
            "final single-fibre WM voxels escaped the WM slab: {:?}",
            sel.counts
        );

        // Erosion peels 3 layers off a solid 20³ cube.
        assert_eq!(sel.counts.eroded, 14 * 14 * 14);
        assert_eq!(sel.counts.safe, sel.counts.eroded);
        // The percentages are honoured (ties broken by the jitter).
        assert_eq!(sel.counts.csf, (sel.counts.refined_csf as f64 * 0.10).round() as usize);
        assert_eq!(sel.counts.gm, (sel.counts.refined_gm as f64 * 0.02).round() as usize);
    }

    #[test]
    fn signal_decay_metric_separates_free_water_from_tissue() {
        let (dwi, dti, csf_end, gm_end) = slab_phantom();
        let sel = select_voxels(&dwi, &dti, &DhollanderSelectConfig::default()).unwrap();
        // Middle of each slab, away from the eroded rim.
        let sdm_at = |z: usize| sel.safe_sdm[(10, 10, z)];
        assert!(sel.safe_mask[(10, 10, 3)]);
        // The crude GM/CSF split is the one the SDM has to carry: free water
        // decays far faster than either tissue. (WM sits below GM here because
        // the phantom's WM is very prolate, so its powder average decays
        // slowly — irrelevant to the algorithm, which removes WM on FA first.)
        assert!(
            sdm_at(3) > 3.0 * sdm_at(csf_end + 2),
            "expected SDM(CSF) ≫ SDM(GM), got {} vs {}",
            sdm_at(3),
            sdm_at(csf_end + 2)
        );
        assert!(sdm_at(gm_end + 3) < sdm_at(csf_end + 2));
    }

    #[test]
    fn a_dwi_with_no_b0_is_rejected_with_an_actionable_message() {
        let (mut dwi, dti, _, _) = slab_phantom();
        // Re-label the b=0 volumes as b=1000 so no b=0 reference survives.
        let mut bvals = dwi.gtab.bvals.clone();
        for b in bvals.iter_mut().take(6) {
            *b = 1000.0;
        }
        let bvecs: Vec<[f64; 3]> = (0..bvals.len())
            .map(|i| {
                if i < 6 {
                    [0.0, 0.0, 1.0]
                } else {
                    dwi.gtab.bvecs[i]
                }
            })
            .collect();
        dwi.gtab = GradientTable::new(bvals, bvecs, Some(0.043), Some(0.011), None).unwrap();
        let err = select_voxels(&dwi, &dti, &DhollanderSelectConfig::default()).unwrap_err();
        assert!(
            format!("{err}").contains("b=0"),
            "unhelpful error: {err}"
        );
    }

    #[test]
    fn shells_merge_undersized_clusters_but_keep_real_ones() {
        let (dwi, _, _, _) = slab_phantom();
        let cfg = DhollanderSelectConfig::default();
        let shells = sdm_shells(&dwi, &cfg);
        assert_eq!(shells.len(), 4, "b=0 plus three acquisition shells");
        assert_eq!(shells[0].1.len(), 6);
        for (s, expected_b) in shells[1..].iter().zip([1000.0, 2000.0, 3000.0]) {
            assert_eq!(s.1.len(), 20);
            assert!((s.0 - expected_b).abs() < 1.0);
        }
    }

    #[test]
    fn median_matches_mrtrix_convention() {
        assert_eq!(median_of_values(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_of_values(vec![4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_of_values(vec![7.0]), 7.0);
    }

    #[test]
    fn erosion_peels_one_layer_per_pass_and_clears_the_boundary() {
        let mut mask = Array3::<bool>::default((7, 7, 7));
        mask.fill(true);
        let once = erode_mask(&mask, 1);
        // Boundary voxels always go, plus their inward neighbours are kept.
        assert!(!once[(0, 3, 3)]);
        assert!(once[(1, 1, 1)]);
        assert_eq!(count(&once), 5 * 5 * 5);
        let twice = erode_mask(&mask, 2);
        assert_eq!(count(&twice), 3 * 3 * 3);
        assert_eq!(count(&erode_mask(&mask, 0)), 7 * 7 * 7);
    }

    #[test]
    fn optimal_threshold_separates_a_clean_bimodal_population() {
        // 100 voxels at ~1 and 100 at ~10: the threshold must land between.
        let mut values = Vec::new();
        for i in 0..100 {
            values.push(1.0 + (i as f64) * 0.001);
            values.push(10.0 + (i as f64) * 0.001);
        }
        let t = estimate_optimal_threshold(&values).expect("bimodal data has a threshold");
        assert!(t > 1.1 && t < 10.0, "threshold {t} not between the modes");
    }

    /// Pinned against the real thing: `mrthreshold` from MRtrix 3
    /// (pennlinc/qsiprep:1.0.2) run on the NIfTI these exact values produce
    /// prints `1.5003`. The generator is a plain LCG so the fixture needs no
    /// data file — regenerate it with the same recurrence to re-derive.
    ///
    /// This is the load-bearing check on the port: the Ridgway cost function
    /// and the golden-section search are where a subtle transcription error
    /// would silently shift every crude and refined boundary.
    #[test]
    fn optimal_threshold_matches_mrtrix_mrthreshold() {
        fn fixture(seed: u32, kind: &str) -> Vec<f64> {
            let mut x = seed;
            let mut next = || {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                x as f64 / 4_294_967_296.0
            };
            (0..8000)
                .map(|_| {
                    let (a, b) = (next(), next());
                    let v = match kind {
                        // Two well-separated modes, 60/40.
                        "bimodal" => {
                            if a < 0.6 {
                                1.0 + 0.5 * b
                            } else {
                                4.0 + 2.0 * b
                            }
                        }
                        // Long right tail, the shape an SDM map actually has.
                        "skew" => 0.5 + 3.0 * a.powi(3) + 0.2 * b,
                        // Broad unimodal — no real boundary to find.
                        _ => 2.0 + (a + b - 1.0),
                    };
                    v as f32 as f64
                })
                .collect()
        }
        // Reference values printed by `mrthreshold` on the NIfTIs these
        // generators produce — identical from a dev build
        // (3.0.8-2097-g99963980) and from 3.0.4.
        for (seed, kind, expected) in [
            (12345u32, "bimodal", 1.5003),
            (999, "skew", 1.72148),
            (777, "uni", 1.99963),
        ] {
            let t = estimate_optimal_threshold(&fixture(seed, kind))
                .unwrap_or_else(|| panic!("{kind} fixture should have a threshold"));
            assert!(
                (t - expected).abs() / expected < 1e-3,
                "{kind}: optimal threshold {t} differs from MRtrix's {expected} by more than 0.1%"
            );
        }
    }

    #[test]
    fn optimal_threshold_declines_constant_input() {
        assert!(estimate_optimal_threshold(&[2.0; 50]).is_none());
        assert!(estimate_optimal_threshold(&[]).is_none());
    }

    #[test]
    fn top_and_bottom_selection_count_correctly() {
        let mut mask = Array3::<bool>::default((4, 1, 1));
        mask.fill(true);
        let mut metric = Array3::<f64>::zeros((4, 1, 1));
        for (i, v) in [1.0, 4.0, 2.0, 3.0].iter().enumerate() {
            metric[(i, 0, 0)] = *v;
        }
        let top2 = select_top(&metric, &mask, 2).unwrap();
        assert_eq!(count(&top2), 2);
        assert!(top2[(1, 0, 0)] && top2[(3, 0, 0)]);
        let bottom2 = select_bottom(&metric, &mask, 2).unwrap();
        assert_eq!(count(&bottom2), 2);
        assert!(bottom2[(0, 0, 0)] && bottom2[(2, 0, 0)]);
    }

    #[test]
    fn pct_count_rounds_like_python() {
        assert_eq!(pct_count(1000, 10.0, "t").unwrap(), 100);
        assert_eq!(pct_count(1000, 0.5, "t").unwrap(), 5);
        assert_eq!(pct_count(100, 2.5, "t").unwrap(), 2); // 2.5 → 2 (ties to even)
        assert!(pct_count(10, 0.5, "t").is_err()); // rounds to 0
    }
}
