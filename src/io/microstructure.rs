// SPDX-License-Identifier: MIT OR Apache-2.0
//! SHORE/MAPMRI microstructure scalars (RTOP, RTAP, RTPP, MSD, QIV, NG):
//! per-voxel compute, fit-failure rejection, display-range hints, ODX DPV
//! embedding, and optional sibling NIfTI export. Shared by `cs-odf` and
//! `cs-fit --odx-output`.
//!
//! Outlier rejection: when a voxel's scalar is many times above the rest of
//! the brain (RTOP/RTAP can be 5000× the median in degenerate fits), the
//! number isn't real signal — it's a fit failure where E(0) collapsed or
//! the SHORE basis produced an unphysical estimate. By default we set those
//! voxels to NaN per-scalar; NaN propagates cleanly through viewers (out of
//! colormap range) and downstream tools (treated as missing data). Threshold
//! is `K × p99` of the brain-wide finite values, default K=10. The chosen K
//! and per-scalar reject counts are stamped on `cs_dmri_microstructure_outliers`.
//!
//! Display hints: after rejection, per-scalar median / p95 / p99 / max +
//! a suggested `[0, p99]` colormap range are written to
//! `cs_dmri_microstructure_display` so a viewer can auto-range against the
//! cleaned distribution.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Result, anyhow};
use ndarray::Array3;
use rayon::prelude::*;

use odx_rs::dtype::DType;
use odx_rs::stream::OdxBuilder;

use crate::ShoreBasis;
use crate::scalars::microstructure::{self, ScalarBasisInfo};

/// Length-unit convention for the emitted scalars. `Um` matches TORTOISE's
/// `EstimateMAPMRI` (q in 1/μm, RTOP in /μm³, …); `Mm` keeps the dipy /
/// cs_dmri internal convention (q in 1/mm).
#[derive(Debug, Clone, Copy)]
pub enum MicrostructureUnits {
    Um,
    Mm,
}

/// Fit-failure rejection config. Voxels whose scalar exceeds
/// `p99_factor × p99` get set to NaN per-scalar so the value can't be
/// mistaken for a measurement.
#[derive(Debug, Clone, Copy)]
pub struct MicrostructureOutlierRejection {
    /// Multiplier on the brain-wide 99th percentile. Default 10. Bigger
    /// values reject only the most pathological outliers; smaller values
    /// trim more aggressively into the upper tail. With p99/median ≈ 8 for
    /// these scalars, K=10 corresponds roughly to "anything more than 80×
    /// the median" — fit failures are typically 100×-thousands× off; real
    /// dense-WM voxels sit below p99.
    pub p99_factor: f32,
}

impl Default for MicrostructureOutlierRejection {
    fn default() -> Self {
        Self { p99_factor: 10.0 }
    }
}

#[derive(Debug, Clone)]
pub struct MicrostructureOptions {
    pub units: MicrostructureUnits,
    /// `Some` (default) NaN's voxels with manifestly broken fits per scalar.
    /// `None` keeps every finite value as-is — useful for parity sweeps or
    /// when you want to inspect the failure modes directly.
    pub outlier_rejection: Option<MicrostructureOutlierRejection>,
    pub quiet: bool,
}

impl Default for MicrostructureOptions {
    fn default() -> Self {
        Self {
            units: MicrostructureUnits::Um,
            outlier_rejection: Some(MicrostructureOutlierRejection::default()),
            quiet: false,
        }
    }
}

/// Per-voxel scalars in masked-voxel order, post-unit-scaling and
/// post-outlier-rejection. NaN voxels mean either (a) a missing peak
/// direction (RTAP/RTPP), (b) a SHORE fit whose E(0) was non-finite
/// (RTOP/QIV/etc.), or (c) a value the rejection step flagged as a fit
/// failure. Display stats are computed *after* rejection so the suggested
/// colormap range tracks the cleaned distribution.
pub struct MicrostructureScalars {
    pub rtop: Vec<f32>,
    pub rtap: Vec<f32>,
    pub rtpp: Vec<f32>,
    pub msd: Vec<f32>,
    pub qiv: Vec<f32>,
    pub ng: Vec<f32>,
    pub display_stats: MicrostructureDisplayStats,
    /// `Some` when outlier rejection was enabled. Per-scalar threshold +
    /// reject count for traceability.
    pub outlier_report: Option<MicrostructureOutlierReport>,
}

/// Per-scalar robust statistics over finite, in-mask values. Embedded as
/// `cs_dmri_microstructure_display` so a viewer can pick a sane colormap
/// range without auto-scaling against a handful of CSF/edge outliers.
#[derive(Debug, Clone)]
pub struct MicrostructureDisplayStats {
    pub n_voxels: usize,
    pub stats: BTreeMap<String, ScalarDisplayStat>,
}

#[derive(Debug, Clone)]
pub struct ScalarDisplayStat {
    pub finite_count: usize,
    pub min: Option<f32>,
    pub median: Option<f32>,
    pub p95: Option<f32>,
    pub p99: Option<f32>,
    pub max: Option<f32>,
    /// Recommended `[0, suggested_max]` colormap range. `suggested_max =
    /// p99` for these non-negative diffusion scalars (NG is bounded [0,1]
    /// physically; for it `suggested_max = max(values).min(1)`).
    pub suggested_max: Option<f32>,
}

/// Per-scalar fit-failure-rejection stats. Serializable for the ODX
/// header extra `cs_dmri_microstructure_outliers`.
#[derive(Debug, Clone)]
pub struct MicrostructureOutlierReport {
    pub p99_factor: f32,
    pub n_voxels: usize,
    pub stats: BTreeMap<String, ScalarOutlierStat>,
}

#[derive(Debug, Clone)]
pub struct ScalarOutlierStat {
    /// Voxels with a finite value before rejection.
    pub finite_count_before: usize,
    /// Voxels NaN'd by rejection.
    pub rejected_count: usize,
    /// Threshold applied (`p99_factor × p99`). `None` if no finite values
    /// existed for this scalar.
    pub threshold: Option<f32>,
    /// `p99` over the pre-rejection distribution. `None` if no finite values.
    pub p99: Option<f32>,
}

/// Compute all six microstructure scalars per masked voxel, apply unit
/// scaling, then winsorize per-scalar at `opts.clip.percentile` if set.
///
/// `coeffs` and `masked_indices` must be in the same canonical frame produced
/// by `build_shore_odx`. `peak0_dirs` (length = masked voxels) provides the
/// first peak direction per voxel for RTAP/RTPP; pass `None` (or a slice with
/// `None` entries) when peaks weren't extracted — those scalars come back as
/// NaN per voxel.
pub fn compute_microstructure(
    basis: &ShoreBasis,
    coeffs: &ndarray::Array4<f32>,
    masked_indices: &[(usize, usize, usize)],
    peak0_dirs: Option<&[Option<[f32; 3]>]>,
    opts: &MicrostructureOptions,
) -> MicrostructureScalars {
    let (s_rtop, s_rtap, s_rtpp, s_msd, s_qiv, s_ng) = match opts.units {
        MicrostructureUnits::Mm => (1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32),
        MicrostructureUnits::Um => (
            1e-9_f32, // mm⁻³ → μm⁻³
            1e-6_f32, // mm⁻² → μm⁻²
            1e-3_f32, // mm⁻¹ → μm⁻¹
            1e6_f32,  // mm² → μm²
            1e15_f32, // mm⁵ → μm⁵
            1.0_f32,  // dimensionless
        ),
    };

    let info = ScalarBasisInfo::from_basis(basis);
    let n_shore = info.n_coeffs;
    let n_voxels = masked_indices.len();

    #[derive(Default, Clone, Copy)]
    struct VoxelScalars {
        rtop: f32,
        rtap: f32,
        rtpp: f32,
        msd: f32,
        qiv: f32,
        ng: f32,
    }

    let raw: Vec<VoxelScalars> = (0..n_voxels)
        .into_par_iter()
        .map_init(
            || vec![0.0_f64; n_shore],
            |shore_coefs, v| {
                let (i, j, k) = masked_indices[v];
                for c in 0..n_shore {
                    shore_coefs[c] = coeffs[(i, j, k, c)] as f64;
                }
                let dir = peak0_dirs
                    .and_then(|d| d.get(v).copied().flatten())
                    .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                    .unwrap_or([0.0, 0.0, 0.0]);
                VoxelScalars {
                    rtop: microstructure::rtop(shore_coefs, &info) as f32,
                    msd: microstructure::msd(shore_coefs, &info) as f32,
                    ng: microstructure::ng(shore_coefs, &info) as f32,
                    qiv: microstructure::qiv(shore_coefs, &info) as f32,
                    rtap: microstructure::rtap(shore_coefs, &info, dir) as f32,
                    rtpp: microstructure::rtpp(shore_coefs, &info, dir) as f32,
                }
            },
        )
        .collect();

    let mut rtop: Vec<f32> = raw.iter().map(|s| s.rtop * s_rtop).collect();
    let mut rtap: Vec<f32> = raw.iter().map(|s| s.rtap * s_rtap).collect();
    let mut rtpp: Vec<f32> = raw.iter().map(|s| s.rtpp * s_rtpp).collect();
    let mut msd: Vec<f32> = raw.iter().map(|s| s.msd * s_msd).collect();
    let mut qiv: Vec<f32> = raw.iter().map(|s| s.qiv * s_qiv).collect();
    let mut ng: Vec<f32> = raw.iter().map(|s| s.ng * s_ng).collect();

    // Step 1: fit-failure rejection. NaN voxels whose scalar is implausibly
    // large vs the brain-wide 99th percentile.
    let outlier_report = opts.outlier_rejection.as_ref().map(|cfg| {
        let mut stats: BTreeMap<String, ScalarOutlierStat> = BTreeMap::new();
        for (name, vec_ref) in [
            ("rtop", &mut rtop),
            ("rtap", &mut rtap),
            ("rtpp", &mut rtpp),
            ("msd", &mut msd),
            ("qiv", &mut qiv),
            ("ng", &mut ng),
        ] {
            let finite_count_before = vec_ref.iter().filter(|v| v.is_finite()).count();
            let p99 = percentile_value(vec_ref, 99.0);
            let stat = match p99 {
                Some(p) => {
                    let threshold = cfg.p99_factor * p;
                    let rejected = nan_above(vec_ref, threshold);
                    if !opts.quiet {
                        let pct = 100.0 * rejected as f64 / finite_count_before.max(1) as f64;
                        eprintln!(
                            "[microstructure] {name}: rejected {rejected}/{finite_count_before} ({pct:.3}%) as fit failures above {:.3e} ({}× p99={:.3e})",
                            threshold,
                            cfg.p99_factor,
                            p
                        );
                    }
                    ScalarOutlierStat {
                        finite_count_before,
                        rejected_count: rejected,
                        threshold: Some(threshold),
                        p99: Some(p),
                    }
                }
                None => {
                    if !opts.quiet {
                        eprintln!(
                            "[microstructure] {name}: no finite values (e.g. RTAP/RTPP without peaks); skipped rejection"
                        );
                    }
                    ScalarOutlierStat {
                        finite_count_before,
                        rejected_count: 0,
                        threshold: None,
                        p99: None,
                    }
                }
            };
            stats.insert(name.to_string(), stat);
        }
        MicrostructureOutlierReport {
            p99_factor: cfg.p99_factor,
            n_voxels,
            stats,
        }
    });

    // Step 2: compute display stats over the *post-rejection* distribution
    // so the suggested colormap range tracks data we trust.
    let mut display_stats = MicrostructureDisplayStats {
        n_voxels,
        stats: BTreeMap::new(),
    };
    for (name, vec_ref) in [
        ("rtop", &rtop),
        ("rtap", &rtap),
        ("rtpp", &rtpp),
        ("msd", &msd),
        ("qiv", &qiv),
        ("ng", &ng),
    ] {
        let stat = compute_display_stat(vec_ref, name);
        if !opts.quiet {
            log_display_stat(name, &stat, n_voxels);
        }
        display_stats.stats.insert(name.to_string(), stat);
    }

    MicrostructureScalars { rtop, rtap, rtpp, msd, qiv, ng, display_stats, outlier_report }
}

/// Read-only sister of `percentile_cap`: returns the percentile value
/// without mutating anything. `None` when no finite values are present.
fn percentile_value(xs: &[f32], percentile: f64) -> Option<f32> {
    let mut finite: Vec<f32> = xs.iter().filter(|v| v.is_finite()).copied().collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = finite.len();
    let p = percentile.clamp(0.0, 100.0) / 100.0;
    let idx = ((p * (n - 1) as f64).round() as usize).min(n - 1);
    Some(finite[idx])
}

/// Set every finite value in `xs` strictly greater than `threshold` to NaN.
/// Returns the number of voxels modified. NaN/inf pass through untouched.
fn nan_above(xs: &mut [f32], threshold: f32) -> usize {
    let mut rejected = 0;
    for v in xs.iter_mut() {
        if v.is_finite() && *v > threshold {
            *v = f32::NAN;
            rejected += 1;
        }
    }
    rejected
}

fn compute_display_stat(values: &[f32], name: &str) -> ScalarDisplayStat {
    let mut finite: Vec<f32> = values.iter().filter(|v| v.is_finite()).copied().collect();
    let finite_count = finite.len();
    if finite.is_empty() {
        return ScalarDisplayStat {
            finite_count: 0,
            min: None,
            median: None,
            p95: None,
            p99: None,
            max: None,
            suggested_max: None,
        };
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = finite.len();
    let pick = |p: f64| -> f32 {
        let p = p.clamp(0.0, 1.0);
        finite[((p * (n - 1) as f64).round() as usize).min(n - 1)]
    };
    let min = finite[0];
    let max = finite[n - 1];
    let median = pick(0.50);
    let p95 = pick(0.95);
    let p99 = pick(0.99);
    // For NG (bounded [0, 1] physically), cap the suggestion at 1.0 so a
    // numerical leak above 1 doesn't widen the colormap. For everything
    // else, p99 is the conservative upper bound for visualization.
    let suggested_max = if name == "ng" {
        Some(p99.min(1.0))
    } else {
        Some(p99)
    };
    ScalarDisplayStat {
        finite_count,
        min: Some(min),
        median: Some(median),
        p95: Some(p95),
        p99: Some(p99),
        max: Some(max),
        suggested_max,
    }
}

fn log_display_stat(name: &str, stat: &ScalarDisplayStat, n_voxels: usize) {
    match (stat.median, stat.p99, stat.max) {
        (Some(m), Some(p99), Some(mx)) => eprintln!(
            "[microstructure] {name}: median={m:.3e} p99={p99:.3e} max={mx:.3e} (suggested colormap [0, {p99:.3e}]; finite {}/{n_voxels})",
            stat.finite_count
        ),
        _ => eprintln!(
            "[microstructure] {name}: no finite values (e.g. RTAP/RTPP without peaks)"
        ),
    }
}

/// Embed the six per-voxel scalar arrays into the ODX as float32 DPVs (one
/// value per masked voxel). Always stamps `cs_dmri_microstructure_display`
/// with per-scalar robust stats + a suggested `[0, p99]` colormap range.
/// Additionally stamps `cs_dmri_microstructure_outliers` if rejection was
/// enabled, so the threshold and per-scalar reject counts are auditable.
pub fn embed_microstructure_dpvs(
    builder: &mut OdxBuilder,
    scalars: &MicrostructureScalars,
) {
    for (name, values) in [
        ("rtop", &scalars.rtop),
        ("rtap", &scalars.rtap),
        ("rtpp", &scalars.rtpp),
        ("msd", &scalars.msd),
        ("qiv", &scalars.qiv),
        ("ng", &scalars.ng),
    ] {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        builder.set_dpv_data(name, bytes, 1, DType::Float32);
    }
    builder.set_extra_value(
        "cs_dmri_microstructure_display",
        display_stats_json(&scalars.display_stats),
    );
    if let Some(report) = &scalars.outlier_report {
        builder.set_extra_value(
            "cs_dmri_microstructure_outliers",
            outlier_report_json(report),
        );
    }
}

fn outlier_report_json(report: &MicrostructureOutlierReport) -> serde_json::Value {
    let mut by_scalar = serde_json::Map::new();
    for (name, s) in &report.stats {
        by_scalar.insert(
            name.clone(),
            serde_json::json!({
                "finite_count_before": s.finite_count_before,
                "rejected_count": s.rejected_count,
                "threshold": s.threshold.map(|v| v as f64),
                "p99": s.p99.map(|v| v as f64),
            }),
        );
    }
    serde_json::json!({
        "p99_factor": report.p99_factor as f64,
        "n_voxels": report.n_voxels,
        "scalars": serde_json::Value::Object(by_scalar),
    })
}

fn display_stats_json(stats: &MicrostructureDisplayStats) -> serde_json::Value {
    let mut by_scalar = serde_json::Map::new();
    for (name, s) in &stats.stats {
        by_scalar.insert(
            name.clone(),
            serde_json::json!({
                "finite_count": s.finite_count,
                "min": s.min.map(|v| v as f64),
                "median": s.median.map(|v| v as f64),
                "p95": s.p95.map(|v| v as f64),
                "p99": s.p99.map(|v| v as f64),
                "max": s.max.map(|v| v as f64),
                "suggested_max": s.suggested_max.map(|v| v as f64),
            }),
        );
    }
    serde_json::json!({
        "n_voxels": stats.n_voxels,
        "scalars": serde_json::Value::Object(by_scalar),
    })
}

/// Write each scalar to a sibling NIfTI of `output_path`. Used by
/// `cs-odf --microstructure-nifti`.
///
/// `output_path` may end in `.odx`, `.nii`, `.nii.gz`, or have no extension —
/// the suffix is stripped before appending `_<scalar>.nii.gz`. `affine` and
/// `spatial` describe the canonical voxel grid so the on-disk NIfTI matches
/// the ODX's frame regardless of the input's original orientation.
pub fn write_microstructure_nifti_siblings(
    output_path: &Path,
    affine: &[[f64; 4]; 4],
    spatial: [usize; 3],
    masked_indices: &[(usize, usize, usize)],
    scalars: &MicrostructureScalars,
    overwrite: bool,
    quiet: bool,
) -> Result<()> {
    for (name, values) in [
        ("rtop", &scalars.rtop),
        ("rtap", &scalars.rtap),
        ("rtpp", &scalars.rtpp),
        ("msd", &scalars.msd),
        ("qiv", &scalars.qiv),
        ("ng", &scalars.ng),
    ] {
        let mut volume = Array3::<f32>::zeros((spatial[0], spatial[1], spatial[2]));
        for (v, &(i, j, k)) in masked_indices.iter().enumerate() {
            volume[(i, j, k)] = values[v];
        }
        let out_nii = scalar_sibling_path(output_path, name);
        crate::io::output::write_3d_f32_with_affine(&out_nii, affine, &volume, overwrite)
            .map_err(|e| anyhow!("write {:?}: {e}", out_nii))?;
        if !quiet {
            eprintln!("[microstructure] wrote {}", out_nii.display());
        }
    }
    Ok(())
}

/// `<output_stem>_<scalar>.nii.gz` next to the ODX (or coefficient) output.
/// Strips `.odx`, `.nii.gz`, or `.nii` from `output` before appending.
pub fn scalar_sibling_path(output: &Path, scalar: &str) -> std::path::PathBuf {
    let s = output.to_string_lossy();
    let stem = if let Some(stripped) = s.strip_suffix(".odx") {
        stripped
    } else if let Some(stripped) = s.strip_suffix(".nii.gz") {
        stripped
    } else if let Some(stripped) = s.strip_suffix(".nii") {
        stripped
    } else {
        s.as_ref()
    };
    std::path::PathBuf::from(format!("{}_{}.nii.gz", stem, scalar))
}

/// Upper-percentile of finite values in `xs`. Returns `None` if no finite
/// values exist (e.g. RTAP/RTPP under `--no-peaks`).
pub fn percentile_cap(xs: &[f32], percentile: f64) -> Option<f32> {
    let mut finite: Vec<f32> = xs.iter().filter(|v| v.is_finite()).copied().collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = finite.len();
    let p = percentile.clamp(0.0, 100.0) / 100.0;
    let idx = ((p * (n - 1) as f64).round() as usize).min(n - 1);
    Some(finite[idx])
}

/// Cap finite values above `cap` to `cap`. Returns the number of voxels
/// modified. NaN/inf values pass through unchanged.
pub fn winsorize(xs: &mut [f32], cap: f32) -> usize {
    let mut clipped = 0;
    for v in xs.iter_mut() {
        if v.is_finite() && *v > cap {
            *v = cap;
            clipped += 1;
        }
    }
    clipped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_cap_basic() {
        // Sorted finite values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
        let xs: Vec<f32> = (0..10).map(|v| v as f32).collect();
        // 50th percentile of 10 values: idx round(0.5 * 9) = 5 → value 5.
        assert_eq!(percentile_cap(&xs, 50.0), Some(5.0));
        // 100th → last element.
        assert_eq!(percentile_cap(&xs, 100.0), Some(9.0));
        // 0th → first.
        assert_eq!(percentile_cap(&xs, 0.0), Some(0.0));
    }

    #[test]
    fn percentile_cap_skips_nan_and_inf() {
        let xs = [0.0_f32, 1.0, f32::NAN, 2.0, f32::INFINITY, 3.0];
        // Finite values: [0, 1, 2, 3]. p100 = 3.0.
        assert_eq!(percentile_cap(&xs, 100.0), Some(3.0));
    }

    #[test]
    fn percentile_cap_all_nan_is_none() {
        let xs = [f32::NAN, f32::NAN, f32::NAN];
        assert_eq!(percentile_cap(&xs, 99.5), None);
    }

    #[test]
    fn winsorize_clips_above_cap_only() {
        let mut xs = [0.0_f32, 5.0, 10.0, 100.0, f32::NAN];
        let n = winsorize(&mut xs, 10.0);
        assert_eq!(n, 1); // only 100 was above the cap
        assert_eq!(xs[0], 0.0);
        assert_eq!(xs[1], 5.0);
        assert_eq!(xs[2], 10.0); // boundary not clipped
        assert_eq!(xs[3], 10.0); // clipped from 100
        assert!(xs[4].is_nan()); // NaN preserved
    }

    #[test]
    fn winsorize_preserves_negative_infinity_and_nan() {
        let mut xs = [f32::NEG_INFINITY, f32::INFINITY, f32::NAN, 5.0];
        let n = winsorize(&mut xs, 10.0);
        assert_eq!(n, 0); // ±∞ is not finite, NaN not finite, 5.0 ≤ cap
        assert!(xs[0].is_infinite() && xs[0] < 0.0);
        assert!(xs[1].is_infinite() && xs[1] > 0.0);
        assert!(xs[2].is_nan());
        assert_eq!(xs[3], 5.0);
    }

    #[test]
    fn display_stat_caps_ng_at_one() {
        let xs: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        // p99 over [0, .01, …, .99] is ≈ 0.99; suggested_max = min(p99, 1).
        let stat = compute_display_stat(&xs, "ng");
        assert!(stat.suggested_max.unwrap() <= 1.0);
    }

    #[test]
    fn display_stat_uses_p99_for_unbounded_scalars() {
        let mut xs: Vec<f32> = (0..1000).map(|i| (i as f32) / 10.0).collect();
        xs.push(1e8_f32);
        let stat = compute_display_stat(&xs, "rtop");
        // suggested_max = p99, which excludes the outlier.
        assert!(
            stat.suggested_max.unwrap() < 200.0,
            "got {:?}",
            stat.suggested_max
        );
        assert_eq!(stat.max, Some(1e8_f32));
    }

    #[test]
    fn display_stat_no_finite_values() {
        let xs = [f32::NAN, f32::NAN];
        let stat = compute_display_stat(&xs, "rtap");
        assert_eq!(stat.finite_count, 0);
        assert!(stat.suggested_max.is_none());
    }

    #[test]
    fn nan_above_only_finite_overshoots() {
        let mut xs = [1.0_f32, 5.0, 10.0, f32::NAN, f32::INFINITY];
        let n = nan_above(&mut xs, 4.0);
        // 1.0 ≤ 4 keep, 5.0 > 4 → NaN, 10.0 > 4 → NaN, NaN preserved, ∞ not
        // finite → preserved unchanged.
        assert_eq!(n, 2);
        assert_eq!(xs[0], 1.0);
        assert!(xs[1].is_nan());
        assert!(xs[2].is_nan());
        assert!(xs[3].is_nan());
        assert!(xs[4].is_infinite() && xs[4] > 0.0);
    }

    #[test]
    fn percentile_value_matches_percentile_cap() {
        let xs: Vec<f32> = (0..100).map(|i| i as f32).collect();
        // Percentile reading should agree with the cap-style mutating fn.
        for p in [0.0_f64, 25.0, 50.0, 99.0, 100.0] {
            assert_eq!(percentile_value(&xs, p), percentile_cap(&xs, p));
        }
    }

    #[test]
    fn outlier_rejection_catches_huge_outliers() {
        // 1000 voxels with one extreme outlier. p99 of finite ≈ 99.0;
        // K=10 threshold = ~990 → outlier (1e8) gets NaN'd, rest survive.
        let mut xs: Vec<f32> = (0..1000).map(|i| i as f32 / 10.0).collect();
        xs.push(1e8_f32);
        let p99 = percentile_value(&xs, 99.0).unwrap();
        let threshold = 10.0 * p99;
        let n = nan_above(&mut xs, threshold);
        assert_eq!(n, 1);
        assert!(xs.last().unwrap().is_nan());
        // Body of distribution is intact.
        let surviving: Vec<f32> = xs.iter().filter(|v| v.is_finite()).copied().collect();
        assert_eq!(surviving.len(), 1000);
    }

    #[test]
    fn percentile_cap_99_5_on_realistic_outliers() {
        // Simulate 1000 voxels with one extreme outlier.
        let mut xs: Vec<f32> = (0..1000).map(|i| (i as f32) / 10.0).collect();
        xs.push(1e8_f32);
        // Without the outlier, range is 0..99.9. p99.5 over the now-1001
        // sorted values picks index round(0.995 * 1000) = 995 → value 99.5.
        let cap = percentile_cap(&xs, 99.5).unwrap();
        assert!(
            cap < 100.0,
            "p99.5 should ignore the lone outlier, got {cap}"
        );
        let mut xs_mut = xs.clone();
        let n = winsorize(&mut xs_mut, cap);
        assert!(n >= 1, "the outlier should be clipped");
    }
}
