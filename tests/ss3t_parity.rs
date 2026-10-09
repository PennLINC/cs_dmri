// SPDX-License-Identifier: MIT OR Apache-2.0
//! Parity test: cs-ss3t vs MRtrix3Tissue reference outputs.
//!
//! Local-only, opt-in via `cargo test --release -- --ignored ss3t_parity`.
//! Requires the test bundle at `<repo>/../ss3t_test_data/` (sibling of cs-dmri).
//! Writes cs-ss3t NIfTIs and reference-derived NIfTIs to
//! `target/ss3t_parity/` for off-line inspection.
//!

use std::path::{Path, PathBuf};

use ndarray::{Array3, Array4};
use nifti::writer::WriterOptions;
use nifti::{NiftiObject, ReaderOptions};

use cs_dmri::io::dwi::load_dwi;
use cs_dmri::multitissue::ss3t::{Ss3tConfig, Ss3tResponses};
use cs_dmri::multitissue::volume::{Ss3tFitConfig, fit_volume_ss3t_reporting};
use cs_dmri::multitissue::TissueResponse;
use cs_dmri::qspace::BvecFrame;
use cs_dmri::solver::csd::CsdConfig;

fn test_data_dir() -> Option<PathBuf> {
    let candidate = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("ss3t_test_data");
    candidate.is_dir().then_some(candidate)
}

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("ss3t_parity");
    std::fs::create_dir_all(&dir).expect("create output dir");
    dir
}

#[test]
#[ignore = "local-only parity test against MRtrix3Tissue reference; opt-in via --ignored"]
fn ss3t_parity_against_mrtrix_reference() {
    let data = match test_data_dir() {
        Some(d) => d,
        None => {
            eprintln!(
                "[ss3t_parity] skipping: ss3t_test_data/ not found alongside cs-dmri/"
            );
            return;
        }
    };
    eprintln!("[ss3t_parity] test data: {}", data.display());

    let stem = "sub-01_ses-1_space-ACPC";
    let dwi_path = data.join(format!("{stem}_desc-preproc_dwi.nii.gz"));
    let bval_path = data.join(format!("{stem}_desc-preproc_dwi.bval"));
    let bvec_path = data.join(format!("{stem}_desc-preproc_dwi.bvec"));
    let mask_path = data.join(format!("{stem}_desc-brain_mask.nii.gz"));
    let resp_wm = data.join(format!("{stem}_model-ss3t_param-fod_label-WM_dwimap.txt"));
    let resp_gm = data.join(format!("{stem}_model-ss3t_param-fod_label-GM_dwimap.txt"));
    let resp_csf = data.join(format!("{stem}_model-ss3t_param-fod_label-CSF_dwimap.txt"));
    let ref_wm_mif = data.join(format!("{stem}_model-ss3t_param-fod_label-WM_dwimap.mif.gz"));
    let ref_gm_mif = data.join(format!("{stem}_model-ss3t_param-fod_label-GM_dwimap.mif.gz"));
    let ref_csf_mif = data.join(format!("{stem}_model-ss3t_param-fod_label-CSF_dwimap.mif.gz"));

    eprintln!("[ss3t_parity] loading DWI bundle...");
    let dwi = load_dwi(
        &dwi_path,
        &bval_path,
        &bvec_path,
        Some(&mask_path),
        None,
        None,
        None,
        BvecFrame::WorldRas,
    )
    .expect("load DWI");
    let mask_voxels = dwi.mask.iter().filter(|&&v| v).count();
    let shape = dwi.shape();
    eprintln!(
        "[ss3t_parity] DWI shape {:?}, mask voxels: {}",
        shape, mask_voxels
    );

    let responses = Ss3tResponses {
        wm: TissueResponse::parse_mrtrix_txt(&resp_wm).expect("parse WM response"),
        gm: TissueResponse::parse_mrtrix_txt(&resp_gm).expect("parse GM response"),
        csf: TissueResponse::parse_mrtrix_txt(&resp_csf).expect("parse CSF response"),
    };
    eprintln!(
        "[ss3t_parity] responses: WM lmax={}, GM lmax={}, CSF lmax={}",
        responses.wm.lmax, responses.gm.lmax, responses.csf.lmax
    );

    // Defaults (niter=3, bzero_pct=10, lmax_wm=8, csd.lambda=1000) match
    // qsirecon's invocation; lmax_wm is qsirecon's hardcoded value.
    let cfg = Ss3tConfig::default();

    eprintln!("[ss3t_parity] fitting (this may take a few minutes)...");
    let t0 = std::time::Instant::now();
    let result = fit_volume_ss3t_reporting(
        &dwi,
        &responses,
        &cfg,
        Ss3tFitConfig { compute_diagnostics: true },
        || {},
    )
    .expect("ss3t fit");
    let elapsed = t0.elapsed();
    eprintln!(
        "[ss3t_parity] fit done in {:.1}s ({:.0} voxels/s)",
        elapsed.as_secs_f64(),
        mask_voxels as f64 / elapsed.as_secs_f64()
    );

    // Write cs-ss3t outputs.
    let out = output_dir();
    eprintln!("[ss3t_parity] writing outputs to {}", out.display());
    let cs_wm_path = out.join("cs_dmri_wm.nii.gz");
    let cs_gm_path = out.join("cs_dmri_gm.nii.gz");
    let cs_csf_path = out.join("cs_dmri_csf.nii.gz");
    write_4d_nifti(&cs_wm_path, &dwi_path, &result.wm);
    write_4d_nifti(&cs_gm_path, &dwi_path, &result.gm);
    write_4d_nifti(&cs_csf_path, &dwi_path, &result.csf);
    if let Some(r) = &result.residual_l2 {
        write_3d_nifti_f32(&out.join("cs_dmri_residual.nii.gz"), &dwi_path, r);
    }
    if let Some(i) = &result.iterations {
        let f = i.mapv(|v| v as f32);
        write_3d_nifti_f32(&out.join("cs_dmri_iterations.nii.gz"), &dwi_path, &f);
    }

    // If mtnormalise is on PATH, run it on our outputs so we can compare
    // normalized-vs-normalized (qsirecon's reference is post-mtnormalise).
    let normalized = run_mtnormalise(
        &cs_wm_path,
        &cs_gm_path,
        &cs_csf_path,
        &mask_path,
        &out,
    );
    let (cs_wm_for_compare, cs_gm_for_compare, cs_csf_for_compare, comparison_label) =
        match &normalized {
            Some((wm, gm, csf)) => {
                eprintln!("[ss3t_parity] using mtnormalised outputs for comparison");
                (
                    load_nifti_4d(wm),
                    load_nifti_4d(gm),
                    load_nifti_4d(csf),
                    "mtnormalised",
                )
            }
            None => {
                eprintln!(
                    "[ss3t_parity] mtnormalise not found on PATH or failed — comparing raw outputs"
                );
                (result.wm.clone(), result.gm.clone(), result.csf.clone(), "raw")
            }
        };

    // Load reference .mif.gz files, aligned to the input DWI's affine so
    // voxel indices line up with cs-ss3t's output (which inherits the input
    // NIfTI grid).
    eprintln!("[ss3t_parity] loading reference MIF files...");
    let target_affine = odx_rs::reference_affine::read_reference_affine(&dwi_path)
        .expect("read input NIfTI affine");
    let ref_wm = load_mif_4d_aligned(&ref_wm_mif, &target_affine).expect("load reference WM mif");
    let ref_gm = load_mif_4d_aligned(&ref_gm_mif, &target_affine).expect("load reference GM mif");
    let ref_csf =
        load_mif_4d_aligned(&ref_csf_mif, &target_affine).expect("load reference CSF mif");
    eprintln!(
        "[ss3t_parity] reference shapes: WM {:?}, GM {:?}, CSF {:?}",
        ref_wm.shape(),
        ref_gm.shape(),
        ref_csf.shape()
    );

    // Convert reference to NIfTI for easy side-by-side viewing.
    write_4d_nifti(&out.join("reference_wm.nii.gz"), &dwi_path, &ref_wm);
    write_4d_nifti(&out.join("reference_gm.nii.gz"), &dwi_path, &ref_gm);
    write_4d_nifti(&out.join("reference_csf.nii.gz"), &dwi_path, &ref_csf);

    // Sanity: dims must match cs_dmri outputs.
    assert_eq!(
        ref_wm.shape(),
        cs_wm_for_compare.shape(),
        "reference WM dims {:?} vs cs-ss3t WM dims {:?}",
        ref_wm.shape(),
        cs_wm_for_compare.shape()
    );
    assert_eq!(
        ref_gm.shape(),
        cs_gm_for_compare.shape(),
        "reference GM dims {:?} vs cs-ss3t GM dims {:?}",
        ref_gm.shape(),
        cs_gm_for_compare.shape()
    );
    assert_eq!(
        ref_csf.shape(),
        cs_csf_for_compare.shape(),
        "reference CSF dims {:?} vs cs-ss3t CSF dims {:?}",
        ref_csf.shape(),
        cs_csf_for_compare.shape()
    );

    // Compare. Print a summary; no hard threshold asserts (per user request:
    // "we can address acceptance once we see how it turns out").
    let wm_report = compare_4d(&cs_wm_for_compare, &ref_wm, &dwi.mask, "WM");
    let gm_report = compare_4d(&cs_gm_for_compare, &ref_gm, &dwi.mask, "GM");
    let csf_report = compare_4d(&cs_csf_for_compare, &ref_csf, &dwi.mask, "CSF");

    eprintln!("\n=== ss3t parity report ({}) ===", comparison_label);
    eprintln!("{}", wm_report);
    eprintln!("{}", gm_report);
    eprintln!("{}", csf_report);
    eprintln!(
        "Outputs written to: {} (cs_dmri_*.nii.gz, *_norm.nii.gz, reference_*.nii.gz)",
        out.display()
    );
    eprintln!("==========================\n");
}

/// If `mtnormalise` is on PATH, run it on the three cs-ss3t outputs and return
/// paths to the normalized versions. Returns `None` if the binary isn't
/// present, can't be invoked, or exits non-zero.
fn run_mtnormalise(
    wm_in: &Path,
    gm_in: &Path,
    csf_in: &Path,
    mask: &Path,
    out_dir: &Path,
) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let wm_out = out_dir.join("cs_dmri_wm_norm.nii.gz");
    let gm_out = out_dir.join("cs_dmri_gm_norm.nii.gz");
    let csf_out = out_dir.join("cs_dmri_csf_norm.nii.gz");
    for p in [&wm_out, &gm_out, &csf_out] {
        if p.exists() {
            let _ = std::fs::remove_file(p);
        }
    }

    eprintln!("[ss3t_parity] running mtnormalise...");
    let status = std::process::Command::new("mtnormalise")
        .arg(wm_in)
        .arg(&wm_out)
        .arg(gm_in)
        .arg(&gm_out)
        .arg(csf_in)
        .arg(&csf_out)
        .arg("-mask")
        .arg(mask)
        .arg("-force")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
        .status();

    match status {
        Ok(s) if s.success() && wm_out.exists() && gm_out.exists() && csf_out.exists() => {
            Some((wm_out, gm_out, csf_out))
        }
        Ok(s) => {
            eprintln!("[ss3t_parity] mtnormalise exited with status {:?}", s);
            None
        }
        Err(e) => {
            eprintln!("[ss3t_parity] could not invoke mtnormalise: {e}");
            None
        }
    }
}

fn load_nifti_4d(path: &Path) -> Array4<f32> {
    use nifti::IntoNdArray;
    let obj = ReaderOptions::new()
        .read_file(path)
        .expect("read normalized NIfTI");
    let nd = obj
        .into_volume()
        .into_ndarray::<f32>()
        .expect("nifti -> ndarray");
    // mtnormalise emits 4D for the WM (multi-volume SH) and 3D for the
    // single-coefficient GM/CSF. Promote 3D to 4D-with-singleton to keep
    // downstream comparison code uniform.
    match nd.ndim() {
        4 => nd
            .into_dimensionality::<ndarray::Ix4>()
            .expect("Ix4 cast"),
        3 => {
            let arr3 = nd
                .into_dimensionality::<ndarray::Ix3>()
                .expect("Ix3 cast");
            let s = arr3.shape();
            let (nx, ny, nz) = (s[0], s[1], s[2]);
            let mut arr4 = Array4::<f32>::zeros((nx, ny, nz, 1));
            for x in 0..nx {
                for y in 0..ny {
                    for z in 0..nz {
                        arr4[(x, y, z, 0)] = arr3[(x, y, z)];
                    }
                }
            }
            arr4
        }
        d => panic!("expected 3D or 4D NIfTI, got {d}D"),
    }
}

/// Per-voxel comparison summary.
struct CompareReport {
    label: &'static str,
    n_voxels: usize,
    /// Per-voxel Pearson correlation of the coefficient vector. Reported as
    /// (median, p10, p90) over masked voxels with non-trivial reference norm.
    voxel_corr: Option<(f64, f64, f64)>,
    /// Per-voxel relative L2 error: ‖cs - ref‖₂ / ‖ref‖₂. (median, p10, p90).
    rel_l2: Option<(f64, f64, f64)>,
    /// Frobenius-norm ratio over the mask: ‖cs‖_F / ‖ref‖_F.
    frob_ratio: f64,
    /// Per-coefficient Pearson correlation across voxels (one r per SH index).
    /// Reported as (min, median, max) when more than 1 coefficient.
    coeff_corr: Option<(f64, f64, f64)>,
}

impl std::fmt::Display for CompareReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[{}] voxels compared: {}", self.label, self.n_voxels)?;
        if let Some((m, p10, p90)) = self.voxel_corr {
            writeln!(
                f,
                "[{}] per-voxel Pearson r: median={:.4}, p10={:.4}, p90={:.4}",
                self.label, m, p10, p90
            )?;
        }
        if let Some((m, p10, p90)) = self.rel_l2 {
            writeln!(
                f,
                "[{}] per-voxel ‖cs-ref‖/‖ref‖: median={:.3e}, p10={:.3e}, p90={:.3e}",
                self.label, m, p10, p90
            )?;
        }
        writeln!(
            f,
            "[{}] Frobenius-norm ratio ‖cs‖/‖ref‖ over mask: {:.4}",
            self.label, self.frob_ratio
        )?;
        if let Some((mn, md, mx)) = self.coeff_corr {
            writeln!(
                f,
                "[{}] per-coefficient r across voxels: min={:.4}, median={:.4}, max={:.4}",
                self.label, mn, md, mx
            )?;
        }
        Ok(())
    }
}

fn compare_4d(
    cs: &Array4<f32>,
    reference: &Array4<f32>,
    mask: &Array3<bool>,
    label: &'static str,
) -> CompareReport {
    let s = cs.shape();
    let (nx, ny, nz, nc) = (s[0], s[1], s[2], s[3]);

    let mut voxel_corrs = Vec::<f64>::new();
    let mut rel_l2 = Vec::<f64>::new();
    let mut cs_frob = 0.0_f64;
    let mut ref_frob = 0.0_f64;
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if !mask[(x, y, z)] {
                    continue;
                }
                let mut a = Vec::with_capacity(nc);
                let mut b = Vec::with_capacity(nc);
                for k in 0..nc {
                    a.push(cs[(x, y, z, k)] as f64);
                    b.push(reference[(x, y, z, k)] as f64);
                }
                cs_frob += a.iter().map(|v| v * v).sum::<f64>();
                ref_frob += b.iter().map(|v| v * v).sum::<f64>();

                let ref_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
                if ref_norm < 1e-12 {
                    continue;
                }
                let diff_norm: f64 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(p, q)| (p - q).powi(2))
                    .sum::<f64>()
                    .sqrt();
                rel_l2.push(diff_norm / ref_norm);

                if nc >= 2 {
                    if let Some(r) = pearson(&a, &b) {
                        voxel_corrs.push(r);
                    }
                }
            }
        }
    }

    let n_voxels = rel_l2.len();
    let voxel_corr = (!voxel_corrs.is_empty()).then(|| percentiles(&mut voxel_corrs));
    let rel_l2_p = (!rel_l2.is_empty()).then(|| percentiles(&mut rel_l2));
    let frob_ratio = if ref_frob > 0.0 {
        (cs_frob / ref_frob).sqrt()
    } else {
        f64::NAN
    };
    // Per-coefficient Pearson correlation across voxels — works for any nc
    // ≥ 1, giving us a "spatial pattern" check on each output channel
    // (especially useful for GM/CSF where there's only one coefficient).
    let mut per_coeff: Vec<f64> = Vec::with_capacity(nc);
    for k in 0..nc {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    if mask[(x, y, z)] {
                        a.push(cs[(x, y, z, k)] as f64);
                        b.push(reference[(x, y, z, k)] as f64);
                    }
                }
            }
        }
        if let Some(r) = pearson(&a, &b) {
            per_coeff.push(r);
        }
    }
    let coeff_corr = if per_coeff.is_empty() {
        None
    } else {
        per_coeff.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mn = per_coeff[0];
        let md = per_coeff[per_coeff.len() / 2];
        let mx = per_coeff[per_coeff.len() - 1];
        Some((mn, md, mx))
    };

    CompareReport {
        label,
        n_voxels,
        voxel_corr,
        rel_l2: rel_l2_p,
        frob_ratio,
        coeff_corr,
    }
}

fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len();
    if n < 2 {
        return None;
    }
    let ma = a.iter().sum::<f64>() / n as f64;
    let mb = b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut da = 0.0_f64;
    let mut db = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - ma;
        let dy = y - mb;
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da <= 0.0 || db <= 0.0 {
        return None;
    }
    Some(num / (da.sqrt() * db.sqrt()))
}

fn percentiles(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f64| -> f64 {
        let i = ((v.len() - 1) as f64 * q).round() as usize;
        v[i.min(v.len() - 1)]
    };
    (pick(0.5), pick(0.10), pick(0.90))
}

/// Load an MRtrix `.mif` or `.mif.gz` file as a 4D `Array4<f32>` whose
/// `(x, y, z)` indices match the supplied target NIfTI affine (typically the
/// input DWI's). Two steps:
///
/// 1. Decode the MIF in its own logical (i, j, k) order — `compute_strides`
///    handles the per-axis "negative" flags that MRtrix uses to encode flipped
///    storage relative to its `transform`.
/// 2. Flip along any axis where the MIF transform's column sign disagrees
///    with the target affine's column sign (e.g. RAS-stored MIF vs LPS-stored
///    NIfTI: flip X and Y). Assumes axes are aligned (no permutations needed),
///    which is the case for any data that stays on the same scanner grid.
fn load_mif_4d_aligned(path: &Path, target_affine: &[[f64; 4]; 4]) -> Result<Array4<f32>, String> {
    let img = odx_rs::formats::mif::read_mif(path).map_err(|e| format!("read_mif: {e}"))?;
    let dims = img.header.dimensions.clone();
    if dims.len() != 4 {
        return Err(format!("expected 4D MIF, got {}D ({:?})", dims.len(), dims));
    }
    let raw = img.as_f32_vec();
    if raw.is_empty() {
        return Err(format!(
            "unsupported MIF datatype {:?}",
            img.header.datatype
        ));
    }
    let strides = img.header.compute_strides();
    let neg_offset: isize = strides
        .iter()
        .enumerate()
        .filter(|(_, &s)| s < 0)
        .map(|(ax, &s)| (dims[ax] as isize - 1) * (-s))
        .sum();

    // Step 1: decode in MIF logical order.
    let mut decoded = Array4::<f32>::zeros((dims[0], dims[1], dims[2], dims[3]));
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for c in 0..dims[3] {
                    let src = neg_offset
                        + (x as isize) * strides[0]
                        + (y as isize) * strides[1]
                        + (z as isize) * strides[2]
                        + (c as isize) * strides[3];
                    decoded[(x, y, z, c)] = raw[src as usize];
                }
            }
        }
    }

    // Step 2: align to target affine by flipping axes whose column-sign
    // disagrees. Default (identity) MIF transform → RAS-canonical; LPS-stored
    // NIfTI → flip X and Y.
    let mif_affine = match img.header.transform {
        Some(t) => [
            [t[0][0], t[0][1], t[0][2], t[0][3]],
            [t[1][0], t[1][1], t[1][2], t[1][3]],
            [t[2][0], t[2][1], t[2][2], t[2][3]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        None => [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };
    let flips = derive_axis_flips(&mif_affine, target_affine)?;
    let aligned = apply_flips(decoded, flips);
    Ok(aligned)
}

/// For each of the 3 spatial axes, return `true` if the MIF data needs to be
/// flipped along that axis to match the target NIfTI affine. Errors out if
/// the affines aren't axis-aligned (would need a permutation).
fn derive_axis_flips(
    mif_affine: &[[f64; 4]; 4],
    target_affine: &[[f64; 4]; 4],
) -> Result<[bool; 3], String> {
    let mut flips = [false; 3];
    for axis in 0..3 {
        // Identify the world component (R/A/S) that this image axis maps to,
        // for both affines. Largest-magnitude entry of the column wins.
        let mif_col = [mif_affine[0][axis], mif_affine[1][axis], mif_affine[2][axis]];
        let tgt_col = [
            target_affine[0][axis],
            target_affine[1][axis],
            target_affine[2][axis],
        ];
        let mif_dom = dominant(&mif_col);
        let tgt_dom = dominant(&tgt_col);
        if mif_dom != tgt_dom {
            return Err(format!(
                "MIF axis {} dominates world component {} but target axis {} dominates {} \
                 — axis-permuted reorientation not supported by this loader",
                axis, mif_dom, axis, tgt_dom
            ));
        }
        let mif_sign = mif_col[mif_dom].signum();
        let tgt_sign = tgt_col[tgt_dom].signum();
        flips[axis] = mif_sign * tgt_sign < 0.0;
    }
    Ok(flips)
}

fn dominant(col: &[f64; 3]) -> usize {
    let mut best = 0;
    let mut best_abs = col[0].abs();
    for i in 1..3 {
        if col[i].abs() > best_abs {
            best_abs = col[i].abs();
            best = i;
        }
    }
    best
}

fn apply_flips(mut arr: Array4<f32>, flips: [bool; 3]) -> Array4<f32> {
    for (axis, &flip) in flips.iter().enumerate() {
        if flip {
            arr.invert_axis(ndarray::Axis(axis));
        }
    }
    arr
}

fn write_4d_nifti(out: &Path, reference: &Path, data: &Array4<f32>) {
    let obj = ReaderOptions::new()
        .read_file(reference)
        .expect("read reference header");
    let mut header = obj.header().clone();
    let s = data.shape();
    header.dim[0] = 4;
    header.dim[1] = s[0] as u16;
    header.dim[2] = s[1] as u16;
    header.dim[3] = s[2] as u16;
    header.dim[4] = s[3] as u16;
    for i in 5..8 {
        header.dim[i] = 1;
    }
    header.datatype = 16;
    header.bitpix = 32;
    header.scl_slope = 0.0;
    header.scl_inter = 0.0;
    if let Some(p) = out.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    if out.exists() {
        let _ = std::fs::remove_file(out);
    }
    WriterOptions::new(out)
        .reference_header(&header)
        .write_nifti(data)
        .expect("write 4D NIfTI");
}

fn write_3d_nifti_f32(out: &Path, reference: &Path, data: &Array3<f32>) {
    let obj = ReaderOptions::new()
        .read_file(reference)
        .expect("read reference header");
    let mut header = obj.header().clone();
    let s = data.shape();
    header.dim[0] = 3;
    header.dim[1] = s[0] as u16;
    header.dim[2] = s[1] as u16;
    header.dim[3] = s[2] as u16;
    for i in 4..8 {
        header.dim[i] = 1;
    }
    header.datatype = 16;
    header.bitpix = 32;
    header.scl_slope = 0.0;
    header.scl_inter = 0.0;
    if let Some(p) = out.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    if out.exists() {
        let _ = std::fs::remove_file(out);
    }
    WriterOptions::new(out)
        .reference_header(&header)
        .write_nifti(data)
        .expect("write 3D NIfTI");
}

