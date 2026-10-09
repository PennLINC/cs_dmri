// SPDX-License-Identifier: MIT OR Apache-2.0
//! Auxiliary NIfTI writers and path helpers shared by the binaries.
//!
//! Each `cs-*` CLI emits a primary output plus a handful of "aux" NIfTIs —
//! per-voxel R², chosen-α, residual, etc. They all want the same recipe:
//! inherit the input DWI's spatial header, override `dim` / `datatype` /
//! `bitpix` for the new array, and land via `atomic_write`. This module
//! is the one place that recipe lives.
//!
//! Conventions:
//! - Single-volume NIfTIs are written as 3-D (dim[0] = 3), not 4-D-with-singleton.
//! - 4-D outputs flatten the 4th dimension into `dim[4]` (with dim[5..]=1).
//! - Sign-or-precision-mismatched dtypes (u32, u64) are cast to f32 before write,
//!   since most downstream viewers handle f32 cleanly and don't agree on
//!   integer NIfTI types.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use ndarray::{Array3, Array4};
use nifti::writer::WriterOptions;
use nifti::NiftiHeader;

use crate::io::atomic_write;
use crate::{CsDmriError, Result};

/// If `path` does not already end in `.nii` or `.nii.gz`, warn on stderr (with
/// `label` for context — e.g. `"--output"`) and rewrite it in place by
/// appending `.nii.gz`. Returns whether the path was modified.
///
/// The downstream NIfTI writer infers the on-disk format from the extension;
/// passing an extensionless prefix would otherwise blow up at the very end of
/// a multi-hour fit. Doing this rewrite up front means every sidecar / sibling
/// derivation downstream sees the corrected path.
pub fn ensure_nifti_extension(path: &mut PathBuf, label: &str) -> bool {
    let s = path.to_string_lossy();
    if s.ends_with(".nii.gz") || s.ends_with(".nii") {
        return false;
    }
    let new_path = PathBuf::from(format!("{}.nii.gz", s));
    eprintln!(
        "[cs_dmri] warning: {label} {} has no .nii.gz/.nii extension; appending .nii.gz → {}",
        path.display(),
        new_path.display()
    );
    *path = new_path;
    true
}

/// Pre-flight check that `path` will be writable when we eventually try to
/// land output there. Refuses immediately (in milliseconds, not after the
/// hour-long fit) if:
///
/// - `path` already exists and `overwrite` is false — same wording as the
///   existing `atomic_write` guard.
/// - `path`'s parent directory is missing AND can't be created.
/// - `path`'s parent directory rejects a write probe (read-only mount,
///   permission denied, etc.).
///
/// The probe is a short-lived sibling file (`.cs_dmri-precheck.<pid>.<nanos>`)
/// created and deleted before we return.
pub fn precheck_writable(path: &Path, overwrite: bool) -> Result<()> {
    if path.exists() && !overwrite {
        return Err(CsDmriError::Other(format!(
            "refusing to overwrite existing {} — pass --overwrite to replace",
            path.display()
        )));
    }
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    let probe_dir: PathBuf = match parent {
        Some(p) => {
            if !p.exists() {
                fs::create_dir_all(p).map_err(|e| {
                    CsDmriError::Other(format!(
                        "cannot create parent directory {} for output {}: {e}",
                        p.display(),
                        path.display()
                    ))
                })?;
            }
            p.to_path_buf()
        }
        None => PathBuf::from("."),
    };
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let probe = probe_dir.join(format!(".cs_dmri-precheck.{pid}.{nanos}"));
    let mut f = fs::File::create(&probe).map_err(|e| {
        CsDmriError::Other(format!(
            "output directory {} is not writable (probe failed): {e}",
            probe_dir.display()
        ))
    })?;
    if let Err(e) = f.write_all(b"x") {
        let _ = fs::remove_file(&probe);
        return Err(CsDmriError::Other(format!(
            "output directory {} is not writable (probe write failed): {e}",
            probe_dir.display()
        )));
    }
    drop(f);
    let _ = fs::remove_file(&probe);
    Ok(())
}

/// Build `<stem><suffix>` next to a `.nii` / `.nii.gz` (or extensionless)
/// path. Used to land sibling diagnostics ("`brain_l1.nii.gz`" + "`_r2.nii.gz`"
/// → "`brain_l1_r2.nii.gz`").
pub fn sibling_path(reference: &Path, suffix: &str) -> PathBuf {
    let s = reference.to_string_lossy();
    let stem = if let Some(stripped) = s.strip_suffix(".nii.gz") {
        stripped
    } else if let Some(stripped) = s.strip_suffix(".nii") {
        stripped
    } else {
        s.as_ref()
    };
    PathBuf::from(format!("{}{}", stem, suffix))
}

/// Write a 3-D `f32` NIfTI inheriting the spatial header (affine, voxel sizes,
/// units) from `reference`. Atomic: writes via temp + rename, refuses to
/// clobber an existing file unless `overwrite` is true.
pub fn write_3d_f32(
    out: &Path,
    reference: &Path,
    data: &Array3<f32>,
    overwrite: bool,
) -> Result<()> {
    let header = build_3d_header(reference, data.shape(), 16, 32)?;
    atomic_write(out, overwrite, |tmp| {
        WriterOptions::new(tmp)
            .reference_header(&header)
            .write_nifti(data)
            .map_err(|e| CsDmriError::Other(format!("write {tmp:?}: {e}")))
    })
}

/// Write a 3-D `f32` NIfTI whose spatial header is built from `affine`
/// directly (sform-only, RAS+). Use this for outputs whose voxel grid was
/// reoriented relative to any on-disk reference file — the affine that
/// describes the data lives in memory, not in another NIfTI.
pub fn write_3d_f32_with_affine(
    out: &Path,
    affine: &[[f64; 4]; 4],
    data: &Array3<f32>,
    overwrite: bool,
) -> Result<()> {
    let s = data.shape();
    let header = build_header_from_affine(affine, [s[0], s[1], s[2]], 16, 32);
    atomic_write(out, overwrite, |tmp| {
        WriterOptions::new(tmp)
            .reference_header(&header)
            .write_nifti(data)
            .map_err(|e| CsDmriError::Other(format!("write {tmp:?}: {e}")))
    })
}

/// Write a 3-D `u8` NIfTI (e.g. boolean mask, small categorical map).
pub fn write_3d_u8(
    out: &Path,
    reference: &Path,
    data: &Array3<u8>,
    overwrite: bool,
) -> Result<()> {
    let header = build_3d_header(reference, data.shape(), 2, 8)?;
    atomic_write(out, overwrite, |tmp| {
        WriterOptions::new(tmp)
            .reference_header(&header)
            .write_nifti(data)
            .map_err(|e| CsDmriError::Other(format!("write {tmp:?}: {e}")))
    })
}

/// Write a 3-D `u32` NIfTI by casting to `f32` first. Most viewers don't
/// handle UINT32 cleanly; emitting f32 keeps the file portable and the
/// loss-of-precision is usually irrelevant for the iteration-count maps
/// these get used for.
pub fn write_3d_u32_as_f32(
    out: &Path,
    reference: &Path,
    data: &Array3<u32>,
    overwrite: bool,
) -> Result<()> {
    let f = data.mapv(|v| v as f32);
    write_3d_f32(out, reference, &f, overwrite)
}

/// Write a 4-D `f32` NIfTI inheriting the spatial header from `reference`.
/// `dim[4]` is set to the array's 4th-axis length.
pub fn write_4d_f32(
    out: &Path,
    reference: &Path,
    data: &Array4<f32>,
    overwrite: bool,
) -> Result<()> {
    let s = data.shape();
    let mut header = read_reference_header(reference)?;
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
    atomic_write(out, overwrite, |tmp| {
        WriterOptions::new(tmp)
            .reference_header(&header)
            .write_nifti(data)
            .map_err(|e| CsDmriError::Other(format!("write {tmp:?}: {e}")))
    })
}

// --- internals ---

/// Header only: `ReaderOptions::read_file` would load the whole reference,
/// often a 4D DWI, once per output written.
fn read_reference_header(reference: &Path) -> Result<nifti::NiftiHeader> {
    nifti::NiftiHeader::from_file(reference).map_err(CsDmriError::from)
}

/// Convert an affine to NIfTI qform parameters: quaternion `(b, c, d)`,
/// translation `(qx, qy, qz)`, voxel sizes `(dx, dy, dz)`, and `qfac` (±1).
/// Mirrors `nifti_mat44_to_quatern` from nifti1.h.
fn affine_to_qform(affine: &[[f64; 4]; 4]) -> ([f32; 3], [f32; 3], [f32; 3], f32) {
    let qx = affine[0][3];
    let qy = affine[1][3];
    let qz = affine[2][3];

    let mut xd = (affine[0][0].powi(2) + affine[1][0].powi(2) + affine[2][0].powi(2)).sqrt();
    let mut yd = (affine[0][1].powi(2) + affine[1][1].powi(2) + affine[2][1].powi(2)).sqrt();
    let mut zd = (affine[0][2].powi(2) + affine[1][2].powi(2) + affine[2][2].powi(2)).sqrt();
    if xd == 0.0 {
        xd = 1.0;
    }
    if yd == 0.0 {
        yd = 1.0;
    }
    if zd == 0.0 {
        zd = 1.0;
    }

    let r11 = affine[0][0] / xd;
    let r21 = affine[1][0] / xd;
    let r31 = affine[2][0] / xd;
    let r12 = affine[0][1] / yd;
    let r22 = affine[1][1] / yd;
    let r32 = affine[2][1] / yd;
    let mut r13 = affine[0][2] / zd;
    let mut r23 = affine[1][2] / zd;
    let mut r33 = affine[2][2] / zd;

    // If the rotation is left-handed, flip the Z column and record qfac=-1.
    let det = r11 * (r22 * r33 - r32 * r23) - r12 * (r21 * r33 - r31 * r23)
        + r13 * (r21 * r32 - r31 * r22);
    let qfac = if det < 0.0 {
        r13 = -r13;
        r23 = -r23;
        r33 = -r33;
        -1.0
    } else {
        1.0
    };

    // Re-orthogonalize via the SVD trick is overkill here; the trace formula
    // is what nifti1.h uses and matches nibabel's output bit-for-bit on
    // well-formed affines.
    let trace = r11 + r22 + r33 + 1.0;
    let (a, b, c, d) = if trace > 0.5 {
        let a = 0.5 * trace.sqrt();
        let inv4a = 0.25 / a;
        (
            a,
            (r32 - r23) * inv4a,
            (r13 - r31) * inv4a,
            (r21 - r12) * inv4a,
        )
    } else {
        let xd = 1.0 + r11 - (r22 + r33);
        let yd = 1.0 + r22 - (r11 + r33);
        let zd = 1.0 + r33 - (r11 + r22);
        let (mut a, mut b, mut c, mut d);
        if xd >= yd && xd >= zd {
            b = 0.5 * xd.sqrt();
            let inv4 = 0.25 / b;
            c = (r12 + r21) * inv4;
            d = (r13 + r31) * inv4;
            a = (r32 - r23) * inv4;
        } else if yd >= zd {
            c = 0.5 * yd.sqrt();
            let inv4 = 0.25 / c;
            b = (r12 + r21) * inv4;
            d = (r23 + r32) * inv4;
            a = (r13 - r31) * inv4;
        } else {
            d = 0.5 * zd.sqrt();
            let inv4 = 0.25 / d;
            b = (r13 + r31) * inv4;
            c = (r23 + r32) * inv4;
            a = (r21 - r12) * inv4;
        }
        if a < 0.0 {
            a = -a;
            b = -b;
            c = -c;
            d = -d;
        }
        (a, b, c, d)
    };
    let _ = a;

    (
        [b as f32, c as f32, d as f32],
        [qx as f32, qy as f32, qz as f32],
        [xd as f32, yd as f32, zd as f32],
        qfac as f32,
    )
}

fn build_header_from_affine(
    affine: &[[f64; 4]; 4],
    shape: [usize; 3],
    datatype: i16,
    bitpix: i16,
) -> NiftiHeader {
    let (quatern, qoffset, voxel_sizes, qfac) = affine_to_qform(affine);
    let mut header = NiftiHeader::default();
    header.dim[0] = 3;
    header.dim[1] = shape[0] as u16;
    header.dim[2] = shape[1] as u16;
    header.dim[3] = shape[2] as u16;
    for i in 4..8 {
        header.dim[i] = 1;
    }
    header.datatype = datatype;
    header.bitpix = bitpix;
    header.scl_slope = 0.0;
    header.scl_inter = 0.0;
    header.qform_code = 1;
    header.sform_code = 0;
    header.pixdim[0] = qfac;
    header.pixdim[1] = voxel_sizes[0];
    header.pixdim[2] = voxel_sizes[1];
    header.pixdim[3] = voxel_sizes[2];
    header.xyzt_units = 2;
    header.quatern_b = quatern[0];
    header.quatern_c = quatern[1];
    header.quatern_d = quatern[2];
    header.quatern_x = qoffset[0];
    header.quatern_y = qoffset[1];
    header.quatern_z = qoffset[2];
    header
}

fn build_3d_header(
    reference: &Path,
    shape: &[usize],
    datatype: i16,
    bitpix: i16,
) -> Result<nifti::NiftiHeader> {
    let mut header = read_reference_header(reference)?;
    header.dim[0] = 3;
    header.dim[1] = shape[0] as u16;
    header.dim[2] = shape[1] as u16;
    header.dim[3] = shape[2] as u16;
    for i in 4..8 {
        header.dim[i] = 1;
    }
    header.datatype = datatype;
    header.bitpix = bitpix;
    header.scl_slope = 0.0;
    header.scl_inter = 0.0;
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qform_to_affine(quat: [f32; 3], qoff: [f32; 3], pix: [f32; 3], qfac: f32) -> [[f64; 4]; 4] {
        let b = quat[0] as f64;
        let c = quat[1] as f64;
        let d = quat[2] as f64;
        let a_sq = 1.0_f64 - (b * b + c * c + d * d);
        let a = if a_sq < 1.0e-7 { 0.0 } else { a_sq.sqrt() };
        let dx = pix[0] as f64;
        let dy = pix[1] as f64;
        let dz = (qfac as f64) * pix[2] as f64;
        [
            [
                (a * a + b * b - c * c - d * d) * dx,
                (2.0 * b * c - 2.0 * a * d) * dy,
                (2.0 * b * d + 2.0 * a * c) * dz,
                qoff[0] as f64,
            ],
            [
                (2.0 * b * c + 2.0 * a * d) * dx,
                (a * a + c * c - b * b - d * d) * dy,
                (2.0 * c * d - 2.0 * a * b) * dz,
                qoff[1] as f64,
            ],
            [
                (2.0 * b * d - 2.0 * a * c) * dx,
                (2.0 * c * d + 2.0 * a * b) * dy,
                (a * a + d * d - b * b - c * c) * dz,
                qoff[2] as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn assert_affine_close(got: &[[f64; 4]; 4], want: &[[f64; 4]; 4], tol: f64) {
        for r in 0..4 {
            for c in 0..4 {
                assert!(
                    (got[r][c] - want[r][c]).abs() < tol,
                    "[{r}][{c}]: got {} want {}",
                    got[r][c],
                    want[r][c]
                );
            }
        }
    }

    #[test]
    fn qform_roundtrip_ras_identity() {
        let aff = [
            [2.0, 0.0, 0.0, -90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let (q, off, pix, qfac) = affine_to_qform(&aff);
        assert_affine_close(&qform_to_affine(q, off, pix, qfac), &aff, 1e-5);
    }

    #[test]
    fn qform_roundtrip_las_left_handed() {
        // LAS+: x flipped relative to RAS+, qfac should go to -1.
        let aff = [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let (q, off, pix, qfac) = affine_to_qform(&aff);
        assert_eq!(qfac, -1.0);
        assert_affine_close(&qform_to_affine(q, off, pix, qfac), &aff, 1e-5);
    }

    #[test]
    fn ensure_nifti_extension_keeps_correct_extensions() {
        let mut p = PathBuf::from("/tmp/foo.nii.gz");
        assert!(!ensure_nifti_extension(&mut p, "--output"));
        assert_eq!(p, PathBuf::from("/tmp/foo.nii.gz"));

        let mut p = PathBuf::from("/tmp/foo.nii");
        assert!(!ensure_nifti_extension(&mut p, "--output"));
        assert_eq!(p, PathBuf::from("/tmp/foo.nii"));
    }

    #[test]
    fn ensure_nifti_extension_appends_when_missing() {
        let mut p = PathBuf::from("/tmp/foo");
        assert!(ensure_nifti_extension(&mut p, "--output"));
        assert_eq!(p, PathBuf::from("/tmp/foo.nii.gz"));

        // Bare extensionless name doesn't get `.nii.gz` appended onto an
        // already-non-nifti suffix — `foo.tar.gz` would be silently
        // mis-extended otherwise. (Spec: only no-NIfTI-extension paths get the
        // append; everything else is the user's problem and surfaces at write
        // time. .tar.gz isn't .nii.gz/.nii, so it gets appended too — that's
        // intentionally simple and matches the warning message.)
        let mut p = PathBuf::from("foo.tar.gz");
        assert!(ensure_nifti_extension(&mut p, "--output"));
        assert_eq!(p, PathBuf::from("foo.tar.gz.nii.gz"));
    }

    #[test]
    fn precheck_writable_creates_missing_parent() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("nested/sub/out.nii.gz");
        precheck_writable(&target, false).unwrap();
        assert!(target.parent().unwrap().is_dir());
        // probe was cleaned up.
        let probes: Vec<_> = std::fs::read_dir(target.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("cs_dmri-precheck"))
            .collect();
        assert!(probes.is_empty(), "probe leaked: {} files", probes.len());
    }

    #[test]
    fn precheck_writable_refuses_existing_without_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.nii.gz");
        std::fs::write(&target, b"existing").unwrap();
        let err = precheck_writable(&target, false).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("refusing to overwrite"), "got: {msg}");
        // Original is untouched.
        assert_eq!(std::fs::read(&target).unwrap(), b"existing");
    }

    #[test]
    fn precheck_writable_accepts_existing_with_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.nii.gz");
        std::fs::write(&target, b"existing").unwrap();
        precheck_writable(&target, true).unwrap();
        // File still untouched (precheck doesn't write the final).
        assert_eq!(std::fs::read(&target).unwrap(), b"existing");
    }

    #[test]
    fn qform_roundtrip_180deg_z_rotation() {
        // 180° about Z: a=0, branches into the xd>=yd>=zd path.
        let aff = [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, -2.0, 0.0, 126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let (q, off, pix, qfac) = affine_to_qform(&aff);
        assert_eq!(qfac, 1.0);
        assert_affine_close(&qform_to_affine(q, off, pix, qfac), &aff, 1e-5);
    }
}
