// SPDX-License-Identifier: MIT OR Apache-2.0
pub mod aux;
pub mod coeffs;
pub mod dwi;
pub mod microstructure;
pub mod odx_out;
pub mod provenance;

use std::fs;
use std::path::{Path, PathBuf};

use crate::{CsDmriError, Result};

/// Write a file via a sibling temp file + atomic rename.
///
/// - Refuses if `final_path` exists and `!overwrite` (returns `CsDmriError::Other`).
/// - Creates `final_path`'s parent dir(s) if missing.
/// - Calls `write` with a sibling temp path (`<final>.tmp.<pid>.<n>`); on
///   success renames into place. On failure removes the temp.
///
/// `rename` within the same directory is atomic on POSIX: a SIGKILL'd job
/// either leaves the temp behind (visible by the `.tmp.` suffix) or the final
/// file fully written, never a partial final file.
pub fn atomic_write<F>(final_path: &Path, overwrite: bool, write: F) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    if final_path.exists() && !overwrite {
        return Err(CsDmriError::Other(format!(
            "refusing to overwrite existing {} — pass --overwrite to replace",
            final_path.display()
        )));
    }
    if let Some(parent) = final_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let tmp = temp_sibling(final_path);
    let res = write(&tmp);
    if res.is_err() {
        let _ = fs::remove_file(&tmp);
        return res;
    }
    fs::rename(&tmp, final_path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        CsDmriError::Io(e)
    })?;
    Ok(())
}

/// Write a coupled pair of files atomically, *as a pair*.
///
/// Both writers run against sibling temps; only after both succeed are the
/// renames performed. Per-file rename is atomic; the gap between the two
/// renames is small but not zero — a hard kill in that window can leave the
/// first file present and the second missing. In practice this is good enough
/// for NIfTI + JSON sidecar pairs (the JSON sidecar is rewritten on next run
/// anyway).
pub fn atomic_write_pair<F1, F2>(
    final_a: &Path,
    final_b: &Path,
    overwrite: bool,
    write_a: F1,
    write_b: F2,
) -> Result<()>
where
    F1: FnOnce(&Path) -> Result<()>,
    F2: FnOnce(&Path) -> Result<()>,
{
    for p in [final_a, final_b] {
        if p.exists() && !overwrite {
            return Err(CsDmriError::Other(format!(
                "refusing to overwrite existing {} — pass --overwrite to replace",
                p.display()
            )));
        }
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
    }
    let tmp_a = temp_sibling(final_a);
    let tmp_b = temp_sibling(final_b);
    let cleanup = || {
        let _ = fs::remove_file(&tmp_a);
        let _ = fs::remove_file(&tmp_b);
    };
    if let Err(e) = write_a(&tmp_a) {
        cleanup();
        return Err(e);
    }
    if let Err(e) = write_b(&tmp_b) {
        cleanup();
        return Err(e);
    }
    if let Err(e) = fs::rename(&tmp_a, final_a) {
        cleanup();
        return Err(CsDmriError::Io(e));
    }
    if let Err(e) = fs::rename(&tmp_b, final_b) {
        let _ = fs::remove_file(&tmp_b);
        return Err(CsDmriError::Io(e));
    }
    Ok(())
}

/// Atomic-rename a directory tree into place.
///
/// Caller passes a `build` closure that populates a sibling temp directory;
/// on success it gets renamed onto `final_path`. If `final_path` is an
/// existing directory and `overwrite=true`, it's removed first (rename onto
/// an existing dir is not portable).
pub fn atomic_write_directory<F>(final_path: &Path, overwrite: bool, build: F) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    if final_path.exists() && !overwrite {
        return Err(CsDmriError::Other(format!(
            "refusing to overwrite existing {} — pass --overwrite to replace",
            final_path.display()
        )));
    }
    if let Some(parent) = final_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let tmp = temp_sibling(final_path);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp)?;
    if let Err(e) = build(&tmp) {
        let _ = fs::remove_dir_all(&tmp);
        return Err(e);
    }
    if final_path.exists() {
        if final_path.is_dir() {
            fs::remove_dir_all(final_path)?;
        } else {
            fs::remove_file(final_path)?;
        }
    }
    fs::rename(&tmp, final_path).map_err(|e| {
        let _ = fs::remove_dir_all(&tmp);
        CsDmriError::Io(e)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn temp_sibling_preserves_compound_extension() {
        // Writers like the nifti crate infer format from the extension
        // (.nii.gz vs .nii vs .hdr), so the temp path must keep it.
        let t = temp_sibling(Path::new("/tmp/foo.nii.gz"));
        let name = t.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.ends_with(".nii.gz"), "got {name}");
        assert!(name.starts_with(".foo.tmp."), "got {name}");

        let t = temp_sibling(Path::new("/tmp/bar.json"));
        let name = t.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.ends_with(".json"), "got {name}");

        let t = temp_sibling(Path::new("/tmp/no_ext"));
        let name = t.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(".no_ext.tmp."), "got {name}");
    }

    #[test]
    fn atomic_write_creates_parent_and_renames() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("nested/sub/out.bin");
        atomic_write(&target, false, |tmp| {
            let mut f = std::fs::File::create(tmp)?;
            f.write_all(b"hello")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"hello");
        assert!(!target.with_extension("tmp").exists());
    }

    #[test]
    fn atomic_write_refuses_overwrite_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.bin");
        std::fs::write(&target, b"existing").unwrap();
        let err = atomic_write(&target, false, |tmp| Ok(std::fs::write(tmp, b"new")?))
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("refusing to overwrite"), "got: {msg}");
        // original untouched
        assert_eq!(std::fs::read(&target).unwrap(), b"existing");
    }

    #[test]
    fn atomic_write_overwrite_replaces_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.bin");
        std::fs::write(&target, b"old").unwrap();
        atomic_write(&target, true, |tmp| Ok(std::fs::write(tmp, b"new")?)).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new");
    }

    #[test]
    fn atomic_write_cleans_up_temp_on_writer_error() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.bin");
        let err = atomic_write(&target, false, |_tmp| {
            Err(CsDmriError::Other("boom".into()))
        })
        .unwrap_err();
        assert!(format!("{err}").contains("boom"));
        assert!(!target.exists());
        let stragglers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(stragglers.is_empty(), "left {} temp files behind", stragglers.len());
    }

    #[test]
    fn atomic_write_pair_lands_both_files_or_neither() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.bin");
        let b = dir.path().join("b.json");
        atomic_write_pair(
            &a,
            &b,
            false,
            |tmp| Ok(std::fs::write(tmp, b"a-data")?),
            |tmp| Ok(std::fs::write(tmp, b"b-data")?),
        )
        .unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), b"a-data");
        assert_eq!(std::fs::read(&b).unwrap(), b"b-data");

        // Second writer fails → no temps left, originals untouched.
        let err = atomic_write_pair(
            &a,
            &b,
            true,
            |tmp| Ok(std::fs::write(tmp, b"new-a")?),
            |_tmp| Err(CsDmriError::Other("boom".into())),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("boom"));
        assert_eq!(std::fs::read(&a).unwrap(), b"a-data");
        assert_eq!(std::fs::read(&b).unwrap(), b"b-data");
    }
}

fn temp_sibling(final_path: &Path) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let file_name = final_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());
    // Preserve the file's (possibly compound) extension on the temp name so
    // downstream writers that infer format from the extension — notably the
    // `nifti` crate's WriterOptions, which checks for `.gz`/`.nii`/`.hdr` —
    // pick the same code path on the temp file as on the final file.
    let (stem, ext) = match file_name.find('.') {
        Some(i) if i > 0 => (&file_name[..i], &file_name[i..]),
        _ => (file_name.as_str(), ""),
    };
    let tmp_name = format!(".{stem}.tmp.{pid}.{nanos}{ext}");
    final_path
        .parent()
        .map(|p| p.join(&tmp_name))
        .unwrap_or_else(|| PathBuf::from(&tmp_name))
}
