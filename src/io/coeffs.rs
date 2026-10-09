// SPDX-License-Identifier: MIT OR Apache-2.0
//! Coefficient NIfTI + JSON sidecar I/O.

use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array4;
use nifti::writer::WriterOptions;
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};
use serde::{Deserialize, Serialize};

use crate::basis::BasisMetadata;
use crate::io::atomic_write_pair;
use crate::io::provenance::Provenance;
use crate::qspace::{BvecFrame, DeltaSource};
use crate::solver::AlphaStrategy;
use crate::{CsDmriError, Result};

fn default_bvec_frame() -> BvecFrame {
    // Older sidecars predate the bvec-frame field; they were written when
    // cs-fit kept bvecs in their FSL/image-axis frame. Default reads to that
    // for backward compatibility.
    BvecFrame::ImageAxis
}

/// JSON sidecar mirroring qsirecon's existing fit metadata where applicable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarMetadata {
    pub basis: BasisMetadata,
    pub big_delta_seconds: f64,
    pub small_delta_seconds: f64,
    pub tau_seconds: f64,
    pub delta_source: DeltaSource,
    pub gmax_tesla_per_meter: Option<f64>,
    pub solver: SolverMetadata,
    pub n_coefficients: usize,
    /// Frame the bvecs (and therefore the fitted SH coefficients) live in.
    /// Older sidecars predate this field and are assumed to be image-axis.
    #[serde(default = "default_bvec_frame")]
    pub bvec_frame: BvecFrame,
    /// Reproducibility provenance. Optional — older sidecars predate this
    /// field, and `--provenance none` skips it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum SolverMetadata {
    #[serde(rename = "tikhonov")]
    Tikhonov {
        lambda_n: f64,
        lambda_l: f64,
    },
    #[serde(rename = "fista")]
    Fista {
        alpha_strategy: AlphaStrategy,
        chosen_alpha: ChosenAlpha,
        non_negative: bool,
        max_iter: u32,
        tol: f64,
    },
    /// SHORE-basis fit with hard non-negativity on the *projected ODF
    /// amplitudes* via Goldfarb-Idnani ICLS. Engaged by `cs-fit
    /// --non-negative-amplitudes`. Distinct from `Fista { non_negative: true }`,
    /// which constrains the raw SHORE coefficients (rarely the right thing).
    #[serde(rename = "shore_icls")]
    ShoreIcls {
        lmax: u32,
        n_constraint_dirs: usize,
        max_iter: usize,
        tol: f64,
        epsilon: f64,
    },
}

/// Summary of the α actually used. `Global` is one value applied to every
/// voxel (Fixed strategy, or future global selectors). `PerVoxel` reports the
/// distribution of voxel-wise αs without bloating the sidecar with a 3-D map
/// (the map itself goes into a separate NIfTI when `--diagnostics` is on).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "kebab-case")]
pub enum ChosenAlpha {
    Global { alpha: f64 },
    PerVoxel { median: f64, p10: f64, p90: f64 },
}

/// Bundle of (4D coefficients NIfTI, JSON sidecar).
pub struct CoefficientsFile {
    pub coeffs: Array4<f32>,
    pub metadata: SidecarMetadata,
}

impl CoefficientsFile {
    /// Write the coefficient NIfTI + JSON sidecar atomically, as a pair.
    ///
    /// Both files land via temp + rename: a SIGKILL'd job leaves either both
    /// final files or neither, modulo a small window between the two renames.
    /// Refuses to clobber existing files unless `overwrite` is true.
    pub fn write(
        &self,
        out_nifti: &Path,
        reference_nifti: &Path,
        overwrite: bool,
    ) -> Result<()> {
        // Read the reference header so we inherit voxel sizes / affine.
        // Done up front so the temp writers below are infallible w.r.t. the
        // reference NIfTI.
        // Header only; the reference is usually the full 4D DWI.
        let mut header =
            nifti::NiftiHeader::from_file(reference_nifti).map_err(CsDmriError::from)?;
        let s = self.coeffs.shape();
        header.dim[0] = 4;
        header.dim[1] = s[0] as u16;
        header.dim[2] = s[1] as u16;
        header.dim[3] = s[2] as u16;
        header.dim[4] = s[3] as u16;
        for i in 5..8 {
            header.dim[i] = 1;
        }
        header.datatype = 16; // NIFTI_TYPE_FLOAT32
        header.bitpix = 32;
        header.scl_slope = 0.0;
        header.scl_inter = 0.0;

        let json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| CsDmriError::Other(format!("sidecar serialize: {e}")))?;

        let sidecar_path = sidecar_path_for(out_nifti);
        atomic_write_pair(
            out_nifti,
            &sidecar_path,
            overwrite,
            |tmp_nii| {
                WriterOptions::new(tmp_nii)
                    .reference_header(&header)
                    .write_nifti(&self.coeffs)
                    .map_err(CsDmriError::from)
            },
            |tmp_json| Ok(fs::write(tmp_json, &json)?),
        )
    }

    pub fn read(coeffs_nifti: &Path) -> Result<Self> {
        let obj = ReaderOptions::new()
            .read_file(coeffs_nifti)
            .map_err(CsDmriError::from)?;
        let nd = obj.into_volume().into_ndarray::<f32>()?;
        let coeffs = nd
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| CsDmriError::Dimension(format!("coeffs not 4D: {e}")))?;

        let sidecar_path = sidecar_path_for(coeffs_nifti);
        let metadata: SidecarMetadata = serde_json::from_str(&fs::read_to_string(&sidecar_path)?)
            .map_err(|e| CsDmriError::Other(format!("sidecar parse: {e}")))?;
        Ok(Self { coeffs, metadata })
    }
}

fn sidecar_path_for(nifti_path: &Path) -> PathBuf {
    // foo.nii.gz -> foo.json ; foo.nii -> foo.json ; foo -> foo.json
    let s = nifti_path.to_string_lossy();
    let stem = if let Some(stripped) = s.strip_suffix(".nii.gz") {
        stripped
    } else if let Some(stripped) = s.strip_suffix(".nii") {
        stripped
    } else {
        s.as_ref()
    };
    PathBuf::from(format!("{}.json", stem))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_path_strips_double_extension() {
        let p = sidecar_path_for(Path::new("/tmp/foo.nii.gz"));
        assert_eq!(p, PathBuf::from("/tmp/foo.json"));
        let p = sidecar_path_for(Path::new("/tmp/foo.nii"));
        assert_eq!(p, PathBuf::from("/tmp/foo.json"));
        let p = sidecar_path_for(Path::new("/tmp/bar"));
        assert_eq!(p, PathBuf::from("/tmp/bar.json"));
    }
}
