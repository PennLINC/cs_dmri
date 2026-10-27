// SPDX-License-Identifier: MIT OR Apache-2.0
//! 4D DWI loader: NIfTI volume + FSL bval/bvec + optional mask.

use std::fs;
use std::path::Path;

use ndarray::{Array3, Array4, ArrayView4, CowArray, Ix3, Ix4, ShapeBuilder};
use nifti::{IntoNdArray, NiftiObject, NiftiVolume, ReaderOptions, ReaderStreamedOptions};

use crate::qspace::{BvecFrame, GradientTable, affine_rotation, rotate_bvecs};
use crate::{CsDmriError, Result};

use odx_rs::reference_affine::read_reference_affine;

/// Loaded DWI bundle ready to feed into a fit.
///
/// `data` is owned when the series was read from disk and may be borrowed
/// when it comes from elsewhere (the Python bindings lend numpy's buffer).
/// The fits read one voxel's signal at a time (`data.slice(s![x, y, z, ..])`),
/// so they are fastest when that axis is contiguous: standard (C) layout,
/// which [`read_dwi_volume`] produces.
pub struct DwiData<'a> {
    /// 4D signal, shape (X, Y, Z, n_grads) in f32.
    pub data: CowArray<'a, f32, Ix4>,
    /// 3D mask (X, Y, Z); true where the voxel should be fit.
    pub mask: Array3<bool>,
    pub gtab: GradientTable,
    /// Frame the bvecs (and hence the resulting SH coefficients) live in.
    pub bvec_frame: BvecFrame,
    /// The gradient directions in the image's voxel-axis frame (dipy/qsiprep
    /// convention), before any rotation into `bvec_frame`. A gradient-deviation
    /// field acts on these: its components are expressed in that same frame.
    pub bvecs_vox: Vec<[f64; 3]>,
    /// Maps `bvecs_vox` into `bvec_frame`: the column-normalized affine 3×3 for
    /// `WorldRas`, identity for `ImageAxis`. `gtab.bvecs[i] = frame_rotation ·
    /// bvecs_vox[i]`.
    pub frame_rotation: [[f64; 3]; 3],
    /// Header path so the writer can copy spatial metadata.
    pub source_nifti: std::path::PathBuf,
}

/// The 3×3 identity, i.e. [`DwiData::frame_rotation`] for `ImageAxis`.
pub const IDENTITY3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl DwiData<'static> {
    /// Bundle an in-memory volume whose `gtab.bvecs` are taken as the voxel-axis
    /// directions (identity `frame_rotation`), for tests and synthetic data.
    pub fn from_table(
        data: Array4<f32>,
        mask: Array3<bool>,
        gtab: GradientTable,
        bvec_frame: BvecFrame,
        source_nifti: std::path::PathBuf,
    ) -> Self {
        let bvecs_vox = gtab.bvecs.clone();
        Self {
            data: data.into(),
            mask,
            gtab,
            bvec_frame,
            bvecs_vox,
            frame_rotation: IDENTITY3,
            source_nifti,
        }
    }
}

impl<'a> DwiData<'a> {
    /// Like [`DwiData::from_table`], but borrowing the signal. A view that is
    /// not in standard layout is copied once into it, so every fit reads
    /// contiguous voxel signals; a standard-layout view is used as is.
    pub fn from_view(
        data: ArrayView4<'a, f32>,
        mask: Array3<bool>,
        gtab: GradientTable,
        bvec_frame: BvecFrame,
        source_nifti: std::path::PathBuf,
    ) -> Self {
        let bvecs_vox = gtab.bvecs.clone();
        Self {
            data: standard_layout(data),
            mask,
            gtab,
            bvec_frame,
            bvecs_vox,
            frame_rotation: IDENTITY3,
            source_nifti,
        }
    }

    pub fn shape(&self) -> [usize; 4] {
        let s = self.data.shape();
        [s[0], s[1], s[2], s[3]]
    }
}

/// Load a 4D DWI plus its FSL bval/bvec/optional mask.
///
/// `gmax` is only used if either delta is missing — it's passed straight to
/// the TORTOISE estimator.
///
/// When `bvec_frame == WorldRas` (the default of cs-fit), bvecs are rotated
/// from FSL's image-axis frame into world-RAS using the column-normalized 3×3
/// of the NIfTI affine. This makes the SH coefficients downstream tools see
/// (e.g. TRXViz) match the world frame they assume when sampling SH on a fixed
/// unit sphere.
pub fn load_dwi(
    dwi_path: &Path,
    bval_path: &Path,
    bvec_path: &Path,
    mask_path: Option<&Path>,
    big_delta: Option<f64>,
    small_delta: Option<f64>,
    gmax: Option<f64>,
    bvec_frame: BvecFrame,
) -> Result<DwiData<'static>> {
    let bvals = parse_bvals(&fs::read_to_string(bval_path)?)?;
    let bvecs = parse_bvecs(&fs::read_to_string(bvec_path)?)?;
    if bvals.len() != bvecs.len() {
        return Err(CsDmriError::Dimension(format!(
            "bval count ({}) != bvec count ({})",
            bvals.len(),
            bvecs.len()
        )));
    }
    load_dwi_vox(dwi_path, bvals, bvecs, mask_path, big_delta, small_delta, gmax, bvec_frame)
}

/// Load a 4D DWI with an MRtrix `.b` gradient file (`x y z b` per row, the
/// direction in world RAS) instead of FSL bval/bvec. The directions are taken
/// back into the voxel-axis frame through the inverse of the affine's
/// column-normalized 3×3, then handled exactly like FSL input — so the fit sees
/// the same table either way, and the voxel-frame vectors a gradient-deviation
/// field needs are unambiguous.
pub fn load_dwi_mrtrix_grad(
    dwi_path: &Path,
    grad_path: &Path,
    mask_path: Option<&Path>,
    big_delta: Option<f64>,
    small_delta: Option<f64>,
    gmax: Option<f64>,
    bvec_frame: BvecFrame,
) -> Result<DwiData<'static>> {
    let (bvals, bvecs_world) = parse_mrtrix_grad(&fs::read_to_string(grad_path)?)?;
    let p = affine_rotation(&dwi_affine(dwi_path)?);
    let p_inv = nalgebra::Matrix3::from_fn(|i, j| p[i][j])
        .try_inverse()
        .ok_or_else(|| CsDmriError::Other(format!("affine of {dwi_path:?} is singular")))?;
    let bvecs_vox = bvecs_world
        .iter()
        .map(|g| {
            let v = p_inv * nalgebra::Vector3::new(g[0], g[1], g[2]);
            [v[0], v[1], v[2]]
        })
        .collect();
    load_dwi_vox(dwi_path, bvals, bvecs_vox, mask_path, big_delta, small_delta, gmax, bvec_frame)
}

fn dwi_affine(dwi_path: &Path) -> Result<[[f64; 4]; 4]> {
    read_reference_affine(dwi_path)
        .map_err(|e| CsDmriError::Other(format!("read affine from {dwi_path:?}: {e}")))
}

/// Shared tail of the loaders: `bvecs_vox` are voxel-axis-frame directions.
#[allow(clippy::too_many_arguments)]
fn load_dwi_vox(
    dwi_path: &Path,
    bvals: Vec<f64>,
    bvecs_vox: Vec<[f64; 3]>,
    mask_path: Option<&Path>,
    big_delta: Option<f64>,
    small_delta: Option<f64>,
    gmax: Option<f64>,
    bvec_frame: BvecFrame,
) -> Result<DwiData<'static>> {
    let data = read_dwi_volume(dwi_path)?;
    if data.shape()[3] != bvals.len() {
        return Err(CsDmriError::Dimension(format!(
            "DWI 4th dimension ({}) does not match bval count ({})",
            data.shape()[3],
            bvals.len()
        )));
    }

    let mask = match mask_path {
        Some(p) => load_mask(p, [data.shape()[0], data.shape()[1], data.shape()[2]])?,
        None => default_mask_from_b0(&data.view(), &bvals),
    };

    let frame_rotation = match bvec_frame {
        BvecFrame::WorldRas => affine_rotation(&dwi_affine(dwi_path)?),
        BvecFrame::ImageAxis => IDENTITY3,
    };
    let mut bvecs = bvecs_vox.clone();
    if matches!(bvec_frame, BvecFrame::WorldRas) {
        rotate_bvecs(&mut bvecs, &frame_rotation);
    }

    let gtab = GradientTable::new(bvals, bvecs, big_delta, small_delta, gmax)?;
    Ok(DwiData {
        data: data.into(),
        mask,
        gtab,
        bvec_frame,
        bvecs_vox,
        frame_rotation,
        source_nifti: dwi_path.to_path_buf(),
    })
}

/// Volumes read before they are written into the voxel-major array, so each
/// voxel receives a contiguous run of this many samples (one cache line) per pass.
const VOLUME_BLOCK: usize = 16;

/// Read a 4D NIfTI into an f32 array of shape (X, Y, Z, N) in standard (C)
/// layout, so each voxel's signal is contiguous.
///
/// The file stores whole volumes one after another, in any data type and with
/// any `scl_slope` / `scl_inter`. They are streamed one at a time and converted
/// to f32 as they arrive: at most one volume is ever held in the file's type (a
/// float64 series is never read whole as float64), and the series is held once,
/// as f32. Volumes are written [`VOLUME_BLOCK`] at a time, so turning
/// volume-major storage into voxel-major memory writes contiguous runs.
pub fn read_dwi_volume(path: &Path) -> Result<Array4<f32>> {
    let obj = ReaderStreamedOptions::new()
        .read_file(path)
        .map_err(CsDmriError::from)?;
    let dim = obj.volume().dim().to_vec();
    if dim.len() != 4 {
        return Err(CsDmriError::Dimension(format!(
            "expected 4D DWI, got {}D ({:?})",
            dim.len(),
            dim
        )));
    }
    let shape = (dim[0] as usize, dim[1] as usize, dim[2] as usize, dim[3] as usize);
    let mut data = Array4::<f32>::zeros(shape);
    let mut block: Vec<Array3<f32>> = Vec::with_capacity(VOLUME_BLOCK);
    let mut written = 0usize;
    for (t, volume) in obj.into_volume().enumerate() {
        if t >= shape.3 {
            return Err(CsDmriError::Dimension(format!(
                "DWI holds more volumes than its header declares ({})",
                shape.3
            )));
        }
        let volume = volume
            .map_err(CsDmriError::from)?
            .into_ndarray::<f32>()?
            .into_dimensionality::<Ix3>()
            .map_err(|e| CsDmriError::Dimension(format!("could not view DWI volume {t} as 3D: {e}")))?;
        block.push(volume);
        if block.len() == VOLUME_BLOCK {
            write_volume_block(&mut data, written, &block);
            written += block.len();
            block.clear();
        }
    }
    write_volume_block(&mut data, written, &block);
    written += block.len();
    if written != shape.3 {
        return Err(CsDmriError::Dimension(format!(
            "DWI holds {written} volumes but its header declares {}",
            shape.3
        )));
    }
    Ok(data)
}

/// Write `block` as volumes `start..start + block.len()` of the standard-layout
/// `data`. Voxels are visited in the volumes' own storage order (x fastest), so
/// the reads are sequential and each voxel's run of samples is written at once.
fn write_volume_block(data: &mut Array4<f32>, start: usize, block: &[Array3<f32>]) {
    if block.is_empty() {
        return;
    }
    let (nx, ny, nz, nt) = data.dim();
    // `into_ndarray` returns Fortran order; anything else is copied into it, so
    // that `x + nx * (y + ny * z)` indexes every volume.
    let fortran: Vec<std::borrow::Cow<'_, Array3<f32>>> = block
        .iter()
        .map(|v| {
            if v.t().is_standard_layout() {
                std::borrow::Cow::Borrowed(v)
            } else {
                let mut f = Array3::<f32>::zeros(v.raw_dim().f());
                f.assign(v);
                std::borrow::Cow::Owned(f)
            }
        })
        .collect();
    let src: Vec<&[f32]> = fortran
        .iter()
        .map(|v| v.as_slice_memory_order().expect("Fortran-order volume is contiguous"))
        .collect();
    let out = data.as_slice_mut().expect("zeros() is in standard layout");
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let s = x + nx * (y + ny * z);
                let d = ((x * ny + y) * nz + z) * nt + start;
                for (o, v) in out[d..d + src.len()].iter_mut().zip(&src) {
                    *o = v[s];
                }
            }
        }
    }
}

/// `view` in standard layout: borrowed when it already is, copied once otherwise.
pub fn standard_layout(view: ArrayView4<'_, f32>) -> CowArray<'_, f32, Ix4> {
    if view.is_standard_layout() {
        CowArray::from(view)
    } else {
        CowArray::from(view.as_standard_layout().into_owned())
    }
}

/// Load a 3D mask NIfTI, validating it against the expected volume shape.
///
/// Public so binaries can read auxiliary masks through the same path the
/// brain mask uses, rather than duplicating NIfTI handling.
pub fn load_mask(path: &Path, expected_shape: [usize; 3]) -> Result<Array3<bool>> {
    let obj = ReaderOptions::new().read_file(path).map_err(CsDmriError::from)?;
    let dim = obj.volume().dim().to_vec();
    if dim.len() != 3 {
        return Err(CsDmriError::Dimension(format!(
            "expected 3D mask, got {}D ({:?})",
            dim.len(),
            dim
        )));
    }
    let nd = obj.into_volume().into_ndarray::<f32>()?;
    let nd = nd.into_dimensionality::<ndarray::Ix3>().map_err(|e| {
        CsDmriError::Dimension(format!("could not view mask as 3D ndarray: {e}"))
    })?;
    if nd.shape() != expected_shape {
        return Err(CsDmriError::Dimension(format!(
            "mask shape {:?} does not match DWI spatial shape {:?}",
            nd.shape(),
            expected_shape
        )));
    }
    Ok(nd.mapv(|v| v > 0.0))
}

/// Quick fallback mask: any voxel whose mean b0 signal exceeds 1% of the
/// global max. Matches the spirit of qsirecon's Otsu fallback without the
/// scikit-image dependency.
pub fn default_mask_from_b0(data: &ArrayView4<f32>, bvals: &[f64]) -> Array3<bool> {
    let s = data.shape();
    let mut b0: Vec<usize> = (0..bvals.len()).filter(|&t| bvals[t] <= 50.0).collect();
    if b0.is_empty() {
        // No b0 frames — fall back to mean across all volumes.
        b0 = (0..s[3]).collect();
    }
    // One voxel's signal at a time, the contiguous axis of the loaders' layout.
    let mut b0_mean = Array3::<f32>::zeros((s[0], s[1], s[2]));
    ndarray::Zip::from(&mut b0_mean).and(data.lanes(ndarray::Axis(3))).for_each(|m, lane| {
        *m = b0.iter().map(|&t| lane[t]).sum::<f32>() / b0.len() as f32;
    });
    let max = b0_mean.iter().cloned().fold(0.0_f32, f32::max);
    let thresh = max * 0.01;
    b0_mean.mapv(|v| v > thresh)
}

pub fn parse_bvals(text: &str) -> Result<Vec<f64>> {
    text.split_whitespace()
        .map(|tok| {
            tok.parse::<f64>()
                .map_err(|e| CsDmriError::Parse(format!("bval token '{tok}': {e}")))
        })
        .collect()
}

/// MRtrix `.b` parsing: one `x y z b` row per gradient (direction in world RAS);
/// `#` comment lines and blank lines are skipped.
pub fn parse_mrtrix_grad(text: &str) -> Result<(Vec<f64>, Vec<[f64; 3]>)> {
    let mut bvals = Vec::new();
    let mut dirs = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let vals: Vec<f64> = line
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse::<f64>().map_err(|e| CsDmriError::Parse(format!(".b token '{t}': {e}"))))
            .collect::<Result<_>>()?;
        if vals.len() != 4 {
            return Err(CsDmriError::Parse(format!(
                ".b rows must have 4 values (x y z b), got {} in '{line}'",
                vals.len()
            )));
        }
        dirs.push([vals[0], vals[1], vals[2]]);
        bvals.push(vals[3]);
    }
    Ok((bvals, dirs))
}

/// FSL bvec parsing: 3 lines (or 3 columns) with one entry per gradient.
pub fn parse_bvecs(text: &str) -> Result<Vec<[f64; 3]>> {
    let lines: Vec<Vec<f64>> = text
        .lines()
        .map(|line| line.split_whitespace().map(|t| t.parse::<f64>()).collect())
        .map(|res: std::result::Result<Vec<f64>, _>| {
            res.map_err(|e| CsDmriError::Parse(format!("bvec parse: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let lines: Vec<Vec<f64>> = lines.into_iter().filter(|l| !l.is_empty()).collect();

    let bvecs = if lines.len() == 3 {
        // Row layout: 3 rows × n_grads cols
        let n = lines[0].len();
        if lines[1].len() != n || lines[2].len() != n {
            return Err(CsDmriError::Parse(
                "bvec rows have inconsistent lengths".into(),
            ));
        }
        (0..n)
            .map(|i| [lines[0][i], lines[1][i], lines[2][i]])
            .collect()
    } else if lines.iter().all(|l| l.len() == 3) {
        // Column layout: n_grads rows × 3 cols
        lines.iter().map(|l| [l[0], l[1], l[2]]).collect()
    } else {
        return Err(CsDmriError::Parse(format!(
            "could not interpret bvec file (lines: {}, widths: {:?})",
            lines.len(),
            lines.iter().map(|l| l.len()).collect::<Vec<_>>()
        )));
    };
    Ok(bvecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mrtrix_grad_rows() {
        let (b, d) = parse_mrtrix_grad("# comment\n0 0 0 0\n0.6 -0.8 0 1000\n\n0,0,1,2000\n").unwrap();
        assert_eq!(b, vec![0.0, 1000.0, 2000.0]);
        assert_eq!(d, vec![[0.0, 0.0, 0.0], [0.6, -0.8, 0.0], [0.0, 0.0, 1.0]]);
        assert!(parse_mrtrix_grad("1 2 3\n").is_err());
    }

    #[test]
    fn parse_bvals_simple() {
        let v = parse_bvals(" 0 1000 2000\n 3000 ").unwrap();
        assert_eq!(v, vec![0.0, 1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn parse_bvecs_row_layout() {
        let txt = "1 0 0\n0 1 0\n0 0 1\n";
        let v = parse_bvecs(txt).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], [1.0, 0.0, 0.0]);
        assert_eq!(v[1], [0.0, 1.0, 0.0]);
        assert_eq!(v[2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn parse_bvecs_column_layout() {
        let txt = "1 0 0\n0 1 0\n0 0 1\n0 0 -1\n";
        let v = parse_bvecs(txt).unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(v[3], [0.0, 0.0, -1.0]);
    }

    /// Every voxel of a float64 file read back at its own (x, y, z, t), in
    /// standard layout. The value encodes the indices, so a swapped axis or a
    /// misplaced block shows up; 37 volumes leave a partial final block.
    #[test]
    fn read_dwi_volume_keeps_axes_in_standard_layout() {
        let (nx, ny, nz, nt) = (5, 6, 7, 37);
        let code = |x: usize, y: usize, z: usize, t: usize| (x + 10 * y + 100 * z + 1000 * t) as f64;
        let src = Array4::from_shape_fn((nx, ny, nz, nt), |(x, y, z, t)| code(x, y, z, t));
        let dir = tempfile::tempdir().unwrap();
        for name in ["coded.nii", "coded.nii.gz"] {
            let path = dir.path().join(name);
            nifti::writer::WriterOptions::new(&path).write_nifti(&src).unwrap();
            let data = read_dwi_volume(&path).unwrap();
            assert_eq!(data.dim(), (nx, ny, nz, nt), "{name}");
            assert!(data.is_standard_layout(), "{name}");
            for ((x, y, z, t), &v) in data.indexed_iter() {
                assert_eq!(v as f64, code(x, y, z, t), "{name} at ({x}, {y}, {z}, {t})");
            }
        }
    }

    #[test]
    fn standard_layout_borrows_c_order_and_copies_fortran_order() {
        let c = Array4::from_shape_fn((2, 3, 4, 5), |(x, y, z, t)| (x + 10 * y + 100 * z + 1000 * t) as f32);
        let mut f = Array4::<f32>::zeros((2, 3, 4, 5).f());
        f.assign(&c);
        let borrowed = standard_layout(c.view());
        assert!(borrowed.is_view());
        assert_eq!(borrowed.as_ptr(), c.as_ptr());
        let copied = standard_layout(f.view());
        assert!(copied.is_owned() && copied.is_standard_layout());
        assert_eq!(copied, c);
    }

    #[test]
    fn default_mask_from_b0_uses_the_b0_volumes() {
        let mut data = Array4::<f32>::zeros((3, 1, 1, 3));
        // Voxel 0 is bright only at b=0, voxel 1 only at b=1000, voxel 2 is empty.
        data[(0, 0, 0, 0)] = 100.0;
        data[(1, 0, 0, 1)] = 100.0;
        let mask = default_mask_from_b0(&data.view(), &[0.0, 1000.0, 2000.0]);
        assert_eq!(mask.iter().copied().collect::<Vec<_>>(), vec![true, false, false]);
    }
}
