// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs_dmri._cs_dmri`: the compiled half of the `cs_dmri` Python package.
//!
//! A thin, functional layer over the `cs_dmri` crate: numpy arrays and
//! keyword arguments in, numpy arrays and plain dicts out. The classes users
//! see (`DWI`, `RestoreModel`, `QCReport`, ...) are pure Python on top of
//! these functions.
//!
//! Conventions:
//! - volumes are `float32` `(X, Y, Z, N)`, masks `bool` `(X, Y, Z)`, any
//!   memory layout (QC functions read them without copying; fits copy once);
//! - gradient tables arrive as `bvals (N,)` + `bvecs (N, 3)` float64;
//! - heavy work releases the GIL and, when `n_threads` is given, runs on a
//!   private rayon pool of that size (the global pool is never configured).

use std::fmt::Display;

use numpy::ndarray::{Array2, Array3, Array4};
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use cs_dmri::qspace::{GradientTable, TORTOISE_DEFAULT_GMAX};

pub(crate) fn map_err<E: Display>(e: E) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}

/// Run `f` with the GIL released, on a private pool of `n_threads` if given.
pub(crate) fn run<T, F>(py: Python<'_>, n_threads: Option<usize>, f: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    py.allow_threads(|| match n_threads {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n.max(1))
            .build()
            .map(|pool| pool.install(f))
            .map_err(map_err),
        None => Ok(f()),
    })
}

pub(crate) fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    use serde_json::Value;
    Ok(match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_py(py),
        Value::Number(n) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => i.into_py(py),
            (None, Some(f)) => f.into_py(py),
            _ => py.None(),
        },
        Value::String(s) => s.into_py(py),
        Value::Array(a) => {
            let list = PyList::empty_bound(py);
            for x in a {
                list.append(json_to_py(py, x)?)?;
            }
            list.into_py(py)
        }
        Value::Object(o) => {
            let d = PyDict::new_bound(py);
            for (k, x) in o {
                d.set_item(k, json_to_py(py, x)?)?;
            }
            d.into_py(py)
        }
    })
}

pub(crate) fn to_py_dict<'py, T: serde::Serialize>(py: Python<'py>, v: &T) -> PyResult<Bound<'py, PyDict>> {
    let value = serde_json::to_value(v).map_err(map_err)?;
    let obj = json_to_py(py, &value)?;
    obj.into_bound(py).downcast_into::<PyDict>().map_err(|e| map_err(format!("{e}")))
}

/// Build a gradient table. Deltas are estimated (TORTOISE heuristic) when
/// either is missing; only SHORE-based fits use them.
pub(crate) fn gradient_table(
    bvals: PyReadonlyArray1<f64>,
    bvecs: PyReadonlyArray2<f64>,
    b0_threshold: f64,
    big_delta: Option<f64>,
    small_delta: Option<f64>,
) -> PyResult<GradientTable> {
    let bvals: Vec<f64> = bvals.as_array().to_vec();
    let bv = bvecs.as_array();
    if bv.ncols() != 3 {
        return Err(map_err(format!("bvecs must have shape (N, 3), got {:?}", bv.shape())));
    }
    let bvecs: Vec<[f64; 3]> = bv.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let mut g = GradientTable::new(bvals, bvecs, big_delta, small_delta, Some(TORTOISE_DEFAULT_GMAX))
        .map_err(map_err)?;
    g.b0_threshold = b0_threshold;
    Ok(g)
}

pub(crate) fn affine_from(a: PyReadonlyArray2<f64>) -> PyResult<[[f64; 4]; 4]> {
    let a = a.as_array();
    if a.shape() != [4, 4] {
        return Err(map_err(format!("affine must be 4x4, got {:?}", a.shape())));
    }
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| a[(i, j)])))
}

fn check_volume(data: &numpy::ndarray::ArrayView4<f32>, n_grads: usize) -> PyResult<()> {
    if data.shape()[3] != n_grads {
        return Err(map_err(format!(
            "data has {} volumes but the gradient table has {}",
            data.shape()[3],
            n_grads
        )));
    }
    Ok(())
}

// ------------------------------------------------------------------- QC

/// Model-free QC report as a dict, plus the per-(volume, slice) outlier
/// arrays under `outlier_flags` / `outlier_ratio`.
#[pyfunction]
#[pyo3(signature = (data, bvals, bvecs, mask=None, *, b0_threshold=50.0, slice_axis=2, min_slice_voxels=100, slice_smoothing_sigma=2.0, slice_threshold=2.5, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn qc_assess<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    b0_threshold: f64,
    slice_axis: usize,
    min_slice_voxels: usize,
    slice_smoothing_sigma: f64,
    slice_threshold: f64,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use cs_dmri::qc::{OutlierSliceOptions, QcOptions, assess};
    let g = gradient_table(bvals, bvecs, b0_threshold, None, None)?;
    let d = data.as_array();
    check_volume(&d, g.n_grads())?;
    let m = mask.as_ref().map(|m| m.as_array());
    let opts = QcOptions {
        b0_threshold,
        outlier_slices: OutlierSliceOptions {
            axis: slice_axis,
            min_voxels: min_slice_voxels,
            smoothing_sigma: slice_smoothing_sigma,
            threshold: slice_threshold,
        },
    };
    let report = run(py, n_threads, || assess(d, &g, m, &opts))?.map_err(map_err)?;
    let out = to_py_dict(py, &report)?;
    out.set_item("warnings", report.warnings())?;
    out.set_item("outlier_flags", report.outlier_slices.flags.clone().into_pyarray_bound(py))?;
    out.set_item("outlier_ratio", report.outlier_slices.ratio.clone().into_pyarray_bound(py))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (data, bvals, bvecs, mask=None, *, b0_threshold=50.0, n_threads=None))]
fn qc_neighboring_dwi_correlation<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    b0_threshold: f64,
    n_threads: Option<usize>,
) -> PyResult<Option<f64>> {
    let g = gradient_table(bvals, bvecs, b0_threshold, None, None)?;
    let (d, m) = (data.as_array(), mask.as_ref().map(|m| m.as_array()));
    run(py, n_threads, || cs_dmri::qc::neighboring_dwi_correlation(d, &g, m, b0_threshold))?
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(signature = (data, bvals, bvecs, mask=None, *, b0_threshold=50.0, n_threads=None))]
fn qc_dwi_contrast_ratio<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    b0_threshold: f64,
    n_threads: Option<usize>,
) -> PyResult<Option<f64>> {
    let g = gradient_table(bvals, bvecs, b0_threshold, None, None)?;
    let (d, m) = (data.as_array(), mask.as_ref().map(|m| m.as_array()));
    run(py, n_threads, || cs_dmri::qc::dwi_contrast_ratio(d, &g, m, b0_threshold))?.map_err(map_err)
}

/// `(flags, ratio)`, both `(n_volumes, n_slices)`.
#[pyfunction]
#[pyo3(signature = (data, mask=None, *, axis=2, min_voxels=100, smoothing_sigma=2.0, threshold=2.5, n_threads=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn qc_outlier_slices<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    axis: usize,
    min_voxels: usize,
    smoothing_sigma: f64,
    threshold: f64,
    n_threads: Option<usize>,
) -> PyResult<(Bound<'py, numpy::PyArray2<bool>>, Bound<'py, numpy::PyArray2<f64>>)> {
    use cs_dmri::qc::{OutlierSliceOptions, outlier_slices};
    let opts = OutlierSliceOptions { axis, min_voxels, smoothing_sigma, threshold };
    let (d, m) = (data.as_array(), mask.as_ref().map(|m| m.as_array()));
    let r = run(py, n_threads, || outlier_slices(d, m, &opts))?.map_err(map_err)?;
    Ok((r.flags.into_pyarray_bound(py), r.ratio.into_pyarray_bound(py)))
}

#[pyfunction]
#[pyo3(signature = (principal_dir, fa, mask, affine, *, world_directions=false, quantile=0.1, angle_degrees=15.0, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn qc_fixel_coherence<'py>(
    py: Python<'py>,
    principal_dir: PyReadonlyArray4<'py, f32>,
    fa: PyReadonlyArray3<'py, f32>,
    mask: PyReadonlyArray3<'py, bool>,
    affine: PyReadonlyArray2<'py, f64>,
    world_directions: bool,
    quantile: f32,
    angle_degrees: f32,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use cs_dmri::qc::{CoherenceOptions, fixel_coherence};
    let aff = affine_from(affine)?;
    let opts = CoherenceOptions { quantile, angle_degrees };
    let (pd, f, m) = (principal_dir.as_array(), fa.as_array(), mask.as_array());
    let r = run(py, n_threads, || fixel_coherence(pd, f, m, aff, world_directions, &opts))?
        .map_err(map_err)?;
    to_py_dict(py, &r)
}

/// The QC table's columns: list of dicts with `name`, `LongName`,
/// `Description`, and `Units` / `Replaces` where applicable.
#[pyfunction]
fn qc_columns(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    cs_dmri::qc::qc_columns().iter().map(|c| to_py_dict(py, c)).collect()
}

/// `(volume, neighbour)` pairs used by NDC.
#[pyfunction]
#[pyo3(signature = (bvals, bvecs, *, b0_threshold=50.0))]
fn qc_neighbor_pairs(
    bvals: PyReadonlyArray1<f64>,
    bvecs: PyReadonlyArray2<f64>,
    b0_threshold: f64,
) -> PyResult<Vec<(usize, usize)>> {
    let g = gradient_table(bvals, bvecs, b0_threshold, None, None)?;
    Ok(cs_dmri::qc::find_qspace_neighbors(&g, b0_threshold))
}

/// `(volume, contrast volume)` pairs used by the contrast ratio.
#[pyfunction]
#[pyo3(signature = (bvals, bvecs, *, b0_threshold=50.0))]
fn qc_contrast_pairs(
    bvals: PyReadonlyArray1<f64>,
    bvecs: PyReadonlyArray2<f64>,
    b0_threshold: f64,
) -> PyResult<Vec<(usize, usize)>> {
    let g = gradient_table(bvals, bvecs, b0_threshold, None, None)?;
    Ok(cs_dmri::qc::find_qspace_contrast(&g, b0_threshold))
}

// ------------------------------------------------------------------ DTI

/// Owned `DwiData` for the fitting entry points (they need owned arrays).
pub(crate) fn dwi_data(
    data: &PyReadonlyArray4<f32>,
    mask: &PyReadonlyArray3<bool>,
    gtab: GradientTable,
) -> PyResult<cs_dmri::io::dwi::DwiData> {
    let d = data.as_array();
    check_volume(&d, gtab.n_grads())?;
    let m = mask.as_array();
    if m.shape() != &d.shape()[..3] {
        return Err(map_err(format!(
            "mask shape {:?} does not match data spatial shape {:?}",
            m.shape(),
            &d.shape()[..3]
        )));
    }
    Ok(cs_dmri::io::dwi::DwiData::from_table(
        d.to_owned(),
        m.to_owned(),
        gtab,
        cs_dmri::qspace::BvecFrame::ImageAxis,
        std::path::PathBuf::new(),
    ))
}

fn put3<'py, T: numpy::Element>(d: &Bound<'py, PyDict>, k: &str, a: Array3<T>) -> PyResult<()> {
    d.set_item(k, a.into_pyarray_bound(d.py()))
}
fn put4<'py, T: numpy::Element>(d: &Bound<'py, PyDict>, k: &str, a: Array4<T>) -> PyResult<()> {
    d.set_item(k, a.into_pyarray_bound(d.py()))
}

/// RESTORE tensor fit. Returns `fa, md, s0, outlier_fraction (X,Y,Z)`,
/// `tensor (X,Y,Z,6)` [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], `principal_dir
/// (X,Y,Z,3)` in the bvecs' frame, and with `diagnostics` `iterations` /
/// `converged`.
#[pyfunction]
#[pyo3(signature = (data, bvals, bvecs, mask, *, b0_threshold=50.0, max_iter=50, tol=1e-6, outlier_threshold=0.04, min_signal=1e-6, diagnostics=false, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn dti_fit_restore<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    b0_threshold: f64,
    max_iter: usize,
    tol: f64,
    outlier_threshold: f64,
    min_signal: f64,
    diagnostics: bool,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use cs_dmri::dti::{DtiFitConfig, RestoreConfig, fit_volume_restore_reporting};
    let g = gradient_table(bvals, bvecs, b0_threshold, None, None)?;
    let dwi = dwi_data(&data, &mask, g)?;
    let cfg = RestoreConfig { max_iter, tol, outlier_threshold, min_signal };
    let r = run(py, n_threads, || {
        fit_volume_restore_reporting(&dwi, &cfg, DtiFitConfig { compute_diagnostics: diagnostics }, || {})
    })?
    .map_err(map_err)?;
    let out = PyDict::new_bound(py);
    put3(&out, "fa", r.fa)?;
    put3(&out, "md", r.md)?;
    put3(&out, "s0", r.s0)?;
    put3(&out, "outlier_fraction", r.outlier_fraction)?;
    put4(&out, "tensor", r.tensor)?;
    put4(&out, "principal_dir", r.principal_dir)?;
    if let Some(a) = r.iterations {
        put3(&out, "iterations", a)?;
    }
    if let Some(a) = r.converged {
        put3(&out, "converged", a)?;
    }
    Ok(out)
}

/// The fallback brain mask: mean b=0 above 1% of its maximum.
#[pyfunction]
#[pyo3(signature = (data, bvals, *, b0_threshold=50.0))]
fn b0_mask<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    b0_threshold: f64,
) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
    let d = data.as_array();
    let b = bvals.as_array();
    let (nx, ny, nz, nt) = d.dim();
    if nt != b.len() {
        return Err(map_err(format!("data has {nt} volumes but {} bvals", b.len())));
    }
    let b0: Vec<usize> = (0..nt).filter(|&t| b[t] <= b0_threshold).collect();
    if b0.is_empty() {
        return Err(map_err(format!("no b=0 volumes (b <= {b0_threshold})")));
    }
    let mean = Array3::from_shape_fn((nx, ny, nz), |(x, y, z)| {
        b0.iter().map(|&t| d[(x, y, z, t)] as f64).sum::<f64>() / b0.len() as f64
    });
    let max = mean.iter().cloned().fold(0.0_f64, f64::max);
    Ok(mean.mapv(|v| v > 0.01 * max).into_pyarray_bound(py))
}

#[pyfunction]
fn version() -> &'static str {
    cs_dmri::VERSION
}

#[pymodule]
fn _cs_dmri(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(qc_assess, m)?)?;
    m.add_function(wrap_pyfunction!(qc_neighboring_dwi_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(qc_dwi_contrast_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(qc_outlier_slices, m)?)?;
    m.add_function(wrap_pyfunction!(qc_fixel_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(qc_columns, m)?)?;
    m.add_function(wrap_pyfunction!(qc_neighbor_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(qc_contrast_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(dti_fit_restore, m)?)?;
    m.add_function(wrap_pyfunction!(b0_mask, m)?)?;
    let _ = Array2::<f64>::zeros((0, 0)); // keep the import for later modules
    Ok(())
}
