// SPDX-License-Identifier: MIT OR Apache-2.0
//! SHORE, response estimation, SS3T, mtnormalise and ODX bindings.

use std::collections::HashMap;
use std::path::PathBuf;

use numpy::ndarray::{Array2, Array3, Array4, Axis};
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use cs_dmri::basis::Basis;
use cs_dmri::dti::DtiVolumeResult;
use cs_dmri::multitissue::TissueResponse;
use cs_dmri::multitissue::ss3t::Ss3tResponses;
use cs_dmri::ShoreBasis;

use crate::{affine_from, dwi_data, gradient_table, map_err, run, to_py_dict};

fn put<'py, T: numpy::Element, D: numpy::ndarray::Dimension>(
    d: &Bound<'py, PyDict>,
    k: &str,
    a: numpy::ndarray::Array<T, D>,
) -> PyResult<()> {
    d.set_item(k, a.into_pyarray_bound(d.py()))
}

/// Unit 3×3 rotation of an affine (columns normalised), the matrix the CLI
/// uses to rotate image-axis bvecs into world RAS.
#[pyfunction]
pub fn affine_rotation<'py>(py: Python<'py>, affine: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let r = cs_dmri::qspace::affine_rotation(&affine_from(affine)?);
    Ok(Array2::from_shape_fn((3, 3), |(i, j)| r[i][j]).into_pyarray_bound(py))
}

// ------------------------------------------------------------------ SHORE

/// SHORE fit. Returns `coefficients (X,Y,Z,K)`, the sidecar metadata dict
/// (`sidecar`), `alpha_distribution` and, with `diagnostics`, the maps
/// `r2, residual_l2, iterations, regularization_kind` (+ `alpha`, `bic`,
/// `rss_l2` where the solver has them).
#[pyfunction]
#[pyo3(signature = (data, bvals, bvecs, mask, *, big_delta=None, small_delta=None, b0_threshold=50.0,
    bvec_frame="image-axis", radial_order=6, zeta=700.0, regularization="l1", alpha_mode="l2-anchored",
    alpha=1.0, alpha_ratio=1e-3, path_n_alphas=20, path_eps=None, slack=0.05, max_iter=1000, tol=1e-6,
    non_negative=false, lambda_n=1e-8, lambda_l=1e-8, nonneg_max_iter=200, nonneg_tol=1e-9,
    nonneg_epsilon=1e-10, diagnostics=false, n_threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn shore_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    big_delta: Option<f64>,
    small_delta: Option<f64>,
    b0_threshold: f64,
    bvec_frame: &str,
    radial_order: u32,
    zeta: f64,
    regularization: &str,
    alpha_mode: &str,
    alpha: f64,
    alpha_ratio: f64,
    path_n_alphas: usize,
    path_eps: Option<f64>,
    slack: f64,
    max_iter: u32,
    tol: f64,
    non_negative: bool,
    lambda_n: f64,
    lambda_l: f64,
    nonneg_max_iter: usize,
    nonneg_tol: f64,
    nonneg_epsilon: f64,
    diagnostics: bool,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use cs_dmri::fit::{AlphaMode, ShoreFitSpec, ShoreRegularization, build_alpha_strategy, fit_shore};
    use cs_dmri::io::coeffs::SidecarMetadata;
    use cs_dmri::qspace::BvecFrame;

    let frame = match bvec_frame {
        "image-axis" => BvecFrame::ImageAxis,
        "world-ras" => BvecFrame::WorldRas,
        other => return Err(map_err(format!("bvec_frame must be 'image-axis' or 'world-ras', got {other:?}"))),
    };
    let reg = match regularization {
        "l1" => {
            let mode = match alpha_mode {
                "fixed" => AlphaMode::Fixed,
                "alpha-ratio" => AlphaMode::AlphaRatio,
                "path-bic" => AlphaMode::PathBic,
                "l2-anchored" => AlphaMode::L2Anchored,
                other => {
                    return Err(map_err(format!(
                        "alpha_mode must be fixed, alpha-ratio, path-bic or l2-anchored, got {other:?}"
                    )))
                }
            };
            ShoreRegularization::L1 {
                strategy: build_alpha_strategy(mode, alpha, alpha_ratio, path_n_alphas, path_eps, slack)
                    .map_err(map_err)?,
                max_iter,
                tol,
                non_negative,
                seed_alpha: alpha,
            }
        }
        "l2" => ShoreRegularization::L2,
        "nonneg" => ShoreRegularization::AmpNonNeg {
            icls: cs_dmri::solver::IclsConfig { max_iter: nonneg_max_iter, tol: nonneg_tol, epsilon: nonneg_epsilon },
        },
        other => return Err(map_err(format!("regularization must be l1, l2 or nonneg, got {other:?}"))),
    };
    if radial_order % 2 != 0 || radial_order == 0 {
        return Err(map_err(format!("radial_order must be a positive even number, got {radial_order}")));
    }
    let g = gradient_table(bvals, bvecs, b0_threshold, big_delta, small_delta)?;
    let mut dwi = dwi_data(&data, &mask, g)?;
    dwi.bvec_frame = frame;
    let spec = ShoreFitSpec {
        radial_order,
        zeta,
        regularization: reg,
        lambda_n,
        lambda_l,
        compute_diagnostics: diagnostics,
    };
    let out = run(py, n_threads, || fit_shore(&dwi, &spec, || {}))?.map_err(map_err)?;

    let sidecar = SidecarMetadata {
        basis: out.basis.metadata(),
        big_delta_seconds: dwi.gtab.big_delta,
        small_delta_seconds: dwi.gtab.small_delta,
        tau_seconds: dwi.gtab.tau(),
        delta_source: dwi.gtab.delta_source,
        gmax_tesla_per_meter: Some(cs_dmri::qspace::TORTOISE_DEFAULT_GMAX),
        solver: out.solver.clone(),
        n_coefficients: out.basis.n_coeffs(),
        bvec_frame: frame,
        provenance: None,
    };
    let d = PyDict::new_bound(py);
    put(&d, "coefficients", out.result.coefficients)?;
    d.set_item("sidecar", to_py_dict(py, &sidecar)?)?;
    d.set_item("alpha_distribution", out.alpha_distribution)?;
    if let Some(diag) = out.result.diagnostics {
        let rmse = cs_dmri::fit::rmse_from_residual_l2(&diag.residual_l2, dwi.gtab.n_grads());
        put(&d, "rmse", rmse)?;
        put(&d, "r2", diag.r2)?;
        put(&d, "residual_l2", diag.residual_l2)?;
        put(&d, "iterations", diag.iterations)?;
        put(&d, "regularization_kind", diag.regularization_kind)?;
        if let Some(a) = diag.alpha {
            put(&d, "alpha", a)?;
        }
        if let Some(a) = diag.bic {
            put(&d, "bic", a)?;
        }
        if let Some(a) = diag.rss_l2 {
            put(&d, "rss_l2", a)?;
        }
    }
    Ok(d)
}

/// Project SHORE coefficients to Tournier (MRtrix) SH of order `lmax`.
#[pyfunction]
#[pyo3(signature = (coefficients, radial_order, zeta, lmax, mask=None, *, n_threads=None))]
pub fn shore_odf_sh<'py>(
    py: Python<'py>,
    coefficients: PyReadonlyArray4<'py, f32>,
    radial_order: u32,
    zeta: f64,
    lmax: u32,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, numpy::PyArray4<f32>>> {
    if lmax % 2 != 0 || lmax > radial_order {
        return Err(map_err(format!("lmax must be even and ≤ radial_order ({radial_order}), got {lmax}")));
    }
    let basis = ShoreBasis::new(radial_order, zeta);
    let c = coefficients.as_array();
    if c.shape()[3] != basis.n_coeffs() {
        return Err(map_err(format!(
            "coefficients have {} channels but a radial-order-{radial_order} SHORE basis has {}",
            c.shape()[3],
            basis.n_coeffs()
        )));
    }
    let m = mask.as_ref().map(|m| m.as_array());
    let mat = cs_dmri::odf::shore_to_tournier_sh_matrix(&basis, lmax);
    let (nx, ny, nz, nk) = c.dim();
    let n_sh = mat.nrows();
    let out = run(py, n_threads, || {
        let mut out = Array4::<f32>::zeros((nx, ny, nz, n_sh));
        out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(x, mut plane)| {
            for y in 0..ny {
                for z in 0..nz {
                    if m.is_some_and(|m| !m[(x, y, z)]) {
                        continue;
                    }
                    for i in 0..n_sh {
                        let mut acc = 0.0f64;
                        for k in 0..nk {
                            acc += mat[(i, k)] * c[(x, y, z, k)] as f64;
                        }
                        plane[(y, z, i)] = acc as f32;
                    }
                }
            }
        });
        out
    })?;
    Ok(out.into_pyarray_bound(py))
}

/// Synthesize a DWI from SHORE coefficients for a new gradient table.
#[pyfunction]
#[pyo3(signature = (coefficients, radial_order, zeta, bvals, bvecs, *, big_delta=None, small_delta=None, n_threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn shore_predict<'py>(
    py: Python<'py>,
    coefficients: PyReadonlyArray4<'py, f32>,
    radial_order: u32,
    zeta: f64,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    big_delta: Option<f64>,
    small_delta: Option<f64>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, numpy::PyArray4<f32>>> {
    let basis = ShoreBasis::new(radial_order, zeta);
    let c = coefficients.as_array().to_owned();
    if c.shape()[3] != basis.n_coeffs() {
        return Err(map_err(format!(
            "coefficients have {} channels but a radial-order-{radial_order} SHORE basis has {}",
            c.shape()[3],
            basis.n_coeffs()
        )));
    }
    let g = gradient_table(bvals, bvecs, 50.0, big_delta, small_delta)?;
    let out = run(py, n_threads, || cs_dmri::synthesize_volume(&c, &basis, &g))?;
    Ok(out.into_pyarray_bound(py))
}

/// RTOP, RTAP, RTPP, MSD, QIV and NG maps (zero outside `mask`; NaN where a
/// voxel's fit was rejected as an outlier). RTAP/RTPP need `directions`
/// `(X,Y,Z,3)` in the coefficients' frame; without them they are NaN.
#[pyfunction]
#[pyo3(signature = (coefficients, radial_order, zeta, mask, directions=None, *, units="um", outlier_factor=Some(10.0), n_threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn shore_microstructure<'py>(
    py: Python<'py>,
    coefficients: PyReadonlyArray4<'py, f32>,
    radial_order: u32,
    zeta: f64,
    mask: PyReadonlyArray3<'py, bool>,
    directions: Option<PyReadonlyArray4<'py, f32>>,
    units: &str,
    outlier_factor: Option<f32>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use cs_dmri::io::microstructure::{
        MicrostructureOptions, MicrostructureOutlierRejection, MicrostructureUnits, compute_microstructure,
    };
    let basis = ShoreBasis::new(radial_order, zeta);
    let c = coefficients.as_array().to_owned();
    let m = mask.as_array();
    let (nx, ny, nz, _) = c.dim();
    if m.shape() != [nx, ny, nz] {
        return Err(map_err("mask does not match the coefficient grid"));
    }
    let idx: Vec<(usize, usize, usize)> = (0..nx)
        .flat_map(|x| (0..ny).flat_map(move |y| (0..nz).map(move |z| (x, y, z))))
        .filter(|&p| m[p])
        .collect();
    let dirs: Option<Vec<Option<[f32; 3]>>> = directions.map(|d| {
        let d = d.as_array();
        idx.iter()
            .map(|&(x, y, z)| {
                let v = [d[(x, y, z, 0)], d[(x, y, z, 1)], d[(x, y, z, 2)]];
                (v.iter().any(|c| *c != 0.0) && v.iter().all(|c| c.is_finite())).then_some(v)
            })
            .collect()
    });
    let opts = MicrostructureOptions {
        units: match units {
            "um" => MicrostructureUnits::Um,
            "mm" => MicrostructureUnits::Mm,
            other => return Err(map_err(format!("units must be 'um' or 'mm', got {other:?}"))),
        },
        outlier_rejection: outlier_factor.map(|p99_factor| MicrostructureOutlierRejection { p99_factor }),
        quiet: true,
    };
    let s = run(py, n_threads, || compute_microstructure(&basis, &c, &idx, dirs.as_deref(), &opts))?;
    let d = PyDict::new_bound(py);
    for (name, vals) in [("rtop", &s.rtop), ("rtap", &s.rtap), ("rtpp", &s.rtpp), ("msd", &s.msd), ("qiv", &s.qiv), ("ng", &s.ng)] {
        let mut vol = Array3::<f32>::zeros((nx, ny, nz));
        for (&p, &v) in idx.iter().zip(vals.iter()) {
            vol[p] = v;
        }
        put(&d, name, vol)?;
    }
    Ok(d)
}

/// Write SHORE coefficients as an ODX (canonical RAS+), like `cs-odf`.
#[pyfunction]
#[pyo3(signature = (path, coefficients, affine, radial_order, zeta, mask=None, *, lmax=None, dpvs=None,
    peaks=true, npeaks=5, peak_relative_threshold=0.5, peak_min_separation_deg=25.0, global_normalize=true,
    anisotropic_power=true, microstructure=true, units="um", outlier_factor=Some(10.0),
    field_name=String::from("coefficients"), directory=false, overwrite=false, n_threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn shore_write_odx<'py>(
    py: Python<'py>,
    path: PathBuf,
    coefficients: PyReadonlyArray4<'py, f32>,
    affine: PyReadonlyArray2<'py, f64>,
    radial_order: u32,
    zeta: f64,
    mask: Option<PyReadonlyArray3<'py, bool>>,
    lmax: Option<u32>,
    dpvs: Option<HashMap<String, PyReadonlyArray3<'py, f32>>>,
    peaks: bool,
    npeaks: usize,
    peak_relative_threshold: f32,
    peak_min_separation_deg: f32,
    global_normalize: bool,
    anisotropic_power: bool,
    microstructure: bool,
    units: &str,
    outlier_factor: Option<f32>,
    field_name: String,
    directory: bool,
    overwrite: bool,
    n_threads: Option<usize>,
) -> PyResult<()> {
    use cs_dmri::io::microstructure::{MicrostructureOptions, MicrostructureOutlierRejection, MicrostructureUnits};
    use cs_dmri::io::odx_out::{
        ShoreOdxOptions, ShoreOdxPeakOpts, ShoreToOdxOptions, finalize_and_write_odx, shore_coeffs_to_odx,
    };
    let aff = affine_from(affine)?;
    let basis = ShoreBasis::new(radial_order, zeta);
    let c = coefficients.as_array().to_owned();
    let m = mask.map(|m| m.as_array().to_owned());
    let dpv_owned: Vec<(String, Array3<f32>)> = dpvs
        .unwrap_or_default()
        .into_iter()
        .map(|(k, v)| (k, v.as_array().to_owned()))
        .collect();
    let units = match units {
        "um" => MicrostructureUnits::Um,
        "mm" => MicrostructureUnits::Mm,
        other => return Err(map_err(format!("units must be 'um' or 'mm', got {other:?}"))),
    };
    let opts = ShoreToOdxOptions {
        lmax,
        odx: ShoreOdxOptions {
            field_name,
            global_normalize,
            anisotropic_power,
            peaks: peaks.then_some(ShoreOdxPeakOpts {
                npeaks,
                relative_threshold: peak_relative_threshold,
                min_separation_deg: peak_min_separation_deg,
            }),
            quiet: true,
            ..ShoreOdxOptions::default()
        },
        microstructure: microstructure.then_some(MicrostructureOptions {
            units,
            outlier_rejection: outlier_factor.map(|p99_factor| MicrostructureOutlierRejection { p99_factor }),
            quiet: true,
        }),
    };
    run(py, n_threads, || -> anyhow::Result<()> {
        let refs: Vec<(&str, &Array3<f32>)> = dpv_owned.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let out = shore_coeffs_to_odx(&c, aff, m.as_ref(), &basis, &refs, &opts)?;
        finalize_and_write_odx(out.build.builder, &path, overwrite, directory)
    })?
    .map_err(map_err)
}

// ------------------------------------------------------------- responses

fn response_from(coeffs: PyReadonlyArray2<f64>, lmax: usize) -> TissueResponse {
    TissueResponse {
        coeffs: coeffs.as_array().rows().into_iter().map(|r| r.to_vec()).collect(),
        lmax,
    }
}

fn response_to_py<'py>(py: Python<'py>, r: &TissueResponse) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
    let ncol = r.coeffs.iter().map(|c| c.len()).max().unwrap_or(0);
    let a = Array2::from_shape_fn((r.coeffs.len(), ncol), |(i, j)| r.coeffs[i].get(j).copied().unwrap_or(0.0));
    Ok((a.into_pyarray_bound(py), r.lmax))
}

/// Parse an MRtrix response `.txt` body: `(coeffs (n_shells, n_coef), lmax)`.
#[pyfunction]
pub fn response_from_text<'py>(py: Python<'py>, text: &str) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
    let r = TissueResponse::parse_mrtrix_text(text, "<text>").map_err(map_err)?;
    response_to_py(py, &r)
}

/// Format a response as MRtrix `.txt` (what `cs-response` writes).
#[pyfunction]
pub fn response_to_text(coeffs: PyReadonlyArray2<f64>, lmax: usize) -> String {
    cs_dmri::multitissue::format_response_txt(&response_from(coeffs, lmax))
}

fn dti_result(dti: &Bound<'_, PyDict>) -> PyResult<DtiVolumeResult> {
    fn get3(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Array3<f32>> {
        let v = d.get_item(k)?.ok_or_else(|| map_err(format!("tensor fit is missing {k:?}")))?;
        Ok(v.extract::<PyReadonlyArray3<f32>>()?.as_array().to_owned())
    }
    fn get4(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Array4<f32>> {
        let v = d.get_item(k)?.ok_or_else(|| map_err(format!("tensor fit is missing {k:?}")))?;
        Ok(v.extract::<PyReadonlyArray4<f32>>()?.as_array().to_owned())
    }
    Ok(DtiVolumeResult {
        s0: get3(dti, "s0")?,
        fa: get3(dti, "fa")?,
        md: get3(dti, "md")?,
        outlier_fraction: get3(dti, "outlier_fraction")?,
        tensor: get4(dti, "tensor")?,
        principal_dir: get4(dti, "principal_dir")?,
        iterations: None,
        converged: None,
    })
}

/// Dhollander (2016) response estimation from a single-shell DWI and its
/// tensor fit (`dti`: dict with `s0, fa, md, outlier_fraction, tensor,
/// principal_dir`, as `dti_fit_restore` returns).
#[pyfunction]
#[pyo3(signature = (data, bvals, bvecs, mask, dti, *, b0_threshold=50.0, lmax_wm=8, legacy_selection=false,
    erode=3, fa=0.2, sfwm_pct=0.5, gm_pct=2.0, csf_pct=10.0, fa_wm_threshold=0.7,
    fiber_dominance_ratio=2.0, md_csf_pct=2.5, n_threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_responses<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    dti: Bound<'py, PyDict>,
    b0_threshold: f64,
    lmax_wm: usize,
    legacy_selection: bool,
    erode: usize,
    fa: f64,
    sfwm_pct: f64,
    gm_pct: f64,
    csf_pct: f64,
    fa_wm_threshold: f64,
    fiber_dominance_ratio: f64,
    md_csf_pct: f64,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use cs_dmri::multitissue::response_estimation::{DhollanderConfig, DhollanderSelectConfig};
    let g = gradient_table(bvals, bvecs, b0_threshold, None, None)?;
    let dwi = dwi_data(&data, &mask, g)?;
    let dti = dti_result(&dti)?;
    let cfg = DhollanderConfig {
        legacy_selection,
        stages: DhollanderSelectConfig {
            erode: erode.try_into().map_err(map_err)?,
            fa,
            sfwm_pct,
            gm_pct,
            csf_pct,
            ..DhollanderSelectConfig::default()
        },
        fa_wm_threshold,
        fiber_dominance_ratio,
        md_csf_pct,
        lmax_wm,
    };
    let est = run(py, n_threads, || cs_dmri::multitissue::estimate_responses(&dwi, &dti, &cfg))?
        .map_err(map_err)?;
    let d = PyDict::new_bound(py);
    for (name, r) in [("wm", &est.wm), ("gm", &est.gm), ("csf", &est.csf)] {
        d.set_item(name, response_to_py(py, r)?)?;
    }
    put(&d, "wm_mask", est.wm_mask.mapv(|v| v > 0))?;
    put(&d, "gm_mask", est.gm_mask.mapv(|v| v > 0))?;
    put(&d, "csf_mask", est.csf_mask.mapv(|v| v > 0))?;
    d.set_item("diagnostics", to_py_dict(py, &est.diagnostics)?)?;
    Ok(d)
}

// -------------------------------------------------------------- SS3T etc.

/// SS3T-CSD. Responses are `(coeffs, lmax)` pairs. `lmax_candidates` selects
/// path-BIC WM lmax per voxel instead of the fixed `lmax_wm`.
#[pyfunction]
#[pyo3(signature = (data, bvals, bvecs, mask, wm, gm, csf, *, b0_threshold=50.0, niter=3, bzero_pct=10.0,
    lmax_wm=8, lmax_candidates=None, icls_max_iter=200, icls_tol=1e-10, icls_epsilon=1e-10,
    diagnostics=false, n_threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn ss3t_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    wm: (PyReadonlyArray2<'py, f64>, usize),
    gm: (PyReadonlyArray2<'py, f64>, usize),
    csf: (PyReadonlyArray2<'py, f64>, usize),
    b0_threshold: f64,
    niter: u32,
    bzero_pct: f64,
    lmax_wm: usize,
    lmax_candidates: Option<Vec<usize>>,
    icls_max_iter: usize,
    icls_tol: f64,
    icls_epsilon: f64,
    diagnostics: bool,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use cs_dmri::multitissue::ss3t::{LmaxWmStrategy, Ss3tConfig};
    use cs_dmri::multitissue::volume::{Ss3tFitConfig, fit_volume_ss3t};
    let g = gradient_table(bvals, bvecs, b0_threshold, None, None)?;
    let dwi = dwi_data(&data, &mask, g)?;
    let responses = Ss3tResponses {
        wm: response_from(wm.0, wm.1),
        gm: response_from(gm.0, gm.1),
        csf: response_from(csf.0, csf.1),
    };
    let cfg = Ss3tConfig {
        niter,
        bzero_pct,
        lmax_wm: match lmax_candidates {
            Some(c) => LmaxWmStrategy::PathBic(c),
            None => LmaxWmStrategy::Fixed(lmax_wm),
        },
        icls: cs_dmri::solver::IclsConfig { max_iter: icls_max_iter, tol: icls_tol, epsilon: icls_epsilon },
    };
    let r = run(py, n_threads, || {
        fit_volume_ss3t(&dwi, &responses, &cfg, Ss3tFitConfig { compute_diagnostics: diagnostics })
    })?
    .map_err(map_err)?;
    let d = PyDict::new_bound(py);
    d.set_item("warnings", r.plan.lmax_clamp_warnings(responses.wm.lmax))?;
    d.set_item("lmax_wm", r.plan.plans.iter().map(|p| p.lmax_wm).max().unwrap_or(0))?;
    put(&d, "wm", r.wm)?;
    put(&d, "gm", r.gm.index_axis_move(Axis(3), 0))?;
    put(&d, "csf", r.csf.index_axis_move(Axis(3), 0))?;
    if let Some(a) = r.iterations {
        put(&d, "iterations", a)?;
    }
    if let Some(a) = r.residual_l2 {
        put(&d, "residual_l2", a)?;
    }
    if let Some(a) = r.converged {
        put(&d, "converged", a)?;
    }
    if let Some(a) = r.chosen_lmax {
        put(&d, "chosen_lmax", a)?;
    }
    if let Some(a) = r.bic {
        put(&d, "bic", a)?;
    }
    Ok(d)
}

/// Multi-tissue log-domain intensity normalisation (MRtrix3 `mtnormalise`
/// port). `target_sum=None` uses the median observed sum. Returns new
/// `(wm, gm, csf, diagnostics)`; inputs are not modified.
#[pyfunction]
#[pyo3(signature = (wm, gm, csf, mask, *, target_sum=Some(0.28209479177387814), poly_order=3, niter=15, balance_maxiter=7, balanced=false))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn mtnormalise<'py>(
    py: Python<'py>,
    wm: PyReadonlyArray4<'py, f32>,
    gm: PyReadonlyArray3<'py, f32>,
    csf: PyReadonlyArray3<'py, f32>,
    mask: PyReadonlyArray3<'py, bool>,
    target_sum: Option<f64>,
    poly_order: usize,
    niter: usize,
    balance_maxiter: usize,
    balanced: bool,
) -> PyResult<(
    Bound<'py, numpy::PyArray4<f32>>,
    Bound<'py, numpy::PyArray3<f32>>,
    Bound<'py, numpy::PyArray3<f32>>,
    Bound<'py, PyDict>,
)> {
    use cs_dmri::multitissue::mtnormalise::{MtnormaliseConfig, mtnormalise as mtn};
    let mut w = wm.as_array().to_owned();
    let mut g = gm.as_array().to_owned().insert_axis(Axis(3));
    let mut c = csf.as_array().to_owned().insert_axis(Axis(3));
    let m = mask.as_array().to_owned();
    let cfg = MtnormaliseConfig {
        poly_order,
        target_sum,
        niter,
        balance_maxiter,
        apply_balance: balanced,
    };
    let diag = run(py, None, || mtn(&mut w, &mut g, &mut c, &m, &cfg))?.map_err(map_err)?;
    let d = PyDict::new_bound(py);
    put(&d, "bias_field", diag.bias_field)?;
    d.set_item("poly_coefs", diag.poly_coefs)?;
    d.set_item("n_fit_voxels", diag.n_fit_voxels)?;
    d.set_item("target_sum", diag.target_sum_used)?;
    d.set_item("mean_abs_log_residual", diag.mean_abs_log_residual)?;
    d.set_item("balance_factors", diag.tissue_scales.to_vec())?;
    d.set_item("iterations", diag.iterations)?;
    d.set_item("lognorm_scale", diag.lognorm_scale)?;
    Ok((
        w.into_pyarray_bound(py),
        g.index_axis_move(Axis(3), 0).into_pyarray_bound(py),
        c.index_axis_move(Axis(3), 0).into_pyarray_bound(py),
        d,
    ))
}

/// Write SS3T tissue maps as an ODX (WM SH glyphs, GM/CSF, peaks).
#[pyfunction]
#[pyo3(signature = (path, affine, mask, wm, gm, csf, lmax_wm, wm_response, gm_response, csf_response, *, directory=false, overwrite=false))]
#[allow(clippy::too_many_arguments)]
pub fn ss3t_write_odx<'py>(
    py: Python<'py>,
    path: PathBuf,
    affine: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    wm: PyReadonlyArray4<'py, f32>,
    gm: PyReadonlyArray3<'py, f32>,
    csf: PyReadonlyArray3<'py, f32>,
    lmax_wm: usize,
    wm_response: (PyReadonlyArray2<'py, f64>, usize),
    gm_response: (PyReadonlyArray2<'py, f64>, usize),
    csf_response: (PyReadonlyArray2<'py, f64>, usize),
    directory: bool,
    overwrite: bool,
) -> PyResult<()> {
    let aff = affine_from(affine)?;
    let responses = Ss3tResponses {
        wm: response_from(wm_response.0, wm_response.1),
        gm: response_from(gm_response.0, gm_response.1),
        csf: response_from(csf_response.0, csf_response.1),
    };
    let (m, w) = (mask.as_array().to_owned(), wm.as_array().to_owned());
    let g = gm.as_array().to_owned().insert_axis(Axis(3));
    let c = csf.as_array().to_owned().insert_axis(Axis(3));
    run(py, None, || {
        cs_dmri::io::odx_out::write_ss3t_odx_with_affine(
            &path, aff, &m, &w, &g, &c, lmax_wm, &responses, overwrite, directory,
        )
    })?
    .map_err(map_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(affine_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(shore_fit, m)?)?;
    m.add_function(wrap_pyfunction!(shore_odf_sh, m)?)?;
    m.add_function(wrap_pyfunction!(shore_predict, m)?)?;
    m.add_function(wrap_pyfunction!(shore_microstructure, m)?)?;
    m.add_function(wrap_pyfunction!(shore_write_odx, m)?)?;
    m.add_function(wrap_pyfunction!(response_from_text, m)?)?;
    m.add_function(wrap_pyfunction!(response_to_text, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_responses, m)?)?;
    m.add_function(wrap_pyfunction!(ss3t_fit, m)?)?;
    m.add_function(wrap_pyfunction!(mtnormalise, m)?)?;
    m.add_function(wrap_pyfunction!(ss3t_write_odx, m)?)?;
    Ok(())
}
