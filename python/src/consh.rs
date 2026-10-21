// SPDX-License-Identifier: MIT OR Apache-2.0
//! Continuous-b multi-tissue CSD (ConSH) bindings.

use std::path::PathBuf;

use numpy::ndarray::Axis;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use cs_dmri::dti::RestoreConfig;
use cs_dmri::graddev::VoxelGradDev;
use cs_dmri::io::dwi::DwiData;
use cs_dmri::io::odx_out::{
    ContinuousBQuantMeta, GeneratedBy, ReferenceB0Source, write_continuous_b_odx_with_affine,
};
use cs_dmri::multitissue::consh::{ConshConfig, ConshResponses};
use cs_dmri::multitissue::consh_pipeline::{
    ConshPipelineConfig, ConshResponseSource, consh_pipeline, estimate_dense_b_responses,
};
use cs_dmri::multitissue::mtnormalise::MtnormaliseConfig;
use cs_dmri::multitissue::pipeline::NoProgress;
use cs_dmri::multitissue::response_estimation::{
    ConshResponseConfig, DhollanderConfig, DhollanderSelectConfig, ResponseEstimator,
};
use cs_dmri::multitissue::response_pool::{PoolStatistic, pool_responses};
use cs_dmri::multitissue::response_smooth::SmoothResponseConfig;
use cs_dmri::qspace::{BvecFrame, GradientTable, TORTOISE_DEFAULT_GMAX};
use cs_dmri::solver::IclsConfig;

use crate::models::{put, resolve_format, response_from, response_to_py};
use crate::{affine_from, map_err, run, to_py_dict};

type PyResponse<'py> = (PyReadonlyArray2<'py, f64>, usize);

fn responses_from(r: (PyResponse<'_>, PyResponse<'_>, PyResponse<'_>)) -> ConshResponses {
    ConshResponses {
        wm: response_from(r.0 .0, r.0 .1),
        gm: response_from(r.1 .0, r.1 .1),
        csf: response_from(r.2 .0, r.2 .1),
    }
}

fn responses_to_py<'py>(py: Python<'py>, r: &ConshResponses) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("wm", response_to_py(py, &r.wm)?)?;
    d.set_item("gm", response_to_py(py, &r.gm)?)?;
    d.set_item("csf", response_to_py(py, &r.csf)?)?;
    Ok(d)
}

fn parse_estimator(name: &str) -> PyResult<Option<ResponseEstimator>> {
    Ok(match name {
        "auto" => None,
        "kernel" => Some(ResponseEstimator::Kernel),
        "smooth" => Some(ResponseEstimator::Smooth),
        "dki" => Some(ResponseEstimator::Dki),
        other => {
            return Err(map_err(format!(
                "estimator must be 'auto', 'kernel', 'smooth' or 'dki', got {other:?}"
            )))
        }
    })
}

fn estimator_name(e: ResponseEstimator) -> &'static str {
    match e {
        ResponseEstimator::Kernel => "kernel",
        ResponseEstimator::Smooth => "smooth",
        ResponseEstimator::Dki => "dki",
    }
}

fn reference_source_name(s: ReferenceB0Source) -> &'static str {
    match s {
        ReferenceB0Source::UserMask => "mask",
        ReferenceB0Source::WhiteMatterSingleFibre => "wm_single_fiber",
        ReferenceB0Source::WhiteMatterFraction => "wm_fraction",
    }
}

/// A `DwiData` whose gradient table is `rotation · bvecs` (the fit frame) while
/// keeping the voxel-axis directions, which a gradient-deviation field acts on.
#[allow(clippy::too_many_arguments)]
fn dwi_with_frame(
    data: &PyReadonlyArray4<f32>,
    mask: &PyReadonlyArray3<bool>,
    bvals: &PyReadonlyArray1<f64>,
    bvecs: &PyReadonlyArray2<f64>,
    rotation: [[f64; 3]; 3],
    world: bool,
    b0_threshold: f64,
) -> PyResult<DwiData> {
    let d = data.as_array();
    let m = mask.as_array();
    let bv = bvecs.as_array();
    let bvals: Vec<f64> = bvals.as_array().to_vec();
    if bv.ncols() != 3 || bv.nrows() != bvals.len() {
        return Err(map_err(format!(
            "bvecs must have shape ({}, 3), got {:?}",
            bvals.len(),
            bv.shape()
        )));
    }
    if d.shape()[3] != bvals.len() {
        return Err(map_err(format!(
            "data has {} volumes but the gradient table has {}",
            d.shape()[3],
            bvals.len()
        )));
    }
    if m.shape() != &d.shape()[..3] {
        return Err(map_err(format!(
            "mask shape {:?} does not match data spatial shape {:?}",
            m.shape(),
            &d.shape()[..3]
        )));
    }
    let vox: Vec<[f64; 3]> = bv.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let fit: Vec<[f64; 3]> = vox
        .iter()
        .map(|v| std::array::from_fn(|i| (0..3).map(|j| rotation[i][j] * v[j]).sum()))
        .collect();
    let mut gtab = GradientTable::new(bvals, fit, None, None, Some(TORTOISE_DEFAULT_GMAX)).map_err(map_err)?;
    gtab.b0_threshold = b0_threshold;
    Ok(DwiData {
        data: d.to_owned(),
        mask: m.to_owned(),
        gtab,
        bvec_frame: if world { BvecFrame::WorldRas } else { BvecFrame::ImageAxis },
        bvecs_vox: vox,
        frame_rotation: rotation,
        source_nifti: PathBuf::new(),
    })
}

fn graddev_from(
    field: Option<PyReadonlyArray4<f32>>,
    identity: &str,
    affine: Option<[[f64; 4]; 4]>,
    dwi: &DwiData,
) -> PyResult<Option<VoxelGradDev>> {
    let Some(field) = field else { return Ok(None) };
    let affine = affine.ok_or_else(|| map_err("a gradient-deviation field needs the series' affine"))?;
    let a = field.as_array();
    let s = a.shape();
    if s.len() != 4 || s[3] != 9 {
        return Err(map_err(format!("graddev must have shape (X, Y, Z, 9), got {s:?}")));
    }
    let policy = match identity {
        "auto" => odx_rs::IdentityPolicy::Auto,
        "included" => odx_rs::IdentityPolicy::Included,
        "absent" => odx_rs::IdentityPolicy::Absent,
        other => {
            return Err(map_err(format!(
                "graddev_identity must be 'auto', 'included' or 'absent', got {other:?}"
            )))
        }
    };
    let data: Vec<f32> = a.iter().copied().collect();
    let field = odx_rs::graddev::GradDevField::from_parts([s[0], s[1], s[2]], affine, data, policy)
        .map_err(map_err)?;
    VoxelGradDev::new(field, dwi, &affine).map(Some).map_err(map_err)
}

#[allow(clippy::too_many_arguments)]
fn response_config(
    step: f64,
    sigma_b: f64,
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
) -> ConshResponseConfig {
    ConshResponseConfig {
        dhollander: DhollanderConfig {
            legacy_selection,
            stages: DhollanderSelectConfig { erode, fa, sfwm_pct, gm_pct, csf_pct, ..DhollanderSelectConfig::default() },
            fa_wm_threshold,
            fiber_dominance_ratio,
            md_csf_pct,
            lmax_wm,
        },
        step,
        sigma_b,
        smooth: SmoothResponseConfig { smoothing_b: sigma_b, ..SmoothResponseConfig::default() },
        ..ConshResponseConfig::default()
    }
}

/// Dense-b responses estimated from a series, as `consh` estimates them.
#[pyfunction]
#[pyo3(signature = (data, bvals, bvecs, mask, rotation, *, world=true, affine=None, b0_threshold=50.0,
    graddev=None, graddev_identity="auto", estimator="auto", b_interp=false, step=50.0, sigma_b=150.0,
    lmax_wm=8, legacy_selection=false, erode=3, fa=0.2, sfwm_pct=0.5, gm_pct=2.0, csf_pct=10.0,
    fa_wm_threshold=0.7, fiber_dominance_ratio=2.0, md_csf_pct=2.5, restore_max_iter=50,
    restore_tol=1e-6, n_threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn consh_estimate_responses<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    rotation: PyReadonlyArray2<'py, f64>,
    world: bool,
    affine: Option<PyReadonlyArray2<'py, f64>>,
    b0_threshold: f64,
    graddev: Option<PyReadonlyArray4<'py, f32>>,
    graddev_identity: &str,
    estimator: &str,
    b_interp: bool,
    step: f64,
    sigma_b: f64,
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
    restore_max_iter: usize,
    restore_tol: f64,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let rot = rotation3(&rotation)?;
    let affine = affine.map(affine_from).transpose()?;
    let dwi = dwi_with_frame(&data, &mask, &bvals, &bvecs, rot, world, b0_threshold)?;
    let graddev = graddev_from(graddev, graddev_identity, affine, &dwi)?;
    let restore = RestoreConfig { max_iter: restore_max_iter, tol: restore_tol, ..RestoreConfig::default() };
    let rcfg = response_config(
        step, sigma_b, lmax_wm, legacy_selection, erode, fa, sfwm_pct, gm_pct, csf_pct,
        fa_wm_threshold, fiber_dominance_ratio, md_csf_pct,
    );
    let estimator = parse_estimator(estimator)?;
    let est = run(py, n_threads, || {
        estimate_dense_b_responses(&dwi, graddev.as_ref(), &restore, &rcfg, estimator, b_interp, &NoProgress)
    })?
    .map_err(map_err)?;
    let d = responses_to_py(py, &est.responses)?;
    d.set_item("estimator", estimator_name(est.estimator))?;
    d.set_item("grid_bmax", est.grid_bmax)?;
    d.set_item("diagnostics", to_py_dict(py, &est.diagnostics)?)?;
    Ok(d)
}

fn rotation3(rotation: &PyReadonlyArray2<f64>) -> PyResult<[[f64; 3]; 3]> {
    let r = rotation.as_array();
    if r.shape() != [3, 3] {
        return Err(map_err(format!("rotation must be 3x3, got {:?}", r.shape())));
    }
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| r[(i, j)])))
}

/// Responses → ConSH fit → mtnormalise (`consh`). With `responses` given they
/// are used as-is; otherwise they are estimated from the series.
#[pyfunction]
#[pyo3(signature = (data, bvals, bvecs, mask, rotation, *, world=true, affine=None, b0_threshold=50.0,
    responses=None, graddev=None, graddev_identity="auto", estimator="auto", step=50.0, sigma_b=150.0,
    lmax_wm=8, legacy_selection=false, erode=3, fa=0.2, sfwm_pct=0.5, gm_pct=2.0, csf_pct=10.0,
    fa_wm_threshold=0.7, fiber_dominance_ratio=2.0, md_csf_pct=2.5, restore_max_iter=50,
    restore_tol=1e-6, b_interp=false, noise_floor=true, quantitative=false, wm_ridge=0.0,
    wm_ridge_anchor=0.0, wm_ridge_relative=false, icls_max_iter=2000, normalize=true,
    mtnorm_target=None, mtnorm_poly_order=3, mtnorm_balanced=false, reference_mask=None,
    diagnostics=false, n_threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn consh_run<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    bvals: PyReadonlyArray1<'py, f64>,
    bvecs: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    rotation: PyReadonlyArray2<'py, f64>,
    world: bool,
    affine: Option<PyReadonlyArray2<'py, f64>>,
    b0_threshold: f64,
    responses: Option<(PyResponse<'py>, PyResponse<'py>, PyResponse<'py>)>,
    graddev: Option<PyReadonlyArray4<'py, f32>>,
    graddev_identity: &str,
    estimator: &str,
    step: f64,
    sigma_b: f64,
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
    restore_max_iter: usize,
    restore_tol: f64,
    b_interp: bool,
    noise_floor: bool,
    quantitative: bool,
    wm_ridge: f64,
    wm_ridge_anchor: f64,
    wm_ridge_relative: bool,
    icls_max_iter: usize,
    normalize: bool,
    mtnorm_target: Option<f64>,
    mtnorm_poly_order: usize,
    mtnorm_balanced: bool,
    reference_mask: Option<PyReadonlyArray3<'py, bool>>,
    diagnostics: bool,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let rot = rotation3(&rotation)?;
    let affine = affine.map(affine_from).transpose()?;
    let dwi = dwi_with_frame(&data, &mask, &bvals, &bvecs, rot, world, b0_threshold)?;
    let graddev = graddev_from(graddev, graddev_identity, affine, &dwi)?;
    let source = match responses {
        Some(r) => ConshResponseSource::Provided(responses_from(r)),
        None => ConshResponseSource::Estimate {
            restore: RestoreConfig { max_iter: restore_max_iter, tol: restore_tol, ..RestoreConfig::default() },
            responses: response_config(
                step, sigma_b, lmax_wm, legacy_selection, erode, fa, sfwm_pct, gm_pct, csf_pct,
                fa_wm_threshold, fiber_dominance_ratio, md_csf_pct,
            ),
        },
    };
    let wm_ridge_path = if wm_ridge_anchor > 0.0 {
        if wm_ridge_relative {
            vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        } else {
            vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        }
    } else {
        Vec::new()
    };
    let cfg = ConshPipelineConfig {
        responses: source,
        estimator: parse_estimator(estimator)?,
        consh: ConshConfig {
            lmax_wm,
            step,
            icls: IclsConfig { max_iter: icls_max_iter, ..IclsConfig::default() },
            wm_l2_ridge: wm_ridge,
            wm_ridge_path,
            l2_anchor_slack: wm_ridge_anchor,
            wm_ridge_relative,
            fit_noise_floor: noise_floor,
            quantitative,
            b_interp,
            grid_bmax: None,
        },
        normalize: normalize.then(|| MtnormaliseConfig {
            poly_order: mtnorm_poly_order,
            target_sum: mtnorm_target,
            apply_balance: mtnorm_balanced,
            ..MtnormaliseConfig::default()
        }),
        compute_diagnostics: diagnostics,
        reference_mask: reference_mask.map(|m| m.as_array().to_owned()),
    };
    let out = run(py, n_threads, || consh_pipeline(&dwi, graddev.as_ref(), &cfg, &NoProgress))?
        .map_err(map_err)?;

    let d = PyDict::new_bound(py);
    put(&d, "wm", out.fit.wm)?;
    put(&d, "gm", out.fit.gm.index_axis_move(Axis(3), 0))?;
    put(&d, "csf", out.fit.csf.index_axis_move(Axis(3), 0))?;
    if let Some(f) = out.fit.floor {
        put(&d, "noise_floor", f.index_axis_move(Axis(3), 0))?;
    }
    if let Some(a) = out.fit.converged {
        put(&d, "converged", a)?;
    }
    if let Some(a) = out.fit.iterations {
        put(&d, "iterations", a)?;
    }
    d.set_item("responses", responses_to_py(py, &out.responses)?)?;
    d.set_item("estimator", out.estimator.map(estimator_name))?;
    d.set_item("grid_bmax", out.grid_bmax)?;
    d.set_item("b_interp", out.b_interp)?;
    d.set_item("b_shift_p95", out.b_shift_p95)?;
    d.set_item("reference_b0", out.reference_b0.map(|(v, s)| (v, reference_source_name(s))))?;
    d.set_item("warnings", out.warnings)?;
    if let Some((diag, _)) = &out.estimate {
        d.set_item("response_diagnostics", to_py_dict(py, diag)?)?;
    }
    if let Some(n) = out.normalization {
        let nd = PyDict::new_bound(py);
        put(&nd, "bias_field", n.bias_field)?;
        nd.set_item("target_sum", n.target_sum_used)?;
        nd.set_item("balance_factors", n.tissue_scales.to_vec())?;
        nd.set_item("lognorm_scale", n.lognorm_scale)?;
        nd.set_item("n_fit_voxels", n.n_fit_voxels)?;
        nd.set_item("mean_abs_log_residual", n.mean_abs_log_residual)?;
        nd.set_item("iterations", n.iterations)?;
        d.set_item("normalization", nd)?;
    }
    Ok(d)
}

/// Pool per-scan ConSH responses into one group response (`consh-response mean`).
#[pyfunction]
#[pyo3(signature = (inputs, *, statistic="mean", legacy=false))]
pub fn consh_pool_responses<'py>(
    py: Python<'py>,
    inputs: Vec<(PyResponse<'py>, PyResponse<'py>, PyResponse<'py>)>,
    statistic: &str,
    legacy: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let stat = match statistic {
        "mean" => PoolStatistic::Mean,
        "median" => PoolStatistic::Median,
        other => return Err(map_err(format!("statistic must be 'mean' or 'median', got {other:?}"))),
    };
    let inputs: Vec<ConshResponses> = inputs.into_iter().map(responses_from).collect();
    let (pooled, tissues) = pool_responses(&inputs, stat, legacy).map_err(map_err)?;
    let d = responses_to_py(py, &pooled)?;
    for (name, t) in ["wm", "gm", "csf"].iter().zip(&tissues) {
        let td = PyDict::new_bound(py);
        td.set_item("multipliers", t.multipliers.clone())?;
        td.set_item("deviation_pct", t.deviation_pct.clone())?;
        d.set_item(format!("{name}_pooling"), td)?;
    }
    Ok(d)
}

/// Write a ConSH fit in any supported format (`consh --odx`, plus the other
/// formats of `odx convert`).
#[pyfunction]
#[pyo3(signature = (path, affine, mask, wm, gm, csf, lmax_wm, wm_response, gm_response, csf_response, step,
    *, noise_floor=None, format=None, fixel_container="nifti", overwrite=false, peak_min_amplitude=0.0,
    peak_min_amplitude_frac=0.0, quantitative=false, reference_b0=None, mtnorm_target=None,
    mtnorm_scale=None, external_responses=false, b_interp=false, estimator=None, source=None))]
#[allow(clippy::too_many_arguments)]
pub fn consh_export<'py>(
    path: PathBuf,
    affine: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    wm: PyReadonlyArray4<'py, f32>,
    gm: PyReadonlyArray3<'py, f32>,
    csf: PyReadonlyArray3<'py, f32>,
    lmax_wm: usize,
    wm_response: PyResponse<'py>,
    gm_response: PyResponse<'py>,
    csf_response: PyResponse<'py>,
    step: f64,
    noise_floor: Option<PyReadonlyArray3<'py, f32>>,
    format: Option<&str>,
    fixel_container: &str,
    overwrite: bool,
    peak_min_amplitude: f32,
    peak_min_amplitude_frac: f32,
    quantitative: bool,
    reference_b0: Option<(f64, String)>,
    mtnorm_target: Option<f64>,
    mtnorm_scale: Option<f64>,
    external_responses: bool,
    b_interp: bool,
    estimator: Option<String>,
    source: Option<PathBuf>,
) -> PyResult<()> {
    let out_format = resolve_format(&path, format, fixel_container)?;
    let aff = affine_from(affine)?;
    let to4 = |a: numpy::ndarray::ArrayView3<f32>| a.to_owned().insert_axis(Axis(3));
    let responses = responses_from((wm_response, gm_response, csf_response));
    let reference_b0 = match reference_b0 {
        Some((v, s)) => Some((
            v,
            match s.as_str() {
                "mask" => ReferenceB0Source::UserMask,
                "wm_single_fiber" => ReferenceB0Source::WhiteMatterSingleFibre,
                "wm_fraction" => ReferenceB0Source::WhiteMatterFraction,
                other => return Err(map_err(format!("unknown reference_b0 source {other:?}"))),
            },
        )),
        None => None,
    };
    let meta = ContinuousBQuantMeta {
        generated_by: GeneratedBy {
            name: "consh".into(),
            version: cs_dmri::VERSION.into(),
            description: "Continuous-b multi-tissue CSD for non-shelled q-space.".into(),
        },
        quantitative,
        reference_b0,
        intensity_normalization_target: mtnorm_target,
        intensity_normalization_scale: mtnorm_scale,
        external_responses,
        noise_floor: noise_floor.is_some(),
        sh_degree: lmax_wm,
        graddev: None,
        b_interp,
        // As `consh` records it.
        response_estimator: estimator.map(|e| match parse_estimator(&e) {
            Ok(Some(est)) => format!("{est:?}"),
            _ => e,
        }),
    };
    let (m, w, g, c) = (mask.as_array().to_owned(), wm.as_array().to_owned(), to4(gm.as_array()), to4(csf.as_array()));
    let floor = noise_floor.map(|f| to4(f.as_array()));
    Python::with_gil(|py| {
        py.allow_threads(|| {
            write_continuous_b_odx_with_affine(
                &path,
                aff,
                source.as_deref(),
                &m,
                &w,
                &g,
                &c,
                lmax_wm,
                [&responses.wm, &responses.gm, &responses.csf],
                "consh_responses",
                step,
                floor.as_ref(),
                overwrite,
                out_format,
                peak_min_amplitude,
                peak_min_amplitude_frac,
                Some(&meta),
            )
        })
    })
    .map_err(map_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(consh_run, m)?)?;
    m.add_function(wrap_pyfunction!(consh_estimate_responses, m)?)?;
    m.add_function(wrap_pyfunction!(consh_pool_responses, m)?)?;
    m.add_function(wrap_pyfunction!(consh_export, m)?)?;
    Ok(())
}
