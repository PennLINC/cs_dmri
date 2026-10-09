import json
import warnings

import numpy as np
import pytest

import cs_dmri as cs

nib = pytest.importorskip("nibabel")


def multishell_series(shape=(6, 6, 4), seed=3):
    """Two-shell single-tensor series with a little noise, on a 2 mm LPS-ish grid."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(60, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((4, 3)), v])
    bvals = np.concatenate([np.zeros(4), np.repeat([1000.0, 2500.0], 30)])
    D = np.diag([1.7e-3, 0.4e-3, 0.4e-3])
    s = np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs, D, bvecs))
    data = 1000.0 * np.broadcast_to(s, shape + (len(bvals),)) * (1 + 0.005 * rng.normal(size=shape + (len(bvals),)))
    affine = np.array([[-2.0, 0, 0, 10], [0, 2.0, 0, -10], [0, 0, 2.0, 5], [0, 0, 0, 1]])
    return data.astype(np.float32), bvals, bvecs, affine


def test_shore_fit_predicts_its_data():
    data, bvals, bvecs, affine = multishell_series()
    dwi = cs.DWI(data, (bvals, bvecs), affine=affine, mask=np.ones(data.shape[:3], bool))
    fit = cs.ShoreModel(dwi.gtab, regularization="l2").fit(dwi)
    assert fit.frame == "world"
    assert fit.coefficients.shape == data.shape[:3] + (72,)  # radial order 6
    pred = fit.predict()
    rel = np.abs(pred - data).mean() / data.mean()
    assert rel < 0.02
    assert float(fit.r2[fit.mask].mean()) > 0.95
    sh = fit.odf_sh()
    assert sh.shape == data.shape[:3] + (28,)  # lmax 6
    ms = fit.microstructure()
    assert set(ms) == {"rtop", "rtap", "rtpp", "msd", "qiv", "ng"}
    assert np.isfinite(ms["rtap"][fit.mask]).all()  # directions from the cached tensor fit


def test_shore_save_load_roundtrip(tmp_path):
    data, bvals, bvecs, affine = multishell_series()
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, 2)  # aligned-anat code should survive
    img.set_sform(affine, 4)  # MNI code should survive
    dwi = cs.DWI.from_nibabel(img, (bvals, bvecs), mask=np.ones(data.shape[:3], bool))
    fit = cs.ShoreModel(dwi.gtab, regularization="l2").fit(dwi)
    fit.save(tmp_path / "coef.nii.gz")
    out = nib.load(tmp_path / "coef.nii.gz")
    assert int(out.header["qform_code"]) == 2 and int(out.header["sform_code"]) == 4
    assert (tmp_path / "coef_r2.nii.gz").exists()
    side = json.loads((tmp_path / "coef.json").read_text())
    assert side["bvec_frame"] == "world-ras" and side["solver"]["kind"] == "tikhonov"
    back = cs.ShoreFit.load(tmp_path / "coef.nii.gz")
    np.testing.assert_array_equal(back.coefficients, fit.coefficients)
    assert back.frame == "world"
    with pytest.raises(FileExistsError):
        fit.save(tmp_path / "coef.nii.gz")


EXPORTS = [
    ("out.odx", None),
    ("out_dir", "odx-directory"),
    ("out.fz", None),
    ("out.fib.gz", None),
    ("out.pam5", None),
    ("out_sh.mif", None),
    ("out_sh.nii.gz", None),
    ("fixels", "mrtrix-fixel-dir"),
]


def _check_export(path, format):
    assert path.exists()
    if format in ("odx-directory", "mrtrix-fixel-dir"):
        assert path.is_dir() and any(path.iterdir())
    else:
        assert path.is_file() and path.stat().st_size > 0


@pytest.fixture(scope="module")
def shore_fit():
    data, bvals, bvecs, affine = multishell_series()
    dwi = cs.DWI(data, (bvals, bvecs), affine=affine, mask=np.ones(data.shape[:3], bool))
    return cs.ShoreModel(dwi.gtab, regularization="l2").fit(dwi)


@pytest.mark.parametrize("name,format", EXPORTS)
def test_shore_export(tmp_path, shore_fit, name, format):
    shore_fit.export(tmp_path / name, format=format)
    _check_export(tmp_path / name, format)
    with pytest.raises(FileExistsError):
        shore_fit.export(tmp_path / name, format=format)
    shore_fit.export(tmp_path / name, format=format, overwrite=True)


def test_shore_export_errors(tmp_path, shore_fit):
    with pytest.raises(ValueError, match="infer"):
        shore_fit.export(tmp_path / "no_extension")
    with pytest.raises(ValueError, match="unknown output format"):
        shore_fit.export(tmp_path / "x.odx", format="trackvis")
    with pytest.raises(ValueError, match="fixel_container"):
        shore_fit.export(tmp_path / "fx", format="mrtrix-fixel-dir", fixel_container="minc")
    shore_fit.export(tmp_path / "fx_mif", format="mrtrix-fixel-dir", fixel_container="mif")
    assert any(p.suffix == ".mif" for p in (tmp_path / "fx_mif").iterdir())


def test_shore_to_odx(tmp_path, shore_fit):
    shore_fit.to_odx(tmp_path / "fit.odx")
    assert (tmp_path / "fit.odx").is_file()
    shore_fit.to_odx(tmp_path / "fit_dir", directory=True)
    assert (tmp_path / "fit_dir").is_dir()
    data, bvals, bvecs, affine = multishell_series()
    dwi = cs.DWI(data, (bvals, bvecs), affine=affine, mask=np.ones(data.shape[:3], bool))
    image_fit = cs.ShoreModel(dwi.gtab, regularization="l2", bvec_frame="image").fit(dwi)
    with pytest.raises(ValueError, match="world"):
        image_fit.to_odx(tmp_path / "bad.odx")


def synthetic_ss3t_fit():
    """An SS3TFit built directly: a z-oriented WM FOD everywhere plus GM/CSF."""
    shape = (5, 5, 4)
    lmax = 8
    n_sh = (lmax + 1) * (lmax + 2) // 2
    wm = np.zeros(shape + (n_sh,), np.float32)
    # Zonal coefficients of a smooth z-aligned lobe (Tournier indices l(l+1)/2).
    for l, c in zip(range(0, lmax + 1, 2), (0.28, 0.3, 0.2, 0.1, 0.04)):
        wm[..., l * (l + 1) // 2] = c
    gm = np.full(shape, 0.05, np.float32)
    csf = np.full(shape, 0.02, np.float32)
    affine = np.array([[-2.0, 0, 0, 10], [0, 2.0, 0, -10], [0, 0, 2.0, 5], [0, 0, 0, 1]])
    responses = cs.ResponseSet(
        wm=cs.TissueResponse(np.array([[3.5, 0, 0, 0, 0], [1.2, -0.6, 0.2, -0.05, 0.01]]), lmax),
        gm=cs.TissueResponse(np.array([[3.0], [1.0]]), 0),
        csf=cs.TissueResponse(np.array([[4.0], [0.3]]), 0))
    return cs.SS3TFit(wm=wm, gm=gm, csf=csf, lmax_wm=lmax, mask=np.ones(shape, bool), affine=affine,
                      frame="world", responses=responses)


@pytest.mark.parametrize("name,format", EXPORTS)
def test_ss3t_export(tmp_path, name, format):
    fit = synthetic_ss3t_fit()
    fit.export(tmp_path / name, format=format)
    _check_export(tmp_path / name, format)


def test_ss3t_mrtrix_sh_image_matches(tmp_path):
    fit = synthetic_ss3t_fit()
    fit.export(tmp_path / "wm.nii.gz")
    img = nib.load(tmp_path / "wm.nii.gz")
    assert img.shape[-1] == fit.wm.shape[-1]
    # Already RAS+ up to the x flip, which the canonical reorientation undoes.
    np.testing.assert_allclose(np.asarray(img.dataobj)[0, 0, 0], fit.wm[0, 0, 0], rtol=1e-5, atol=1e-6)


def test_response_text_roundtrip(tmp_path):
    r = cs.TissueResponse(np.array([[3.5, 0.0, 0.0], [2.0, -0.4, 0.05]]), 4)
    p = tmp_path / "wm.txt"
    r.to_mrtrix_txt(p)
    back = cs.TissueResponse.from_mrtrix_txt(p)
    np.testing.assert_allclose(back.coeffs, r.coeffs)


def test_mtnormalise_returns_new_arrays():
    rng = np.random.default_rng(0)
    shape = (10, 10, 6)
    wm = (0.2 + 0.01 * rng.random(shape + (15,))).astype(np.float32)
    gm = (0.05 + 0.01 * rng.random(shape)).astype(np.float32)
    csf = (0.03 + 0.01 * rng.random(shape)).astype(np.float32)
    mask = np.ones(shape, bool)
    wm0 = wm.copy()
    out_wm, out_gm, out_csf, diag = cs.mtnormalise(wm, gm, csf, mask)
    np.testing.assert_array_equal(wm, wm0)
    assert out_gm.shape == shape and diag["n_fit_voxels"] > 0
    # With the balance factors applied, the tissue sum lands on the target.
    out_wm, out_gm, out_csf, diag = cs.mtnormalise(wm, gm, csf, mask, balanced=True)
    total = out_wm[..., 0] + out_gm + out_csf
    assert np.median(total[mask]) == pytest.approx(cs.multitissue.MTNORMALISE_TARGET, rel=0.05)


# ------------------------------------------------------------ nibabel checks

def test_mask_on_another_grid_is_rejected(tmp_path):
    data, bvals, bvecs, affine = multishell_series()
    img = nib.Nifti1Image(data, affine)
    shifted = affine.copy()
    shifted[0, 3] += 2.0
    mask_img = nib.Nifti1Image(np.ones(data.shape[:3], np.uint8), shifted)
    with pytest.raises(ValueError, match="not on the DWI's grid"):
        cs.DWI.from_nibabel(img, (bvals, bvecs), mask=mask_img)
    small = nib.Nifti1Image(np.ones((5, 6, 4), np.uint8), affine)
    with pytest.raises(ValueError, match=r"shape \(5, 6, 4\)"):
        cs.DWI.from_nibabel(img, (bvals, bvecs), mask=small)


def test_header_problems_warn():
    data, bvals, bvecs, affine = multishell_series()
    img = nib.Nifti1Image(data, affine)
    other = affine.copy()
    other[1, 3] += 3.0
    img.set_qform(other, 1)
    img.set_sform(affine, 1)
    with pytest.warns(cs.SpatialWarning, match="qform and sform differ"):
        dwi = cs.DWI.from_nibabel(img, (bvals, bvecs))
    assert any("qform" in w for w in dwi.warnings)
    assert any("qform" in w for w in dwi.qc(coherence=False).warnings)

    oblique = affine.copy()
    c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
    oblique[:3, :3] = oblique[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    with pytest.warns(cs.SpatialWarning, match="oblique"):
        cs.DWI.from_nibabel(nib.Nifti1Image(data, oblique), (bvals, bvecs))

    clean = nib.Nifti1Image(data, affine)
    with warnings.catch_warnings():
        warnings.simplefilter("error", cs.SpatialWarning)
        cs.DWI.from_nibabel(clean, (bvals, bvecs))


def test_to_image_and_zooms():
    data, bvals, bvecs, affine = multishell_series()
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units("mm", "sec")
    dwi = cs.DWI.from_nibabel(img, (bvals, bvecs))
    assert dwi.zooms == pytest.approx((2.0, 2.0, 2.0))
    out = dwi.to_image(dwi.tensor.fa)
    assert out.shape == data.shape[:3]
    assert out.header.get_xyzt_units()[0] == "mm"
    np.testing.assert_allclose(out.affine, affine)
    with pytest.raises(ValueError, match="not on the DWI grid"):
        dwi.to_image(np.zeros((2, 2, 2)))


def test_pam5_export_loads_in_dipy(tmp_path):
    peaks = pytest.importorskip("dipy.io.peaks")
    shm = pytest.importorskip("dipy.reconst.shm")
    fit = synthetic_ss3t_fit()
    fit.export(tmp_path / "fit.pam5")
    fit.export(tmp_path / "wm.nii.gz")
    pam = peaks.load_pam(str(tmp_path / "fit.pam5"))
    assert pam.shm_coeff.shape == fit.wm.shape
    # MRtrix SH are in world RAS; the PAM grid here is RAS+, so both describe
    # the same function in the same frame.
    rng = np.random.default_rng(0)
    v = rng.normal(size=(100, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    theta, phi = np.arccos(v[:, 2]), np.arctan2(v[:, 1], v[:, 0])
    assert np.allclose(pam.affine[:3, :3], np.diag(np.abs(np.diag(pam.affine[:3, :3]))))
    b_tournier = shm.real_sh_tournier(8, theta, phi, legacy=False)[0]
    b_dipy = shm.real_sh_descoteaux(8, theta, phi)[0]  # dipy's default basis
    mrtrix = np.asarray(nib.load(tmp_path / "wm.nii.gz").dataobj)[2, 2, 2]
    np.testing.assert_allclose(b_dipy @ pam.shm_coeff[2, 2, 2], b_tournier @ mrtrix, atol=1e-5)
