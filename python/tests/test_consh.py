import warnings

import numpy as np
import pytest

import cs_dmri as cs

CENTER = 9.5


def dsi_lattice():
    """DSI lattice points with |q|² ≤ 16, one per antipodal pair, plus 2 b=0:
    14 distinct b-values up to 3000 s/mm², no shells."""
    bvals, bvecs = [0.0, 0.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(0, 5):
                n2 = i * i + j * j + k * k
                upper = k > 0 or j > 0 or (j == 0 and i > 0)
                if n2 == 0 or n2 > 16 or not upper:
                    continue
                bvals.append(3000.0 * n2 / 16)
                bvecs.append(np.array([i, j, k]) / np.sqrt(n2))
    return np.array(bvals), np.array(bvecs, dtype=float)


def circumferential(x, y):
    dx, dy = x - CENTER, y - CENTER
    r = max(np.hypot(dx, dy), 1e-9)
    return np.array([-dy / r, dx / r, 0.0])


def synthetic_brain(shape=(20, 20, 4), seed=3):
    """CSF core, a ring of circumferential WM, a WM/z crossing ring and GM outside."""
    bvals, bvecs = dsi_lattice()
    rng = np.random.default_rng(seed)
    data = np.zeros(shape + (len(bvals),), np.float32)
    mask = np.zeros(shape, bool)
    tissue = np.full(shape[:2], "", dtype=object)

    def zeppelin(axis):
        c = bvecs @ axis
        return np.exp(-bvals * (0.2e-3 + 1.5e-3 * c * c))

    for x in range(shape[0]):
        for y in range(shape[1]):
            r = np.hypot(x - CENTER, y - CENTER)
            if r < 3:
                t, s = "csf", 2600 * np.exp(-3.0e-3 * bvals)
            elif r < 6.5:
                t, s = "wm", 1000 * zeppelin(circumferential(x, y))
            elif r < 7.8:
                t, s = "crossing", 500 * zeppelin(circumferential(x, y)) + 500 * zeppelin(np.array([0, 0, 1.0]))
            elif r < 9.5:
                t, s = "gm", 1300 * np.exp(-0.8e-3 * bvals)
            else:
                continue
            tissue[x, y] = t
            mask[x, y, :] = True
            data[x, y, :, :] = s + rng.normal(0, 4, size=(shape[2], len(bvals)))
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    dwi = cs.DWI(np.maximum(data, 0), (bvals, bvecs), affine=affine, mask=mask)
    return dwi, tissue


@pytest.fixture(scope="module")
def brain():
    return synthetic_brain()


def _fast(**kw):
    # The synthetic volume is too small for the default mask erosion.
    return dict(erode=0, **kw)


def test_dense_b_responses_estimated_and_round_trip(brain, tmp_path):
    dwi, _ = brain
    resp = cs.estimate_dense_b_responses(dwi, estimator="kernel", **_fast())
    assert resp.estimator == "kernel" and resp.step == 50.0
    assert resp.wm.coeffs.shape[1] == 5 and resp.gm.coeffs.shape[1] == 1
    assert resp.b[0] == 0 and resp.b[-1] >= 3000
    assert resp.diagnostics["n_wm_voxels"] > 0
    paths = resp.save(tmp_path / "resp")
    assert sorted(p.name for p in paths.values()) == [
        "csf_response_denseb.txt", "gm_response_denseb.txt", "wm_response_denseb.txt"]
    with pytest.raises(FileExistsError):
        resp.save(tmp_path / "resp")
    back = cs.DenseBResponses.load(tmp_path / "resp")
    np.testing.assert_allclose(back.wm.coeffs, resp.wm.coeffs, rtol=1e-6)


@pytest.mark.parametrize("estimator", ["kernel", "smooth", "dki"])
def test_pipeline_recovers_tissues_and_fibers(brain, estimator):
    dwi, tissue = brain
    fit = cs.consh_pipeline(dwi, estimator=estimator, **_fast())
    assert fit.responses.estimator == estimator and not fit.external_responses
    assert fit.noise_floor is not None and fit.normalization is not None
    assert fit.reference_b0[1] == "wm_single_fiber"
    z = 2
    total = fit.wm[..., z, 0] + fit.gm[..., z] + fit.csf[..., z]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = {"csf": fit.csf[..., z] / total, "gm": fit.gm[..., z] / total,
                 "wm": fit.wm[..., z, 0] / total}
    for name in ("csf", "gm", "wm"):
        inside = tissue == name
        assert np.mean(share[name][inside] > 0.8) > 0.95, name
    # WM peak direction: evaluate the fODF on a dense set of directions.
    n = 3000
    zz = 1 - (np.arange(n) + 0.5) / n
    phi = np.pi * (3 - np.sqrt(5)) * np.arange(n)
    dirs = np.stack([np.sqrt(1 - zz**2) * np.cos(phi), np.sqrt(1 - zz**2) * np.sin(phi), zz], 1)
    errors = []
    for x, y in zip(*np.nonzero(tissue == "wm")):
        sh = fit.wm[x, y, z]
        amp = _amplitudes(sh, dirs)
        peak = dirs[np.argmax(amp)]
        errors.append(np.degrees(np.arccos(min(1.0, abs(peak @ circumferential(x, y))))))
    assert np.median(errors) < 2.0 and np.percentile(errors, 90) < 4.0


def _amplitudes(sh, dirs):
    """Amplitudes of MRtrix-basis (Tournier) SH coefficients at unit vectors."""
    shm = pytest.importorskip("dipy.reconst.shm")
    theta = np.arccos(np.clip(dirs[:, 2], -1, 1))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    basis = shm.real_sh_tournier(8, theta, phi, legacy=False)[0]
    return basis @ sh


def test_model_fit_with_fixed_responses_matches_pipeline_without_normalization(brain):
    dwi, _ = brain
    resp = cs.estimate_dense_b_responses(dwi, estimator="kernel", **_fast())
    s0_fit = cs.ConshModel(dwi.gtab, resp, quantitative=False).fit(dwi)
    pipe = cs.consh_pipeline(dwi, responses=resp, quantitative=False, normalize=False)
    np.testing.assert_array_equal(s0_fit.wm, pipe.wm)
    assert s0_fit.external_responses and s0_fit.normalization is None
    model_fit = cs.ConshModel(dwi.gtab, resp).fit(dwi)
    assert model_fit.quantitative and model_fit.normalization is None
    normalized = model_fit.mtnormalise()
    full = cs.consh_pipeline(dwi, responses=resp)
    np.testing.assert_allclose(normalized.wm, full.wm, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(normalized.noise_floor, full.noise_floor, rtol=1e-5, atol=1e-6)


def test_identity_graddev_matches_interpolated_fit(brain):
    dwi, _ = brain
    resp = cs.estimate_dense_b_responses(dwi, estimator="kernel", **_fast())
    identity = np.broadcast_to(np.eye(3).ravel(), dwi.shape[:3] + (9,)).astype(np.float32)
    with_field = cs.ConshModel(dwi.gtab, resp).fit(dwi, graddev=identity, graddev_identity="included")
    interp = cs.ConshModel(dwi.gtab, resp, b_interp=True).fit(dwi)
    np.testing.assert_allclose(with_field.wm, interp.wm, rtol=1e-5, atol=1e-6)
    assert with_field.b_interp


def test_pooling_and_validation(brain):
    dwi, _ = brain
    a = cs.estimate_dense_b_responses(dwi, estimator="kernel", **_fast())
    b = cs.estimate_dense_b_responses(dwi, estimator="smooth", **_fast())
    pooled = cs.DenseBResponses.pool([a, b])
    assert pooled.estimator is None and len(pooled.diagnostics["wm"]["multipliers"]) == 2
    np.testing.assert_allclose(cs.DenseBResponses.pool([a, a]).wm.coeffs, a.wm.coeffs, rtol=1e-12)
    with pytest.raises(ValueError):
        cs.DenseBResponses.pool([a])
    with pytest.raises(ValueError, match="grid steps"):
        cs.DenseBResponses.pool([a, cs.DenseBResponses(a.wm, a.gm, a.csf, step=25.0)])
    with pytest.raises(ValueError, match="normalize"):
        cs.consh_pipeline(dwi, quantitative=True, normalize=False)
    with pytest.raises(ValueError, match="estimator"):
        cs.consh_pipeline(dwi, estimator="tensor")
    with pytest.raises(TypeError, match="unexpected"):
        cs.consh_pipeline(dwi, sfwm=1.0)


@pytest.mark.parametrize("name,format", [("fit.odx", None), ("fit.fz", None), ("fit.pam5", None),
                                         ("fit_wm.mif", None), ("fixels", "mrtrix-fixel-dir")])
def test_export(brain, tmp_path, name, format):
    dwi, _ = brain
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = cs.consh_pipeline(dwi, estimator="kernel", quantitative=True, **_fast())
    fit.export(tmp_path / name, format=format)
    assert (tmp_path / name).exists()
    with pytest.raises(FileExistsError):
        fit.export(tmp_path / name, format=format)


def test_single_shell_data_are_refused():
    bvals, bvecs = dsi_lattice()
    keep = (bvals == 0) | (np.round(bvals) == 1500)
    rng = np.random.default_rng(0)
    data = rng.uniform(100, 1000, size=(6, 6, 2, int(keep.sum()))).astype(np.float32)
    dwi = cs.DWI(data, (bvals[keep], bvecs[keep]), affine=np.eye(4), mask=np.ones((6, 6, 2), bool))
    with pytest.raises(ValueError, match="single-shell three-tissue CSD"):
        cs.consh_pipeline(dwi)
    with pytest.raises(ValueError, match="3 or more distinct b-values"):
        cs.estimate_dense_b_responses(dwi)
