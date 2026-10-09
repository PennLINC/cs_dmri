import numpy as np
import pytest

import cs_dmri as cs


def tensor_series(fa_axis=(1.0, 0.0, 0.0), evals=(1.7e-3, 0.3e-3, 0.3e-3), shape=(6, 6, 4)):
    rng = np.random.default_rng(1)
    v = rng.normal(size=(40, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((3, 3)), v])
    bvals = np.concatenate([np.zeros(3), np.full(40, 1000.0)])
    e1 = np.asarray(fa_axis, float)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, [0, 0, 1] if abs(e1[2]) < 0.9 else [1, 0, 0])
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    D = sum(l * np.outer(e, e) for l, e in zip(evals, (e1, e2, e3)))
    s = 1000.0 * np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs, D, bvecs))
    data = np.broadcast_to(s, shape + (len(bvals),)).astype(np.float32).copy()
    return data, bvals, bvecs, evals


def test_restore_recovers_tensor():
    data, bvals, bvecs, evals = tensor_series()
    fit = cs.RestoreModel((bvals, bvecs)).fit(data, np.ones(data.shape[:3], bool))
    l1, l2, l3 = evals
    md = (l1 + l2 + l3) / 3
    fa = np.sqrt(1.5 * ((l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2) / (l1**2 + l2**2 + l3**2))
    assert fit.fa[2, 2, 2] == pytest.approx(fa, rel=1e-3)
    assert fit.md[2, 2, 2] == pytest.approx(md, rel=1e-3)
    assert abs(fit.principal_dir[2, 2, 2, 0]) == pytest.approx(1.0, abs=1e-3)
    assert fit.evals[2, 2, 2] == pytest.approx(evals, rel=1e-2)
    assert fit.quadratic_form.shape == data.shape[:3] + (3, 3)


def test_dipy_gradient_table_accepted():
    pytest.importorskip("dipy")
    from dipy.core.gradients import gradient_table

    data, bvals, bvecs, _ = tensor_series()
    g = gradient_table(bvals, bvecs=bvecs)
    dwi = cs.DWI(data, g)
    assert isinstance(dwi.gtab, cs.GradientTable)
    assert np.array_equal(dwi.gtab.bvals, bvals)


def test_dwi_caches_and_with_mask():
    data, bvals, bvecs, _ = tensor_series()
    dwi = cs.DWI(data, (bvals, bvecs), affine=np.eye(4))
    assert dwi.mask_source == "auto-b0"
    t = dwi.tensor
    assert dwi.tensor is t
    dwi.qc()
    assert dwi.tensor is t  # qc reused the cached fit
    m = np.zeros(data.shape[:3], bool)
    m[1:4, 1:4, 1:3] = True
    other = dwi.with_mask(m)
    assert other.data is dwi.data
    assert other.mask_source == "provided"
    assert other.tensor is not t
    assert other.tensor.mask.sum() == m.sum()


def test_from_files_roundtrip(tmp_path):
    nib = pytest.importorskip("nibabel")
    data, bvals, bvecs, _ = tensor_series()
    aff = np.diag([1.5, 1.5, 2.0, 1.0])
    nib.Nifti1Image(data, aff).to_filename(tmp_path / "dwi.nii.gz")
    nib.Nifti1Image(np.ones(data.shape[:3], np.uint8), aff).to_filename(tmp_path / "mask.nii.gz")
    np.savetxt(tmp_path / "dwi.bval", bvals[None])
    np.savetxt(tmp_path / "dwi.bvec", bvecs.T)
    dwi = cs.DWI.from_files(tmp_path / "dwi.nii.gz", tmp_path / "dwi.bval", tmp_path / "dwi.bvec",
                            mask=tmp_path / "mask.nii.gz")
    assert dwi.zooms == pytest.approx((1.5, 1.5, 2.0))
    assert dwi.mask.all()
    assert np.allclose(dwi.gtab.bvecs, bvecs)
