"""Loading a series: values, axes and memory layout.

``DWI.from_files`` streams a NIfTI through the compiled reader into C-ordered
float32, and other images go through nibabel a block at a time. Both must give
the values nibabel gives, at the same (x, y, z, t), whatever the on-disk type,
scaling or compression.
"""

import nibabel as nib
import numpy as np
import pytest

import cs_dmri as cs
from cs_dmri import _cs_dmri

SHAPE = (5, 6, 7, 37)  # distinct sizes, and a volume count that leaves a partial block


def coded():
    """Each voxel's value encodes its indices, so a swapped axis is visible."""
    x, y, z, t = np.indices(SHAPE)
    return (x + 10 * y + 100 * z + 1000 * t).astype(np.float64)


def gradients(tmp_path, n=SHAPE[3]):
    bvals = np.r_[0.0, np.full(n - 1, 1000.0)]
    rng = np.random.default_rng(0)
    bvecs = rng.normal(size=(n, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs[0] = 0
    np.savetxt(tmp_path / "dwi.bval", bvals[None])
    np.savetxt(tmp_path / "dwi.bvec", bvecs.T)
    return tmp_path / "dwi.bval", tmp_path / "dwi.bvec"


def save(tmp_path, data, dtype, ext, slope=None, inter=None):
    img = nib.Nifti1Image(data.astype(dtype), np.diag([1.25, 1.25, 1.25, 1.0]))
    if slope is not None:
        img.header.set_slope_inter(slope, inter)
    path = tmp_path / f"dwi{ext}"
    nib.save(img, path)
    return path


@pytest.mark.parametrize("ext", [".nii", ".nii.gz"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int16, np.uint16])
def test_from_files_matches_nibabel(tmp_path, ext, dtype):
    path = save(tmp_path, coded() % 30000, dtype, ext)
    dwi = cs.DWI.from_files(path, *gradients(tmp_path))
    assert dwi.data.dtype == np.float32
    assert dwi.data.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(dwi.data, nib.load(path).get_fdata(dtype=np.float32))


@pytest.mark.parametrize("ext", [".nii", ".nii.gz"])
def test_scaled_integers_match_nibabel(tmp_path, ext):
    path = save(tmp_path, coded() % 30000, np.int16, ext, slope=0.5, inter=-3.0)
    nib_img = nib.load(path)
    assert nib_img.dataobj.slope == 0.5
    dwi = cs.DWI.from_files(path, *gradients(tmp_path))
    np.testing.assert_allclose(dwi.data, nib_img.get_fdata(dtype=np.float32), rtol=1e-6)


def test_axes_are_kept(tmp_path):
    path = save(tmp_path, coded(), np.float64, ".nii.gz")
    data = _cs_dmri.read_dwi_volume(str(path))
    assert data.shape == SHAPE and data.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(data, coded())


def test_in_memory_image_uses_nibabel(tmp_path):
    img = nib.Nifti1Image(coded(), np.eye(4))
    dwi = cs.DWI.from_nibabel(img, cs.GradientTable.from_files(*gradients(tmp_path)))
    assert dwi.data.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(dwi.data, coded())


def test_fortran_ordered_input_is_stored_c_ordered(tmp_path):
    data = np.asfortranarray(coded(), dtype=np.float32)
    dwi = cs.DWI(data, cs.GradientTable.from_files(*gradients(tmp_path)))
    assert dwi.data.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(dwi.data, data)


def test_c_ordered_float32_input_is_not_copied(tmp_path):
    data = np.ascontiguousarray(coded(), dtype=np.float32)
    dwi = cs.DWI(data, cs.GradientTable.from_files(*gradients(tmp_path)))
    assert np.shares_memory(dwi.data, data)


def test_fits_do_not_depend_on_input_layout(series):
    """The bindings borrow a C-ordered volume and copy any other layout into
    that order, so both give the same fit."""
    data, bvals, bvecs = series
    gtab = cs.GradientTable(bvals, bvecs)
    mask = np.ones(data.shape[:3], bool)
    c = cs.RestoreModel(gtab).fit(np.ascontiguousarray(data), mask=mask)
    f = cs.RestoreModel(gtab).fit(np.asfortranarray(data), mask=mask)
    np.testing.assert_array_equal(c.fa, f.fa)
