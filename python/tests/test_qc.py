import json

import numpy as np
import pytest

import cs_dmri as cs
from cs_dmri import _cs_dmri

from _dipy_reference import dwi_contrast as ref_contrast
from _dipy_reference import find_qspace_contrast as ref_contrast_pairs
from conftest import fiber_series


def test_ndc_matches_dipy(series):
    dipy_qc = pytest.importorskip("dipy.stats.qc")
    from dipy.core.gradients import gradient_table

    data, bvals, bvecs = series
    mask = np.zeros(data.shape[:3], bool)
    mask[2:10, 2:10, 1:7] = True
    g = gradient_table(bvals, bvecs=bvecs)
    for m in (None, mask):
        ours = cs.qc.neighboring_dwi_correlation(data, cs.GradientTable(bvals, bvecs), m)
        theirs = dipy_qc.neighboring_dwi_correlation(data, g, mask=m)
        assert ours == pytest.approx(theirs, abs=1e-6)


def test_contrast_matches_pr_4224(series):
    data, bvals, bvecs = series
    gtab = cs.GradientTable(bvals, bvecs)
    pairs = _cs_dmri.qc_neighbor_pairs(bvals, bvecs)
    contrast = _cs_dmri.qc_contrast_pairs(bvals, bvecs)
    assert contrast == [(int(i), int(c)) for i, c in ref_contrast_pairs(bvals, bvecs)]
    expect = ref_contrast(data.astype(np.float64), pairs, contrast)
    assert cs.qc.dwi_contrast_ratio(data, gtab) == pytest.approx(expect, abs=1e-6)


def test_repeats_may_pair():
    bvals = np.array([1000.0] * 3)
    bvecs = np.array([[1, 0, 0], [0.9, 0.43589, 0], [-1, 0, 0]])
    pairs = _cs_dmri.qc_neighbor_pairs(bvals, bvecs)
    assert pairs[0] == (0, 2) and pairs[2] == (2, 0)


def test_ndc_with_repeats_matches_dipy(series):
    dipy_qc = pytest.importorskip("dipy.stats.qc")
    from dipy.core.gradients import gradient_table

    data, bvals, bvecs = series
    data = np.concatenate([data, data[..., 1:] * 1.01], axis=-1)  # a repeated run
    bvals = np.concatenate([bvals, bvals[1:]])
    bvecs = np.vstack([bvecs, -bvecs[1:]])
    ours = cs.qc.neighboring_dwi_correlation(data, (bvals, bvecs))
    theirs = dipy_qc.neighboring_dwi_correlation(data, gradient_table(bvals, bvecs=bvecs))
    assert ours == pytest.approx(theirs, abs=1e-6)


def test_ndc_is_order_invariant(series):
    data, bvals, bvecs = series
    order = np.random.default_rng(0).permutation(len(bvals))
    a = cs.qc.neighboring_dwi_correlation(data, (bvals, bvecs))
    b = cs.qc.neighboring_dwi_correlation(data[..., order], (bvals[order], bvecs[order]))
    assert a == pytest.approx(b, abs=1e-12)


def test_outlier_slices_find_dropout():
    data, bvals, bvecs = fiber_series(shape=(24, 24, 16))
    # smooth anatomy along z so adjacent slices predict each other
    z = np.linspace(0, 1, 16)[None, None, :, None]
    data = data * (1 + 0.2 * z)
    clean, _ = cs.qc.outlier_slices(data, min_voxels=50)
    data[:, :, 7, 5] *= 0.3
    flags, ratio = cs.qc.outlier_slices(data, min_voxels=50)
    assert flags[5, 7]
    assert ratio.shape == (len(bvals), 16)
    assert flags.sum() - clean.sum() >= 1


def test_report_row_and_dictionary(series):
    data, bvals, bvecs = series
    dwi = cs.DWI(data, (bvals, bvecs), affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    r = dwi.qc()
    row = r.to_dict(prefix="raw_")
    assert list(row) == ["raw_" + c for c in cs.qc.columns()]
    assert row["raw_voxel_size_x"] == pytest.approx(2.0)
    assert row["raw_n_dwi_volumes"] == 30 and row["raw_n_b0_volumes"] == 1
    assert r.ndc > 0.9 and r.contrast_grade == "good"
    assert 0.0 <= r.fixel_coherence <= 1.0
    desc = cs.QCReport.column_descriptions(prefix="raw_")
    assert set(desc) == set(row)
    assert desc["raw_ndc"]["Replaces"] == "raw_neighbor_corr"
    assert "Units" in desc["raw_max_b"]
    json.dumps(desc)


def test_bad_inputs_raise(series):
    data, bvals, bvecs = series
    with pytest.raises(ValueError, match="volumes"):
        cs.qc.neighboring_dwi_correlation(data[..., :5], (bvals, bvecs))
    with pytest.raises(ValueError, match="mask shape"):
        cs.qc.neighboring_dwi_correlation(data, (bvals, bvecs), np.ones((2, 2, 2), bool))
