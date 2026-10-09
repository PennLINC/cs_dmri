"""Python models reproduce the CLI binaries bit for bit.

Local-only: needs the release binaries (``cargo build --release``) and the
qsiprep test outputs under ``<repo>/../qsiprep_testing``; skipped otherwise.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

import cs_dmri as cs

nib = pytest.importorskip("nibabel")

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "target" / "release"
PNC = REPO.parent / "qsiprep_testing/DSDTI_nofmap/derivatives/qsiprep/sub-PNC/dwi/sub-PNC_acq-realistic_space-T1w_desc-"

pytestmark = pytest.mark.skipif(
    not (BIN / "cs-dti").exists() or not Path(str(PNC) + "preproc_dwi.nii.gz").exists(),
    reason="needs release binaries and ../qsiprep_testing",
)


def run(*args):
    subprocess.run([str(a) for a in args], check=True, capture_output=True)


def load(p):
    return np.asarray(nib.load(p).dataobj)


@pytest.fixture(scope="module")
def pnc():
    return cs.DWI.from_files(f"{PNC}preproc_dwi.nii.gz", f"{PNC}preproc_dwi.bval", f"{PNC}preproc_dwi.bvec",
                             mask=f"{PNC}brain_mask.nii.gz")


def common(tmp):
    return ["--dwi", f"{PNC}preproc_dwi.nii.gz", "--bval", f"{PNC}preproc_dwi.bval", "--bvec",
            f"{PNC}preproc_dwi.bvec", "--mask", f"{PNC}brain_mask.nii.gz", "--quiet", "--threads", "4"]


def test_restore_matches_cs_dti(pnc, tmp_path):
    out = {k: tmp_path / f"{k}.nii.gz" for k in ("fa", "md", "s0", "of")}
    run(BIN / "cs-dti", *common(tmp_path), "--output-fa", out["fa"], "--output-md", out["md"],
        "--output-s0", out["s0"], "--output-outlier-fraction", out["of"])
    t = pnc.tensor
    for name, key in (("fa", "fa"), ("md", "md"), ("s0", "s0"), ("outlier_fraction", "of")):
        np.testing.assert_array_equal(getattr(t, name)[pnc.mask], load(out[key])[pnc.mask])


def test_pipeline_matches_cs_ss3t_full(pnc, tmp_path):
    run(BIN / "cs-ss3t-full", *common(tmp_path), "--output-wm", tmp_path / "wm.nii.gz", "--output-gm",
        tmp_path / "gm.nii.gz", "--output-csf", tmp_path / "csf.nii.gz", "--write-responses-to", tmp_path)
    fit = cs.ss3t_pipeline(pnc)
    np.testing.assert_array_equal(fit.wm[pnc.mask], load(tmp_path / "wm.nii.gz")[pnc.mask])
    np.testing.assert_array_equal(fit.gm[pnc.mask], np.squeeze(load(tmp_path / "gm.nii.gz"))[pnc.mask])
    cli = cs.TissueResponse.from_mrtrix_txt(tmp_path / "wm_response.txt")
    np.testing.assert_allclose(fit.responses.wm.coeffs, cli.coeffs, rtol=1e-12)


def test_qc_matches_cs_qc(pnc, tmp_path):
    import json

    run(BIN / "cs-qc", *common(tmp_path), "--output-json", tmp_path / "qc.json")
    cli = json.loads((tmp_path / "qc.json").read_text())["metrics"]
    ours = pnc.qc().to_dict()
    for k, v in ours.items():
        if v is None:
            assert cli[k] is None, k
        else:
            assert v == pytest.approx(cli[k], rel=1e-9, abs=1e-12), k


def test_shore_l2_matches_cs_fit(pnc, tmp_path):
    out = tmp_path / "coef.nii.gz"
    run(BIN / "cs-fit", *common(tmp_path), "--output", out, "--reg", "l2", "--diagnostics")
    fit = cs.ShoreModel(pnc.gtab, regularization="l2").fit(pnc)
    np.testing.assert_array_equal(fit.coefficients[pnc.mask], load(out)[pnc.mask])
    np.testing.assert_array_equal(fit.r2[pnc.mask], load(tmp_path / "coef_r2.nii.gz")[pnc.mask])
