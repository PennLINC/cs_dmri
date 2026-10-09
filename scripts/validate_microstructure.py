#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cross-validate cs_dmri's BrainSuiteSHORE microstructure scalars against
analytical multi-tensor truth and dipy iso-MAPMRI on the same simulated data.

Run inside the qsirecon conda env::

    mamba run -n qsirecon python scripts/validate_microstructure.py

The script:

1. Synthesizes a 4D DWI volume where each voxel has a known single-tensor
   propagator (varying parallel/perpendicular diffusivities), all aligned to
   the +x axis.
2. Writes the DWI as NIfTI + bval/bvec.
3. Runs ``cs-fit`` (L2 Tikhonov for speed) and ``cs-odf --microstructure``.
4. Loads the resulting scalar NIfTIs.
5. Fits dipy iso-MAPMRI on the same voxels.
6. Plots BrainSuite vs analytical (primary reference) and BrainSuite vs dipy
   iso-MAPMRI (secondary), reports Pearson r and median absolute relative
   error per scalar.

Outputs:

* ``cs_dmri/tests/figures/microstructure_validation.png``
* ``cs_dmri/tests/figures/microstructure_validation.csv``
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.mapmri import MapmriModel
from dipy.sims.voxel import single_tensor
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def fibonacci_directions(n: int) -> np.ndarray:
    """Spherical Fibonacci spiral — well-spread, avoids the poles."""
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (np.arange(n) + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    a = phi * np.arange(n)
    return np.stack([np.cos(a) * r, np.sin(a) * r, z], axis=1)


def build_gtab(
    *,
    bvals: Iterable[float],
    n_per_shell: int,
    big_delta: float,
    small_delta: float,
):
    """Multi-shell gradient table: 1 b0 + ``n_per_shell`` directions per shell."""
    bv_list = [0.0]
    bvec_list = [[0.0, 0.0, 0.0]]
    for b in bvals:
        dirs = fibonacci_directions(n_per_shell)
        bv_list.extend([float(b)] * n_per_shell)
        bvec_list.extend(dirs.tolist())
    bv_arr = np.asarray(bv_list, dtype=np.float64)
    bvec_arr = np.asarray(bvec_list, dtype=np.float64)
    return bv_arr, bvec_arr, big_delta, small_delta


# ---------------------------------------------------------------------------
# Ground truth (matches dipy/reconst/tests/test_mapmri.py:585-593)
# ---------------------------------------------------------------------------

@dataclass
class TruthScalars:
    rtop: float
    rtap: float
    rtpp: float
    msd: float
    qiv: float
    ng: float


def analytical_truth(l1: float, l2: float, l3: float, tau: float) -> TruthScalars:
    rtpp = 1.0 / (2.0 * np.sqrt(np.pi * l1 * tau))
    rtap = (
        1.0 / (2.0 * np.sqrt(np.pi * l2 * tau))
        * 1.0 / (2.0 * np.sqrt(np.pi * l3 * tau))
    )
    rtop = rtpp * rtap
    msd = 2.0 * (l1 + l2 + l3) * tau
    qiv = (64.0 * np.pi ** (7 / 2.0) * (l1 * l2 * l3 * tau ** 3) ** (3 / 2.0)) / (
        (l2 * l3 + l1 * (l2 + l3)) * tau ** 2
    )
    # NG is the propagator non-Gaussianity. For a *single* Gaussian it's 0.
    ng = 0.0
    return TruthScalars(rtop=rtop, rtap=rtap, rtpp=rtpp, msd=msd, qiv=qiv, ng=ng)


# ---------------------------------------------------------------------------
# Voxel grid: a sweep of (l1, l2) covering single-fiber regimes
# ---------------------------------------------------------------------------

def build_voxel_grid(
    *,
    l1_values: np.ndarray,
    l2_values: np.ndarray,
):
    """Return per-voxel (l1, l2, l3) triples, principal direction always +x."""
    voxels = []
    for l1 in l1_values:
        for l2 in l2_values:
            l3 = l2  # axially symmetric
            voxels.append((float(l1), float(l2), float(l3)))
    return voxels


def synthesize_dwi(
    voxels,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    *,
    s0: float = 1.0,
):
    n_voxels = len(voxels)
    nx = n_voxels
    ny = nz = 1
    n_dirs = bvals.size
    data = np.zeros((nx, ny, nz, n_dirs), dtype=np.float32)
    gtab = gradient_table(bvals, bvecs=bvecs)
    for i, (l1, l2, l3) in enumerate(voxels):
        evals = np.array([l1, l2, l3])
        # single_tensor synthesizes E(q) such that S(b=0) = S0; the principal
        # eigenvector defaults to +x when evecs is None and evals is rank-1
        # ordered. Confirm by passing evecs explicitly.
        evecs = np.eye(3)
        S = single_tensor(gtab, S0=s0, evals=evals, evecs=evecs, snr=None)
        data[i, 0, 0, :] = S.astype(np.float32)
    return data, gtab


# ---------------------------------------------------------------------------
# cs_dmri pipeline runners
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def run_cs_dmri(
    *,
    cs_dmri_root: Path,
    workdir: Path,
    dwi_path: Path,
    bval_path: Path,
    bvec_path: Path,
    big_delta: float,
    small_delta: float,
    radial_order: int,
    zeta: float,
):
    """Run cs-fit and cs-odf --microstructure, return path to ODX output."""
    cs_fit = cs_dmri_root / "target" / "release" / "cs-fit"
    cs_odf = cs_dmri_root / "target" / "release" / "cs-odf"

    if not cs_fit.exists():
        run(["cargo", "build", "--release", "--bin", "cs-fit"], cwd=cs_dmri_root)
    if not cs_odf.exists():
        run(["cargo", "build", "--release", "--bin", "cs-odf"], cwd=cs_dmri_root)

    coeffs_path = workdir / "coeffs.nii.gz"
    odx_path = workdir / "out.odx"

    run([
        cs_fit,
        "--dwi", dwi_path,
        "--bval", bval_path,
        "--bvec", bvec_path,
        "--big-delta", str(big_delta),
        "--small-delta", str(small_delta),
        "--radial-order", str(radial_order),
        "--zeta", str(zeta),
        "--reg", "l2",  # closed-form Tikhonov — fast, deterministic
        "--output", coeffs_path,
        "--no-bvec-rotation",  # keep image-axis bvecs (matches our synthetic data)
    ])
    # Use mm units so the comparison against analytical truth (computed in
    # mm-based dipy convention) is apples-to-apples. cs-odf's user-facing
    # default is `um` (TORTOISE-style); for the validation we want raw
    # internal-convention values.
    run([
        cs_odf,
        "--coeffs", coeffs_path,
        "--output", odx_path,
        "--microstructure",
        "--scalar-units", "mm",
    ])
    return coeffs_path, odx_path


def load_scalar_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.dataobj).astype(np.float64)


# ---------------------------------------------------------------------------
# dipy iso-MAPMRI reference fit
# ---------------------------------------------------------------------------

def fit_dipy_iso_mapmri(data, gtab, big_delta, small_delta, radial_order):
    """Fit dipy iso-MAPMRI per voxel and return a dict of scalar maps.

    NG is intentionally omitted: dipy disallows ``ng()`` in iso mode (the iso
    basis lacks a tensor-aligned (0,0,0) mode, so the iso NG estimate is
    *anisotropy-biased* rather than non-Gaussianity-biased and the dipy authors
    chose to fail-fast). cs_dmri reports the same iso-frame NG for inspection
    but it should be interpreted as "departure from isotropic Gaussian", not
    propagator non-Gaussianity in the strict sense.
    """
    bvals = gtab.bvals
    bvecs = gtab.bvecs
    gtab2 = gradient_table(
        bvals,
        bvecs=bvecs,
        big_delta=big_delta,
        small_delta=small_delta,
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MapmriModel(
            gtab2,
            radial_order=radial_order,
            laplacian_regularization=False,
            anisotropic_scaling=False,
            bval_threshold=np.inf,
        )
        fit = model.fit(data)
        out = {
            "rtop": fit.rtop(),
            "rtap": fit.rtap(),
            "rtpp": fit.rtpp(),
            "msd": fit.msd(),
            "qiv": fit.qiv(),
        }
    return out


# ---------------------------------------------------------------------------
# Plot + report
# ---------------------------------------------------------------------------

def correlate_and_summarize(name, x, y, *, log=False):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return float("nan"), float("nan"), x, y
    r, _ = pearsonr(x, y)
    rel_err = np.abs(y - x) / np.maximum(np.abs(x), 1e-30)
    median_rel = float(np.median(rel_err))
    return float(r), median_rel, x, y


def plot_scatter(ax, x, y, label, color, log=False):
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.scatter(x, y, s=14, alpha=0.7, color=color, label=label, edgecolors="none")
    if x.size:
        lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
        hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
        if log and lo > 0 and hi > 0:
            xs = np.array([lo, hi])
        else:
            xs = np.array([lo, hi])
        ax.plot(xs, xs, "k--", linewidth=0.8, alpha=0.6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cs-dmri-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "figures",
    )
    parser.add_argument("--radial-order", type=int, default=6)
    parser.add_argument("--zeta", type=float, default=700.0)
    parser.add_argument("--big-delta", type=float, default=0.0431)
    parser.add_argument("--small-delta", type=float, default=0.0107)
    parser.add_argument("--n-per-shell", type=int, default=64)
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="leave the synthetic DWI / coefficient files for inspection",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # b-values matching a typical cs_dmri input.
    bvals_shells = [1000.0, 2000.0, 3000.0, 4000.0]
    bvals, bvecs, big_delta, small_delta = build_gtab(
        bvals=bvals_shells,
        n_per_shell=args.n_per_shell,
        big_delta=args.big_delta,
        small_delta=args.small_delta,
    )
    tau = big_delta - small_delta / 3.0
    print(f"τ = Δ - δ/3 = {tau:.5e} s")

    # Voxel sweep: parallel and perpendicular diffusivities.
    l1_values = np.linspace(1.0e-3, 2.0e-3, 6)
    l2_values = np.linspace(0.2e-3, 0.6e-3, 6)
    voxels = build_voxel_grid(l1_values=l1_values, l2_values=l2_values)
    n_voxels = len(voxels)
    print(f"Sweeping {n_voxels} single-fiber voxels (l1 ∈ {l1_values.min():.2e}..{l1_values.max():.2e}, "
          f"l2 ∈ {l2_values.min():.2e}..{l2_values.max():.2e})")

    # Truth.
    truths = [analytical_truth(*v, tau=tau) for v in voxels]
    truth_arr = {
        "rtop": np.array([t.rtop for t in truths]),
        "rtap": np.array([t.rtap for t in truths]),
        "rtpp": np.array([t.rtpp for t in truths]),
        "msd": np.array([t.msd for t in truths]),
        "qiv": np.array([t.qiv for t in truths]),
        "ng": np.array([t.ng for t in truths]),
    }

    # Synthesize DWI.
    data, gtab = synthesize_dwi(voxels, bvals, bvecs)
    print(f"DWI shape {data.shape}, S0={data[0, 0, 0, 0]:.3f}")

    workdir = Path(tempfile.mkdtemp(prefix="cs_dmri_validate_"))
    print(f"workdir: {workdir}")
    try:
        # Write NIfTI + bval/bvec.
        affine = np.eye(4, dtype=np.float64)
        dwi_path = workdir / "dwi.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(dwi_path))
        np.savetxt(workdir / "dwi.bval", bvals.reshape(1, -1), fmt="%.4f")
        np.savetxt(workdir / "dwi.bvec", bvecs.T, fmt="%.6f")

        # Run cs_dmri.
        coeffs_path, odx_path = run_cs_dmri(
            cs_dmri_root=args.cs_dmri_root,
            workdir=workdir,
            dwi_path=dwi_path,
            bval_path=workdir / "dwi.bval",
            bvec_path=workdir / "dwi.bvec",
            big_delta=big_delta,
            small_delta=small_delta,
            radial_order=args.radial_order,
            zeta=args.zeta,
        )

        # Load scalar NIfTIs.
        bs = {}
        for name in ("rtop", "rtap", "rtpp", "msd", "qiv", "ng"):
            stem = str(odx_path).rstrip(".odx")
            scalar_path = workdir / f"out_{name}.nii.gz"
            if not scalar_path.exists():
                print(f"WARNING: missing {scalar_path}", file=sys.stderr)
                continue
            vol = load_scalar_volume(scalar_path)
            bs[name] = vol[:, 0, 0]  # 1×ny×nz with ny=nz=1
            print(f"  loaded {name}: shape {vol.shape}, range [{vol.min():.3e}, {vol.max():.3e}]")

        # dipy reference.
        print("Fitting dipy iso-MAPMRI…")
        # MapmriModel expects (n,) or (..., n_dirs); we pass (n_voxels, n_dirs).
        flat = data[:, 0, 0, :]
        dipy_out = fit_dipy_iso_mapmri(
            flat, gtab, big_delta, small_delta, args.radial_order
        )

        # Aggregate + correlate. NG omitted from cross-comparison (dipy iso
        # disallows it; the BS value is reported but not validated against a
        # reference here).
        rows = []
        scalars = ("rtop", "rtap", "rtpp", "msd", "qiv")
        for s in scalars:
            t = truth_arr[s]
            b = bs.get(s)
            d = np.asarray(dipy_out[s]).ravel()
            if b is None:
                continue
            r_bs_truth, mre_bs_truth, *_ = correlate_and_summarize(s, t, b)
            r_dipy_truth, mre_dipy_truth, *_ = correlate_and_summarize(s, t, d)
            r_bs_dipy, mre_bs_dipy, *_ = correlate_and_summarize(s, d, b)
            rows.append({
                "scalar": s,
                "truth_min": float(np.nanmin(t)),
                "truth_max": float(np.nanmax(t)),
                "bs_min": float(np.nanmin(b)),
                "bs_max": float(np.nanmax(b)),
                "dipy_min": float(np.nanmin(d)),
                "dipy_max": float(np.nanmax(d)),
                "r_bs_vs_truth": r_bs_truth,
                "med_rel_err_bs_vs_truth": mre_bs_truth,
                "r_dipy_vs_truth": r_dipy_truth,
                "med_rel_err_dipy_vs_truth": mre_dipy_truth,
                "r_bs_vs_dipy": r_bs_dipy,
                "med_rel_err_bs_vs_dipy": mre_bs_dipy,
            })

        # Print table.
        print("\n=== microstructure scalar validation ===")
        header = (
            "scalar  | r(BS,truth)  med|relerr|  | r(dipy,truth) med|relerr|  | "
            "r(BS,dipy)  med|relerr|"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['scalar']:<7s} | "
                f"{row['r_bs_vs_truth']:+.4f}      {row['med_rel_err_bs_vs_truth']:.3f}     | "
                f"{row['r_dipy_vs_truth']:+.4f}       {row['med_rel_err_dipy_vs_truth']:.3f}     | "
                f"{row['r_bs_vs_dipy']:+.4f}      {row['med_rel_err_bs_vs_dipy']:.3f}"
            )

        # CSV.
        csv_path = args.out_dir / "microstructure_validation.csv"
        with open(csv_path, "w") as fh:
            keys = list(rows[0].keys()) if rows else []
            fh.write(",".join(keys) + "\n")
            for row in rows:
                fh.write(",".join(str(row[k]) for k in keys) + "\n")
        print(f"wrote {csv_path}")

        # Plot.
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, s in zip(axes.ravel()[: len(scalars)], scalars):
            t = truth_arr[s]
            b = bs.get(s)
            d = np.asarray(dipy_out[s]).ravel() if s in dipy_out else None
            if b is None:
                ax.set_visible(False)
                continue
            log = s in ("rtop", "rtap", "rtpp", "qiv")  # span many decades
            plot_scatter(ax, t, b, "BS vs truth", "C0", log=log)
            if d is not None:
                plot_scatter(ax, t, d, "dipy vs truth", "C1", log=log)
            r_bs, mre_bs, *_ = correlate_and_summarize(s, t, b)
            title = f"{s.upper()}  r(BS)={r_bs:.3f} mre={mre_bs:.2f}"
            if d is not None:
                r_dipy, mre_dipy, *_ = correlate_and_summarize(s, t, d)
                title += f" | r(dipy)={r_dipy:.3f} mre={mre_dipy:.2f}"
            ax.set_title(title)
            ax.set_xlabel("analytical truth")
            ax.set_ylabel("estimate")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # NG panel: scatter against ground-truth FA. For a single Gaussian, the
        # strict propagator NG is 0 — but BrainSuiteSHORE's iso-frame NG instead
        # measures departure-from-isotropic-Gaussian, which scales monotonically
        # with FA. Plot it that way so the panel is informative.
        ng_ax = axes.ravel()[len(scalars)]
        if "ng" in bs:
            fas = []
            for (l1, l2, l3) in voxels:
                md = (l1 + l2 + l3) / 3.0
                num = (l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2
                den = l1 ** 2 + l2 ** 2 + l3 ** 2
                fas.append(np.sqrt(1.5 * num / den))
            fas = np.asarray(fas)
            ng_ax.scatter(fas, bs["ng"], color="C0", s=14, alpha=0.7,
                          label="BS iso-frame NG")
            r_ng_fa, _ = pearsonr(fas, bs["ng"])
            ng_ax.set_title(
                f"NG vs FA  r(BS, FA)={r_ng_fa:+.3f}\n"
                "(iso-frame NG ≠ strict propagator NG — see math doc)"
            )
            ng_ax.set_xlabel("ground-truth FA")
            ng_ax.set_ylabel("BS NG")
            ng_ax.legend(fontsize=8)
            ng_ax.grid(True, alpha=0.3)
        else:
            ng_ax.set_visible(False)
        fig.suptitle(
            f"BrainSuiteSHORE microstructure validation "
            f"(N={n_voxels} single-fiber voxels, radial_order={args.radial_order}, "
            f"ζ={args.zeta})"
        )
        fig.tight_layout()
        png_path = args.out_dir / "microstructure_validation.png"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        print(f"wrote {png_path}")

    finally:
        if not args.keep_tmp:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
            print(f"removed {workdir}")
        else:
            print(f"kept {workdir} for inspection (--keep-tmp)")


if __name__ == "__main__":
    main()
