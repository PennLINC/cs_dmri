#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Benchmark cs-dmri fit options on real DWI.

For each (bundle, config) pair: runs cs-fit, then cs-odf, then `odx qc`,
captures wall times, then reads the resulting NIfTIs/JSON to extract
fit-side metrics (R^2, RMSE, sparsity, alpha) and fixel-side QC
(coherence_index, incoherence_index, evaluated_fixels). Writes one
CSV + markdown report per bundle, plus a top-level summary spanning
all bundles (so HASC55-vs-ABCD on the same subject is one row pair away).

Bundle discovery (when `--bundles-root` is given) looks for
`*desc-preproc_dwi.nii(.gz)` under the root and derives the matching
bval/bvec/mask by suffix substitution (BIDS-ish layout). Single-bundle
mode (`--bundle <dwi.nii.gz>`) takes one bundle explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nb
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CS_DMRI = REPO_ROOT / "cs-dmri"
ODX_RS = REPO_ROOT / "odx-rs"

CS_FIT = CS_DMRI / "target" / "release" / "cs-fit"
CS_ODF = CS_DMRI / "target" / "release" / "cs-odf"
CS_SYNTH = CS_DMRI / "target" / "release" / "cs-synth"
ODX_BIN = ODX_RS / "target" / "release" / "odx"

DEFAULT_DWI = REPO_ROOT / "tortoisev4" / "test_data" / "sub-01_ses-1_space-ACPC_desc-preproc_dwi.nii"


@dataclass
class FitConfig:
    name: str
    cs_fit_args: list[str] = field(default_factory=list)


@dataclass
class Bundle:
    name: str
    dwi: Path
    bval: Path
    bvec: Path
    mask: Path


# Trimmed config matrix: anchors + the interesting middle band identified by
# the prior single-subject sweep. Drops catastrophic high-α and the
# n=40/ro8 redundancies. The full matrix lives behind --configs-preset full.
CONFIGS_FOCUSED: list[FitConfig] = [
    FitConfig("l2_baseline",    ["--reg", "l2"]),
    FitConfig("l1_ratio_1e-3",  ["--reg", "l1", "--alpha-mode", "alpha-ratio", "--alpha-ratio", "1e-3"]),
    FitConfig("l1_pathbic_n20", ["--reg", "l1", "--alpha-mode", "path-bic"]),
    FitConfig("l1_pathbic_ro4", ["--reg", "l1", "--alpha-mode", "path-bic", "--radial-order", "4"]),
    FitConfig("l1_pathbic_nn",  ["--reg", "l1", "--alpha-mode", "path-bic", "--non-negative"]),
    # L2-residual-anchored α selector — per voxel, pick the largest α whose
    # RSS is within (1+slack)·RSS_L2. Defaults: slack=0.05, path-eps=1e-4.
    # The recommended-replacement for path-bic on high-b CS-DSI / infants.
    FitConfig("l1_l2anchored_ro4", ["--reg", "l1", "--alpha-mode", "l2-anchored",
                                     "--radial-order", "4"]),
]

CONFIGS_FULL: list[FitConfig] = [
    FitConfig("l2_baseline",    ["--reg", "l2"]),
    FitConfig("l1_fixed_1e-1",  ["--reg", "l1", "--alpha-mode", "fixed", "--alpha", "1e-1"]),
    FitConfig("l1_fixed_1e-2",  ["--reg", "l1", "--alpha-mode", "fixed", "--alpha", "1e-2"]),
    FitConfig("l1_fixed_1e-3",  ["--reg", "l1", "--alpha-mode", "fixed", "--alpha", "1e-3"]),
    FitConfig("l1_fixed_1e-4",  ["--reg", "l1", "--alpha-mode", "fixed", "--alpha", "1e-4"]),
    FitConfig("l1_ratio_1e-2",  ["--reg", "l1", "--alpha-mode", "alpha-ratio", "--alpha-ratio", "1e-2"]),
    FitConfig("l1_ratio_1e-3",  ["--reg", "l1", "--alpha-mode", "alpha-ratio", "--alpha-ratio", "1e-3"]),
    FitConfig("l1_pathbic_n20", ["--reg", "l1", "--alpha-mode", "path-bic"]),
    FitConfig("l1_pathbic_n40", ["--reg", "l1", "--alpha-mode", "path-bic", "--path-n-alphas", "40", "--path-eps", "1e-4"]),
    FitConfig("l1_pathbic_nn",  ["--reg", "l1", "--alpha-mode", "path-bic", "--non-negative"]),
    FitConfig("l1_pathbic_ro4", ["--reg", "l1", "--alpha-mode", "path-bic", "--radial-order", "4"]),
    FitConfig("l1_pathbic_ro8", ["--reg", "l1", "--alpha-mode", "path-bic", "--radial-order", "8"]),
    # L2-residual-anchored α selector — multiple slack settings for the
    # tradeoff sweep (lower slack = closer to L2's ODFs; higher = more
    # L1 sparsity). The 0.05 default is between these two.
    FitConfig("l1_l2anchored_ro4_slack02",
              ["--reg", "l1", "--alpha-mode", "l2-anchored",
               "--radial-order", "4", "--slack", "0.02"]),
    FitConfig("l1_l2anchored_ro4_slack05",
              ["--reg", "l1", "--alpha-mode", "l2-anchored",
               "--radial-order", "4", "--slack", "0.05"]),
    FitConfig("l1_l2anchored_ro4_slack10",
              ["--reg", "l1", "--alpha-mode", "l2-anchored",
               "--radial-order", "4", "--slack", "0.10"]),
    FitConfig("l1_l2anchored_ro6",
              ["--reg", "l1", "--alpha-mode", "l2-anchored", "--radial-order", "6"]),
]


PERCENTILES = (10, 50, 90)


def strip_dwi_ext(path: Path) -> str:
    """Return the file path string with .nii.gz / .nii stripped from the end."""
    s = str(path)
    for ext in (".nii.gz", ".nii"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def derive_siblings(dwi: Path) -> Bundle:
    """Build a Bundle from a `*desc-preproc_dwi.nii(.gz)` path by deriving
    sibling bval/bvec/mask via suffix substitution (BIDS layout).
    """
    stem_no_ext = strip_dwi_ext(dwi)
    bval = Path(stem_no_ext + ".bval")
    bvec = Path(stem_no_ext + ".bvec")

    mask_candidates: list[Path] = []
    if "desc-preproc_dwi" in stem_no_ext:
        mask_stem = stem_no_ext.replace("desc-preproc_dwi", "desc-brain_mask")
        for ext in (".nii.gz", ".nii"):
            mask_candidates.append(Path(mask_stem + ext))
    # Loose fallback: any *desc-brain* sibling in the same directory.
    if not any(p.exists() for p in mask_candidates):
        for cand in dwi.parent.glob("*desc-brain*mask*.nii*"):
            mask_candidates.append(cand)
    mask = next((p for p in mask_candidates if p.exists()), None)
    if mask is None:
        raise FileNotFoundError(
            f"could not locate brain mask next to {dwi}; tried {mask_candidates}"
        )

    name = bundle_name_from_dwi(dwi)
    return Bundle(name=name, dwi=dwi, bval=bval, bvec=bvec, mask=mask)


def bundle_name_from_dwi(dwi: Path) -> str:
    """Extract `sub-XXXX[_ses-Y][_acq-ZZZZ]` from a BIDS-like filename."""
    base = dwi.name
    parts = []
    for key in ("sub-", "ses-", "acq-", "run-"):
        m = re.search(rf"{key}([A-Za-z0-9]+)", base)
        if m:
            parts.append(f"{key}{m.group(1)}")
    if not parts:
        return strip_dwi_ext(dwi).rsplit("/", 1)[-1]
    return "_".join(parts)


def discover_bundles(root: Path) -> list[Bundle]:
    paths = sorted(root.rglob("*desc-preproc_dwi.nii.gz")) + sorted(root.rglob("*desc-preproc_dwi.nii"))
    bundles: list[Bundle] = []
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        try:
            bundles.append(derive_siblings(p))
        except FileNotFoundError as e:
            print(f"[bench] skipping {p.name}: {e}", file=sys.stderr)
    return bundles


def ensure_built(skip: bool) -> None:
    if skip:
        return
    print("[bench] cargo build --release for cs-fit/cs-odf...", flush=True)
    subprocess.run(["cargo", "build", "--release", "--bins"], cwd=CS_DMRI, check=True)
    print("[bench] cargo build --release for odx...", flush=True)
    subprocess.run(["cargo", "build", "--release", "--bin", "odx"], cwd=ODX_RS, check=True)
    for p in (CS_FIT, CS_ODF, CS_SYNTH, ODX_BIN):
        if not p.exists():
            sys.exit(f"[bench] expected binary missing after build: {p}")


def run_capture(cmd: list[str], log_path: Path) -> tuple[float, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bench] $ {' '.join(shlex.quote(str(c)) for c in cmd)}", flush=True)
    t0 = time.perf_counter()
    with log_path.open("wb") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return time.perf_counter() - t0, proc.returncode


def sibling(coeffs_path: Path, suffix: str) -> Path:
    return Path(strip_dwi_ext(coeffs_path) + suffix)


def percentile_set(values: np.ndarray, prefix: str) -> dict:
    if values.size == 0:
        return {f"{prefix}_p{p}": None for p in PERCENTILES}
    return {f"{prefix}_p{p}": float(np.percentile(values, p)) for p in PERCENTILES}


def _strip_acq_run(name: str) -> str:
    """Drop `_acq-*` and `_run-*` tokens so HASC and ABCD share a key."""
    return "_".join(
        p for p in name.split("_")
        if not (p.startswith("acq-") or p.startswith("run-"))
    )


def _acq_slot(name: str) -> str:
    """Classify an `acq-*` token as 'HASC' / 'ABCD' / its raw value / ''."""
    m = re.search(r"acq-([A-Za-z0-9]+)", name)
    if not m:
        return ""
    acq = m.group(1).upper()
    if acq == "ABCD":
        return "ABCD"
    if acq.startswith("HASC"):
        return "HASC"
    return acq


def pair_hasc_to_abcd(bundles: list[Bundle]) -> dict[str, Bundle]:
    """Map HASC bundle.name -> matching ABCD Bundle for the same subject+session.

    Strips `_acq-*` and `_run-*` to derive the (sub, ses) key. A pair exists
    only when both an HASC* and an ABCD bundle share that key. Subjects with
    only one acquisition do not appear in the result.
    """
    by_key: dict[str, dict[str, Bundle]] = {}
    for b in bundles:
        slot = _acq_slot(b.name)
        if slot not in ("HASC", "ABCD"):
            continue
        key = _strip_acq_run(b.name)
        by_key.setdefault(key, {})[slot] = b

    pairs: dict[str, Bundle] = {}
    for slots in by_key.values():
        if "HASC" in slots and "ABCD" in slots:
            pairs[slots["HASC"].name] = slots["ABCD"]
    return pairs


def parse_bvals_file(path: Path) -> np.ndarray:
    return np.array([float(t) for t in path.read_text().split()], dtype=np.float64)


def qvals_per_grad(bvals: np.ndarray, big_delta: float, small_delta: float) -> np.ndarray:
    """q-magnitude per gradient: sqrt(b / (4π² · (Δ − δ/3)))."""
    tau = big_delta - small_delta / 3.0
    return np.sqrt(np.maximum(bvals, 0.0) / (4.0 * np.pi * np.pi * tau))


def per_voxel_r2_rmse(measured: np.ndarray, predicted: np.ndarray,
                      mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel R² and RMSE across the gradient axis, restricted to mask voxels.

    Returns 1D arrays of length equal to `mask.sum()`. Voxels with SS_tot ≈ 0
    yield NaN R² (filtered out by the caller); RMSE remains finite.
    """
    m = measured[mask].astype(np.float64)
    p = predicted[mask].astype(np.float64)
    n_g = m.shape[1]
    diff = m - p
    ss_res = np.einsum("vg,vg->v", diff, diff)
    centered = m - m.mean(axis=1, keepdims=True)
    ss_tot = np.einsum("vg,vg->v", centered, centered)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)
    rmse = np.sqrt(ss_res / n_g)
    return r2, rmse


def run_cross_prediction(cfg: FitConfig, hasc_bundle: Bundle,
                         abcd_bundle: Bundle, cfg_dir: Path,
                         threads: int, force: bool) -> dict:
    """Synthesize ABCD-scheme signal from HASC coefficients, then compare
    per-voxel against measured ABCD inside the intersection of brain masks."""
    coeffs = cfg_dir / "coeffs.nii.gz"
    coeffs_json = Path(strip_dwi_ext(coeffs) + ".json")
    cross_dir = cfg_dir / "cross"
    cross_dir.mkdir(parents=True, exist_ok=True)
    predicted = cross_dir / "predicted.nii.gz"
    cross_metrics_path = cross_dir / "cross_metrics.json"
    r2_vol_path = cross_dir / "r2.nii.gz"
    r2_clipped_vol_path = cross_dir / "r2_clipped.nii.gz"
    rmse_vol_path = cross_dir / "rmse.nii.gz"
    synth_log = cross_dir / "cs-synth.stderr.log"

    out: dict = {"cross_target": abcd_bundle.name}

    sidecar = json.loads(coeffs_json.read_text()) if coeffs_json.exists() else {}
    big = sidecar.get("big_delta_seconds")
    small = sidecar.get("small_delta_seconds")

    bvals_h = parse_bvals_file(hasc_bundle.bval)
    bvals_a = parse_bvals_file(abcd_bundle.bval)
    if big is not None and small is not None and bvals_h.size and bvals_a.size:
        q_h = qvals_per_grad(bvals_h, big, small)
        q_a = qvals_per_grad(bvals_a, big, small)
        out["cross_max_q_hasc_inv_mm"] = float(q_h.max())
        out["cross_max_q_abcd_inv_mm"] = float(q_a.max())
        extrap = float((q_a > q_h.max()).mean())
        out["cross_q_extrap_fraction"] = extrap
        if extrap > 0.10:
            print(
                f"[bench] WARN {hasc_bundle.name}/{cfg.name}: "
                f"{extrap:.1%} of ABCD q-magnitudes exceed HASC max q (extrapolation)",
                file=sys.stderr,
            )

    if not predicted.exists() or force:
        cmd = [
            str(CS_SYNTH),
            "--coeffs", str(coeffs),
            "--bval", str(abcd_bundle.bval),
            "--bvec", str(abcd_bundle.bvec),
            "--output", str(predicted),
            "--threads", str(threads),
        ]
        wall, rc = run_capture(cmd, synth_log)
        out["cross_synth_wall_s"] = wall
        if rc != 0:
            out["cross_error"] = f"cs-synth rc={rc}"
            return out
    else:
        print(f"[bench] {hasc_bundle.name}/{cfg.name}: cross prediction exists, skipping cs-synth")

    volumes_present = (
        r2_vol_path.exists() and r2_clipped_vol_path.exists() and rmse_vol_path.exists()
    )
    if cross_metrics_path.exists() and volumes_present and not force:
        out.update(json.loads(cross_metrics_path.read_text()))
        return out

    measured_img = nb.load(str(abcd_bundle.dwi))
    predicted_img = nb.load(str(predicted))
    if measured_img.shape != predicted_img.shape:
        out["cross_error"] = (
            f"shape_mismatch measured={measured_img.shape} predicted={predicted_img.shape}"
        )
        return out

    mask_h = np.asarray(nb.load(str(hasc_bundle.mask)).dataobj) > 0
    mask_a = np.asarray(nb.load(str(abcd_bundle.mask)).dataobj) > 0
    if mask_h.shape != mask_a.shape or mask_h.shape != measured_img.shape[:3]:
        out["cross_error"] = "mask_shape_mismatch"
        return out
    mask = mask_h & mask_a

    measured = np.asarray(measured_img.get_fdata(dtype=np.float32))
    pred = np.asarray(predicted_img.get_fdata(dtype=np.float32))
    r2, rmse = per_voxel_r2_rmse(measured, pred, mask)
    finite = np.isfinite(r2)
    r2_f = r2[finite]
    rmse_f = rmse[finite]
    # Tail voxels (low-variance CSF / mask edge) can produce arbitrarily
    # negative R² that drown the raw mean. The clipped mean floors per-voxel
    # R² at 0 before averaging — comparable across configs and bounded in
    # [0, 1]. Always report the unclipped mean too, since a low clipped mean
    # alongside a healthy median signals tail-voxel pathology specifically.
    r2_clipped = np.maximum(r2_f, 0.0)
    metrics: dict = {
        "cross_evaluated_voxels": int(finite.sum()),
        "cross_r2_mean": float(r2_f.mean()) if r2_f.size else None,
        "cross_r2_clipped_mean": float(r2_clipped.mean()) if r2_clipped.size else None,
        "cross_rmse_mean": float(rmse_f.mean()) if rmse_f.size else None,
        "cross_rmse_p50": float(np.percentile(rmse_f, 50)) if rmse_f.size else None,
    }
    for p in PERCENTILES:
        metrics[f"cross_r2_p{p}"] = float(np.percentile(r2_f, p)) if r2_f.size else None
    cross_metrics_path.write_text(json.dumps(metrics, indent=2))

    # Per-voxel volumes: scatter masked arrays back into the ABCD grid, with
    # NaN outside the intersection mask so viewers display those voxels as
    # transparent rather than as a misleading 0.
    shape3 = measured_img.shape[:3]
    affine = measured_img.affine
    r2_vol = np.full(shape3, np.nan, dtype=np.float32)
    rmse_vol = np.full(shape3, np.nan, dtype=np.float32)
    r2_vol[mask] = r2.astype(np.float32)
    rmse_vol[mask] = rmse.astype(np.float32)
    r2_clipped_vol = np.where(
        np.isnan(r2_vol), np.nan, np.maximum(r2_vol, 0.0)
    ).astype(np.float32)
    nb.Nifti1Image(r2_vol, affine=affine).to_filename(str(r2_vol_path))
    nb.Nifti1Image(r2_clipped_vol, affine=affine).to_filename(str(r2_clipped_vol_path))
    nb.Nifti1Image(rmse_vol, affine=affine).to_filename(str(rmse_vol_path))

    out.update(metrics)
    return out


def run_one(cfg: FitConfig, bundle: Bundle, out_dir: Path,
            threads: int, force: bool, skip_qc: bool) -> dict:
    cfg_dir = out_dir / bundle.name / cfg.name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    coeffs = cfg_dir / "coeffs.nii.gz"
    odx = cfg_dir / "coeffs.odx"
    qc_json = cfg_dir / "qc.json"
    fit_log = cfg_dir / "cs-fit.stderr.log"
    odf_log = cfg_dir / "cs-odf.stderr.log"
    qc_log = cfg_dir / "odx-qc.stderr.log"

    cs_fit_cmd = [
        str(CS_FIT),
        "--dwi", str(bundle.dwi),
        "--bval", str(bundle.bval),
        "--bvec", str(bundle.bvec),
        "--mask", str(bundle.mask),
        "--output", str(coeffs),
        "--diagnostics",
        "--threads", str(threads),
        *cfg.cs_fit_args,
    ]

    fit_wall: Optional[float] = None
    if coeffs.exists() and not force:
        print(f"[bench] {bundle.name}/{cfg.name}: coeffs exist, skipping cs-fit")
    else:
        fit_wall, rc = run_capture(cs_fit_cmd, fit_log)
        if rc != 0:
            print(f"[bench] {bundle.name}/{cfg.name}: cs-fit FAILED (rc={rc}); see {fit_log}")
            return {"bundle": bundle.name, "name": cfg.name, "error": f"cs-fit rc={rc}"}

    odf_wall: Optional[float] = None
    if odx.exists() and not force:
        print(f"[bench] {bundle.name}/{cfg.name}: odx exists, skipping cs-odf")
    else:
        odf_wall, rc = run_capture(
            [str(CS_ODF), "--coeffs", str(coeffs), "--output", str(odx), "--threads", str(threads)],
            odf_log,
        )
        if rc != 0:
            print(f"[bench] {bundle.name}/{cfg.name}: cs-odf FAILED (rc={rc}); see {odf_log}")
            return {"bundle": bundle.name, "name": cfg.name, "error": f"cs-odf rc={rc}"}

    qc_data: dict = {}
    qc_wall: Optional[float] = None
    if not skip_qc:
        if qc_json.exists() and not force:
            qc_data = json.loads(qc_json.read_text())
        else:
            t0 = time.perf_counter()
            with qc_log.open("wb") as logf:
                proc = subprocess.run(
                    [str(ODX_BIN), "qc", str(odx), "--json"],
                    stdout=subprocess.PIPE, stderr=logf,
                )
            qc_wall = time.perf_counter() - t0
            if proc.returncode != 0:
                print(f"[bench] {bundle.name}/{cfg.name}: odx qc FAILED (rc={proc.returncode})")
                return {"bundle": bundle.name, "name": cfg.name, "error": f"odx-qc rc={proc.returncode}"}
            qc_json.write_bytes(proc.stdout)
            qc_data = json.loads(proc.stdout)

    mask = np.asarray(nb.load(str(bundle.mask)).dataobj) > 0

    coeffs_json = Path(strip_dwi_ext(coeffs) + ".json")
    sidecar = json.loads(coeffs_json.read_text()) if coeffs_json.exists() else {}

    r2_v = np.asarray(nb.load(str(sibling(coeffs, "_r2.nii.gz"))).get_fdata(dtype=np.float32))[mask]
    rmse_v = np.asarray(nb.load(str(sibling(coeffs, "_rmse.nii.gz"))).get_fdata(dtype=np.float32))[mask]
    sparsity_v = np.asarray(nb.load(str(sibling(coeffs, "_sparsity.nii.gz"))).get_fdata(dtype=np.float32))[mask]

    alpha_p = sibling(coeffs, "_alpha.nii.gz")
    alpha_v = np.asarray(nb.load(str(alpha_p)).get_fdata(dtype=np.float32))[mask] if alpha_p.exists() else np.array([], dtype=np.float32)

    bic_p = sibling(coeffs, "_bic.nii.gz")
    bic_v = np.asarray(nb.load(str(bic_p)).get_fdata(dtype=np.float32))[mask] if bic_p.exists() else np.array([], dtype=np.float32)

    row: dict = {"bundle": bundle.name, "name": cfg.name}
    row["fit_wall_s"] = fit_wall
    row["odf_wall_s"] = odf_wall
    row["qc_wall_s"] = qc_wall
    row["r2_mean"] = float(r2_v.mean()) if r2_v.size else None
    row.update({f"r2_p{p}": float(np.percentile(r2_v, p)) for p in PERCENTILES})
    row["rmse_mean"] = float(rmse_v.mean()) if rmse_v.size else None
    row["rmse_p50"] = float(np.percentile(rmse_v, 50)) if rmse_v.size else None
    row["sparsity_mean"] = float(sparsity_v.mean()) if sparsity_v.size else None
    row.update({f"sparsity_p{p}": float(np.percentile(sparsity_v, p)) for p in PERCENTILES})
    if alpha_v.size:
        row.update({f"alpha_p{p}": float(np.percentile(alpha_v, p)) for p in PERCENTILES})
    else:
        chosen = (sidecar.get("solver", {}) or {}).get("chosen_alpha", {}) or {}
        global_alpha = chosen.get("alpha") if chosen.get("scope") == "global" else None
        row.update({f"alpha_p{p}": global_alpha for p in PERCENTILES})
    row["bic_p50"] = float(np.percentile(bic_v, 50)) if bic_v.size else None
    row["coeffs_size_mb"] = coeffs.stat().st_size / (1024 * 1024) if coeffs.exists() else None

    row["total_fixels"] = qc_data.get("total_fixels")
    row["evaluated_fixels"] = qc_data.get("evaluated_fixels")
    row["coherence_index"] = qc_data.get("coherence_index")
    row["incoherence_index"] = qc_data.get("incoherence_index")
    row["connected_to_disconnected_ratio"] = qc_data.get("connected_to_disconnected_ratio")
    row["qa_otsu_threshold"] = qc_data.get("threshold_value")

    row["_solver_summary"] = describe_solver(sidecar.get("solver", {}) or {})
    return row


def describe_solver(solver: dict) -> str:
    kind = solver.get("kind")
    if kind == "tikhonov":
        return f"L2(λ_N={_fmt_value(solver.get('lambda_n'))}, λ_L={_fmt_value(solver.get('lambda_l'))})"
    if kind == "fista":
        strat = solver.get("alpha_strategy", {}) or {}
        sk = strat.get("kind", "?")
        if sk == "fixed":
            chosen = solver.get("chosen_alpha", {}) or {}
            return f"L1/fixed(α={_fmt_value(chosen.get('alpha'))})"
        if sk in ("alpha-ratio", "alpha-max-ratio"):
            return f"L1/alpha-ratio(ratio={_fmt_value(strat.get('ratio'))})"
        if sk == "path-bic":
            path = strat.get("path", {}) or {}
            return f"L1/path-bic(n={path.get('n')}, eps={_fmt_value(path.get('eps'))})"
        if sk == "path-l2-anchored":
            path = strat.get("path", {}) or {}
            return (f"L1/l2-anchored(n={path.get('n')}, "
                    f"eps={_fmt_value(path.get('eps'))}, "
                    f"slack={_fmt_value(strat.get('slack'))})")
        return f"L1/{sk}"
    return kind or "?"


def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def fmt_cell(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k.startswith("_") or k in seen:
                continue
            seen.add(k)
            keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in keys})


HEADLINE_COLS: list[tuple[str, str]] = [
    ("name", "config"),
    ("_solver_summary", "solver"),
    ("r2_mean", "R² mean"),
    ("r2_p10", "R² p10"),
    ("rmse_mean", "RMSE mean"),
    ("sparsity_p50", "sparsity p50"),
    ("alpha_p50", "α p50"),
    ("coherence_index", "coherence"),
    ("incoherence_index", "incoherence"),
    ("evaluated_fixels", "eval. fixels"),
    ("fit_wall_s", "fit s"),
    ("odf_wall_s", "odf s"),
    ("coeffs_size_mb", "coeffs MB"),
]


def render_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(fmt_cell(r.get(key)) for key, _ in cols) + " |")
    return "\n".join([head, sep, *body])


def write_per_bundle_report(bundle: Bundle, rows: list[dict], path: Path) -> None:
    by_r2 = sorted(rows, key=lambda r: (r.get("r2_mean") is None, -(r.get("r2_mean") or 0.0)))
    by_coh = sorted(rows, key=lambda r: (r.get("coherence_index") is None, -(r.get("coherence_index") or 0.0)))
    out = [
        f"# Benchmark: {bundle.name}",
        "",
        f"DWI: `{bundle.dwi.name}`. Mask: `{bundle.mask.name}`.",
        "",
        "## Sorted by R²",
        "",
        render_table(by_r2, HEADLINE_COLS),
        "",
        "## Sorted by coherence_index",
        "",
        render_table(by_coh, HEADLINE_COLS),
        "",
    ]
    path.write_text("\n".join(out) + "\n")


def write_summary(all_rows: list[dict], path_md: Path, path_csv: Path) -> None:
    """Top-level summary: every (bundle, config) row, plus per-bundle pivot tables
    on the most useful columns."""
    write_csv(all_rows, path_csv)

    bundles = sorted({r["bundle"] for r in all_rows})
    configs = sorted({r["name"] for r in all_rows})

    def pivot(metric: str) -> str:
        head = "| bundle | " + " | ".join(configs) + " |"
        sep = "|" + "|".join("---" for _ in range(len(configs) + 1)) + "|"
        body = []
        for b in bundles:
            cells = []
            for c in configs:
                row = next((r for r in all_rows if r["bundle"] == b and r["name"] == c), None)
                cells.append(fmt_cell(row.get(metric) if row else None))
            body.append("| " + b + " | " + " | ".join(cells) + " |")
        return "\n".join([head, sep, *body])

    cols_for_full_table = [("bundle", "bundle")] + HEADLINE_COLS

    cross_rows = [r for r in all_rows if r.get("cross_r2_mean") is not None or r.get("cross_error")]
    cross_section: list[str] = []
    if cross_rows:
        cross_cols = [
            ("bundle", "HASC bundle"),
            ("name", "config"),
            ("cross_target", "ABCD bundle"),
            ("r2_mean", "in-sample R²"),
            ("cross_r2_clipped_mean", "cross R² (≥0 mean)"),
            ("cross_r2_p50", "cross R² p50"),
            ("cross_r2_p10", "cross R² p10"),
            ("cross_r2_mean", "raw R² mean"),
            ("cross_rmse_mean", "cross RMSE"),
            ("cross_evaluated_voxels", "voxels"),
            ("cross_q_extrap_fraction", "q extrap"),
            ("cross_error", "error"),
        ]
        cross_sorted = sorted(
            cross_rows,
            key=lambda r: (
                r.get("cross_r2_clipped_mean") is None,
                -(r.get("cross_r2_clipped_mean") or 0.0),
            ),
        )
        cross_section = [
            "## HASC → ABCD prediction",
            "",
            "Per (HASC bundle, config) where the subject also has a paired ABCD acquisition. "
            "`in-sample R²` is the HASC fit's own R² (mean over its mask) — the gap "
            "`in-sample − cross_r2_mean` is the generalization cost from HASC's q-space "
            "sampling to ABCD's. R² is per-voxel across gradients, restricted to the "
            "intersection of HASC and ABCD brain masks.",
            "",
            render_table(cross_sorted, cross_cols),
            "",
        ]

    out = [
        "# cs-dmri benchmark — multi-bundle summary",
        "",
        f"Bundles: {len(bundles)}. Configs: {', '.join(configs)}.",
        "",
        "## Pivot: R² mean (per bundle × config)",
        "",
        pivot("r2_mean"),
        "",
        "## Pivot: coherence_index (per bundle × config)",
        "",
        pivot("coherence_index"),
        "",
        "## Pivot: sparsity p50",
        "",
        pivot("sparsity_p50"),
        "",
        "## Pivot: fit wall seconds",
        "",
        pivot("fit_wall_s"),
        "",
        *cross_section,
        "## All rows",
        "",
        render_table(all_rows, cols_for_full_table),
        "",
        "## Notes",
        "",
        "- Per-bundle CSV/report under `<out>/<bundle>/`.",
        "- `coherence`/`incoherence` from `odx qc` are weighted by the `qa` DPF; they sum to 1.",
        "- `α p50` for `fixed` modes is the global α from the JSON sidecar.",
        "- HASC55 acquisitions are compressed-sensing undersampled (63 grad); ABCD are dense (103 grad). Compare same-subject ABCD vs HASC55 rows to see the effect of undersampling.",
    ]
    path_md.write_text("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "cs-bench",
                    help="Output root. Defaults to ~/cs-bench (durable). Avoid /tmp on macOS — files untouched for 3 days are auto-deleted.")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--bundles-root", type=Path, default=None,
                    help="Discover BIDS-like *desc-preproc_dwi* bundles under this dir.")
    ap.add_argument("--bundle", type=Path, default=None,
                    help="Single bundle: path to `*desc-preproc_dwi.nii(.gz)`.")
    ap.add_argument("--filter-bundles", type=str, default=None,
                    help="Comma-separated bundle-name substrings to include.")
    ap.add_argument("--configs-preset", choices=["focused", "full"], default="focused",
                    help="Which built-in config matrix to use.")
    ap.add_argument("--filter-configs", type=str, default=None,
                    help="Comma-separated config names to run (subset of preset).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run cs-fit/cs-odf/qc even if outputs exist.")
    ap.add_argument("--skip-qc", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    ensure_built(args.skip_build)

    if args.bundles_root:
        all_bundles = discover_bundles(args.bundles_root)
        if not all_bundles:
            sys.exit(f"[bench] no bundles found under {args.bundles_root}")
    elif args.bundle:
        all_bundles = [derive_siblings(args.bundle)]
    else:
        all_bundles = [derive_siblings(DEFAULT_DWI)]

    # Pair HASC↔ABCD on the unfiltered set so --filter-bundles can target
    # just one subject (HASC) and still resolve its ABCD partner.
    cross_pairs = pair_hasc_to_abcd(all_bundles)
    if cross_pairs:
        print(f"[bench] HASC→ABCD pairs detected: {len(cross_pairs)}")

    bundles = all_bundles
    if args.filter_bundles:
        keys = [s.strip() for s in args.filter_bundles.split(",") if s.strip()]
        bundles = [b for b in bundles if any(k in b.name for k in keys)]
        if not bundles:
            sys.exit("[bench] --filter-bundles matched zero bundles")

    selected_configs = CONFIGS_FULL if args.configs_preset == "full" else CONFIGS_FOCUSED
    if args.filter_configs:
        names = {n.strip() for n in args.filter_configs.split(",") if n.strip()}
        selected_configs = [c for c in selected_configs if c.name in names]
        if not selected_configs:
            sys.exit("[bench] --filter-configs matched zero configs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] {len(bundles)} bundles × {len(selected_configs)} configs")
    for b in bundles:
        print(f"        - {b.name}: {b.dwi.name}")

    all_rows: list[dict] = []
    t_start = time.perf_counter()
    for bundle in bundles:
        bundle_rows: list[dict] = []
        for cfg in selected_configs:
            print(f"\n[bench] === {bundle.name} / {cfg.name} ===", flush=True)
            row = run_one(cfg, bundle, args.out_dir, args.threads, args.force, args.skip_qc)

            if bundle.name in cross_pairs and not row.get("error"):
                print(f"[bench] --- cross-prediction {bundle.name} → {cross_pairs[bundle.name].name}", flush=True)
                cfg_dir = args.out_dir / bundle.name / cfg.name
                cross_row = run_cross_prediction(
                    cfg, bundle, cross_pairs[bundle.name], cfg_dir,
                    args.threads, args.force,
                )
                row.update(cross_row)

            bundle_rows.append(row)
            all_rows.append(row)
            # Incremental writes so a long run is recoverable.
            write_csv(bundle_rows, args.out_dir / bundle.name / "results.csv")
            write_per_bundle_report(bundle, bundle_rows, args.out_dir / bundle.name / "report.md")
            write_summary(all_rows,
                          args.out_dir / "summary.md",
                          args.out_dir / "summary.csv")
    print(f"\n[bench] total wall: {time.perf_counter() - t_start:.1f}s", flush=True)
    print(f"[bench] summary: {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
