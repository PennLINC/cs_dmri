# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare the fiber density ConSH recovers in the simulated study with the
planted truth.

Fiber density here is the l=0 coefficient of each subject's white-matter fODF,
which is proportional to the fODF's integral. Because every simulated subject is
the same anatomy without motion, voxels correspond across subjects without
registration. The group difference (patients over controls, minus one) is
compared with the true change in white-matter fraction, for the two fits that
run_consh.sh writes:

- the recommended one: the group response, in absolute signal units
  (``desc-wm_fod``);
- the per-scan response on the default per-voxel S/S0 scale
  (``desc-individual_wm_fod``).

    python analyze.py /data/consh_group/bids /data/consh_group/consh figures/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Diverging: loss red, no change neutral gray, gain blue. Two series: blue, orange.
DIVERGING = LinearSegmentedColormap.from_list(
    "loss_gain", ["#a3201f", "#e34948", "#f0efec", "#2a78d6", "#174a8c"])
SERIES = {"recommended": "#2a78d6", "default": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bids", type=Path)
    p.add_argument("consh", type=Path)
    p.add_argument("figures", type=Path)
    args = p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.bids / "participants.tsv"), delimiter="\t"))
    groups = {g: [r["participant_id"] for r in rows if r["group"] == g] for g in ("control", "patient")}
    subjects = groups["control"] + groups["patient"]
    deriv = args.bids / "derivatives"
    mask = np.logical_and.reduce(
        [load(deriv / f"masks/{s}/dwi/{s}_desc-brain_mask.nii.gz") > 0 for s in subjects])
    truth = {s: load(deriv / f"truth/{s}/dwi/{s}_desc-wm_fraction.nii.gz") for s in subjects}

    def group_mean(maps, group):
        return np.mean([maps[s] for s in groups[group]], axis=0)

    # White matter: a white-matter fraction above 0.5 in the controls.
    wm = mask & (group_mean(truth, "control") > 0.5)
    true_change = np.zeros(mask.shape, np.float32)
    true_change[wm] = group_mean(truth, "patient")[wm] / group_mean(truth, "control")[wm] - 1
    planted = wm & (true_change < -0.01)
    unaffected = wm & (np.abs(true_change) < 0.002)

    fits = {"recommended": "desc-wm_fod", "default": "desc-individual_wm_fod"}
    estimates, summary = {}, {"wm_voxels": int(wm.sum()), "planted_voxels": int(planted.sum())}
    for name, desc in fits.items():
        fd = {s: load(args.consh / s / f"{s}_{desc}.nii.gz")[..., 0] for s in subjects}
        control = group_mean(fd, "control")
        valid = wm & (control > 0.05 * np.median(control[wm]))
        est = np.full(mask.shape, np.nan, np.float32)
        est[valid] = group_mean(fd, "patient")[valid] / control[valid] - 1
        estimates[name] = est
        both = planted & valid
        rest = unaffected & valid
        slope = np.polyfit(true_change[both], est[both], 1)[0]
        summary[name] = {
            "planted_mean_true_pct": round(100 * float(true_change[both].mean()), 1),
            "planted_mean_estimated_pct": round(100 * float(est[both].mean()), 1),
            "planted_correlation": round(float(np.corrcoef(true_change[both], est[both])[0, 1]), 2),
            "planted_slope": round(float(slope), 2),
            "unaffected_median_pct": round(100 * float(np.median(est[rest])), 2),
            "unaffected_95pct_range_pct": [round(100 * float(q), 1) for q in np.percentile(est[rest], [2.5, 97.5])],
        }
    (args.figures / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    # Axial slice through the planted region; background is the mean b=0.
    z = int(np.argmax(planted.sum(axis=(0, 1))))
    s0 = subjects[0]
    dwi = load(args.bids / s0 / "dwi" / f"{s0}_dwi.nii.gz")
    b0 = dwi[..., np.loadtxt(args.bids / s0 / "dwi" / f"{s0}_dwi.bval") <= 50].mean(-1)
    panels = [("Planted change", true_change),
              ("Group response, absolute units", estimates["recommended"]),
              ("Per-scan response, S/S0 scale", estimates["default"])]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, (title, data) in zip(axes, panels):
        ax.imshow(np.rot90(b0[:, :, z]), cmap="gray", vmin=0, vmax=np.percentile(b0[mask], 99))
        shown = np.where(wm[:, :, z], 100 * data[:, :, z], np.nan)
        im = ax.imshow(np.rot90(shown), cmap=DIVERGING, vmin=-40, vmax=40, interpolation="nearest")
        ax.set_title(title, fontsize=10, color=INK)
        ax.axis("off")
    bar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    bar.set_label("Fiber density change, patients vs controls (%)", fontsize=9, color=INK)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    fig.savefig(args.figures / "consh_group_maps.png", dpi=150)
    plt.close(fig)

    # Estimated against true change in the planted voxels.
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    lim = (-45, 5)
    ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.text(-44, -41.5, "identity", color=MUTED, fontsize=8, rotation=37)
    labels = {"recommended": "Group response, absolute units",
              "default": "Per-scan response, S/S0 scale"}
    for name in ("default", "recommended"):
        sel = planted & np.isfinite(estimates[name])
        ax.scatter(100 * true_change[sel], 100 * estimates[name][sel], s=6, alpha=0.45, linewidths=0,
                   color=SERIES[name], label=labels[name], zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(-70, 15)
    ax.set_xlabel("True change (%)", color=INK, fontsize=9)
    ax.set_ylabel("Estimated change (%)", color=INK, fontsize=9)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(loc="upper left", fontsize=8, frameon=False, markerscale=2.5)
    fig.savefig(args.figures / "consh_group_scatter.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
