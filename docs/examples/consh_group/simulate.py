# SPDX-License-Identifier: MIT OR Apache-2.0
"""Simulate a small compressed-sensing DSI group study with TRXScan.

Every subject is the TRXScan ``sub-60501`` phantom (a real brain's tissue maps
and tractogram), with its own random subset of 400,000 streamlines and its own
noise, scanned with one compressed-sensing DSI protocol: a random half of the
q-space points of a DSI lattice up to b = 4000 s/mm², plus b=0 volumes. In the
"patient" group, fiber density in a bundle through the body of the corpus
callosum is reduced by ``--reduction`` (default 30%): the bundle's streamlines
lose that fraction of their weight, and the white-matter fraction they account
for becomes CSF, as tissue loss would.

The data need no preprocessing: the phantom does not move and the simulation has
no distortion, motion or eddy currents. Outputs (BIDS):

    <out>/sub-XX/dwi/sub-XX_dwi.{nii.gz,bval,bvec,json}
    <out>/derivatives/masks/sub-XX/dwi/sub-XX_desc-brain_mask.nii.gz   (brain mask)
    <out>/derivatives/truth/sub-XX/dwi/sub-XX_desc-wm_fraction.nii.gz
    <out>/participants.tsv                              (group column)

Requires ``pip install "trxscan[trx]"`` and the phantom (``trxscan-fetch
sub-60501``). Run time is about 20 s per subject.

    python simulate.py /data/consh_group --per-group 4
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import nibabel as nib
import numpy as np
import trxscan as ts

B_MAX = 4000.0
LATTICE_RADIUS = 5
VOXEL_MM = 2.5
N_STREAMLINES = 400_000
#: Measured with the default TRXScan protocol at these settings: the white-matter
#: b=0 signal, and the noise standard deviation per unit of ``sqrt(noise)``.
WM_B0 = 26.7
SIGMA_PER_SQRT_NOISE = 0.93


def cs_dsi_scheme(seed: int = 0, keep: float = 0.5, n_b0: int = 6):
    """b-values and unit vectors of a compressed-sensing DSI scheme: a random
    fraction ``keep`` of the half-lattice points with |q| ≤ ``LATTICE_RADIUS``,
    with b proportional to |q|², plus ``n_b0`` b=0 volumes."""
    pts = []
    r = LATTICE_RADIUS
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            for k in range(0, r + 1):
                n2 = i * i + j * j + k * k
                upper = k > 0 or j > 0 or (j == 0 and i > 0)
                if 0 < n2 <= r * r and upper:
                    pts.append((i, j, k))
    pts = np.array(pts, dtype=float)
    rng = np.random.default_rng(seed)
    pts = pts[np.sort(rng.choice(len(pts), int(round(keep * len(pts))), replace=False))]
    n2 = (pts**2).sum(1)
    bvals = np.concatenate([np.zeros(n_b0), B_MAX * n2 / (r * r)])
    bvecs = np.vstack([np.zeros((n_b0, 3)), pts / np.sqrt(n2)[:, None]])
    return bvals, bvecs


def callosal_bundle(streamlines: ts.Streamlines) -> np.ndarray:
    """Streamlines that cross the midsagittal plane (x = 0) in the body of the
    corpus callosum (-20 < y < 0 mm, 15 < z < 35 mm)."""
    pos, off = streamlines.positions, streamlines.offsets.astype(np.int64)
    sid = np.repeat(np.arange(streamlines.n), np.diff(off))
    a, b = pos[:-1], pos[1:]
    cross = (sid[1:] == sid[:-1]) & (np.sign(a[:, 0]) != np.sign(b[:, 0]))
    w = a[cross, 0] / (a[cross, 0] - b[cross, 0])
    pt = a[cross] + w[:, None] * (b[cross] - a[cross])
    inside = (pt[:, 1] > -20) & (pt[:, 1] < 0) & (pt[:, 2] > 15) & (pt[:, 2] < 35)
    bundle = np.zeros(streamlines.n, bool)
    bundle[sid[:-1][cross][inside]] = True
    return bundle


def reduce_bundle(phantom: ts.Phantom, bundle: np.ndarray, reduction: float) -> ts.Phantom:
    """Lower the fiber density of ``bundle`` by ``reduction`` (a fraction).

    TRXScan normalizes each voxel's fiber orientation distribution and scales
    the fiber signal by the white-matter fraction, so both change: the bundle's
    streamline weights are multiplied by ``1 - reduction``, and in every voxel
    the white-matter fraction the bundle accounts for (its share of the voxel's
    streamline density) is reduced by the same factor, the difference becoming
    CSF. The other fiber populations in a voxel keep their density.
    """
    sl = phantom.streamlines
    weights = np.ones(sl.n, np.float32) if sl.weights is None else sl.weights.copy()
    weights[bundle] *= 1.0 - reduction
    # The bundle's share of each 1 mm voxel's streamline density.
    affine = phantom.wm.affine
    inv = np.linalg.inv(affine)
    ijk = np.rint(sl.positions @ inv[:3, :3].T + inv[:3, 3]).astype(int)
    shape = phantom.wm.shape
    lin = np.ravel_multi_index(ijk.T, shape, mode="clip")
    sid = np.repeat(np.arange(sl.n), np.diff(sl.offsets.astype(np.int64)))
    total = np.bincount(lin, minlength=int(np.prod(shape))).reshape(shape)
    in_bundle = np.bincount(lin[bundle[sid]], minlength=int(np.prod(shape))).reshape(shape)
    share = np.where(total > 0, in_bundle / np.maximum(total, 1), 0.0)
    wm = np.asarray(phantom.wm.dataobj, dtype=np.float32)
    csf = np.asarray(phantom.csf.dataobj, dtype=np.float32)
    lost = wm * reduction * share
    as_img = lambda data, like: nib.Nifti1Image(data.astype(np.float32), like.affine, like.header)  # noqa: E731
    return dataclasses.replace(
        phantom,
        wm=as_img(wm - lost, phantom.wm),
        csf=as_img(csf + lost, phantom.csf),
        streamlines=ts.Streamlines(sl.positions, sl.offsets, weights),
        _grids={},
    )


def on_grid(image: nib.Nifti1Image, affine: np.ndarray) -> np.ndarray:
    """The data of ``image`` in the voxel order of ``affine``, a grid that differs
    from ``image``'s only by axis flips or permutations (TRXScan's tissue maps are
    stored LPS, its DWI LAS)."""
    from nibabel import orientations as o

    transform = o.ornt_transform(o.io_orientation(image.affine), o.io_orientation(affine))
    reoriented = image.as_reoriented(transform)
    if not np.allclose(reoriented.affine, affine, atol=1e-4):
        raise ValueError("the image is not on the same grid as the DWI")
    return np.asarray(reoriented.dataobj, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="BIDS dataset root to write")
    p.add_argument("--per-group", type=int, default=4, help="subjects per group (default 4)")
    p.add_argument("--reduction", type=float, default=0.3, help="fiber density reduction in patients (default 0.3)")
    p.add_argument("--snr", type=float, default=30.0, help="white-matter b=0 SNR (default 30)")
    args = p.parse_args()

    bvals, bvecs = cs_dsi_scheme()
    protocol = ts.Protocol.DEFAULT.replace(voxel_mm=VOXEL_MM, fsl_orientation=True, phase_model="none",
                                           pad=(2, 2, 2))
    dataset = ts.Dataset(args.out)
    base = ts.Phantom.load("sub-60501")
    subjects = [("control", i) for i in range(args.per_group)] + [("patient", i) for i in range(args.per_group)]
    for n, (group, _) in enumerate(subjects, start=1):
        sub = f"{n:02d}"
        phantom = base.subsample(N_STREAMLINES, seed=n)
        if group == "patient":
            phantom = reduce_bundle(phantom, callosal_bundle(phantom.streamlines), args.reduction)
        noise = (WM_B0 / (args.snr * SIGMA_PER_SQRT_NOISE)) ** 2
        artifacts = ts.Artifacts(distortion=False, noise=noise, seed=n)
        sim = phantom.simulate((bvals, bvecs), protocol, artifacts, chunk=8)
        dataset.add_dwi(sim, sub, parts=("mag",), truth=False, dir=None)
        dataset.add_participant(sub, group=group)
        # Tissue fractions on the acquisition grid give the true white-matter
        # fraction and the brain mask (tissue fractions summing to at least 0.5).
        grid = phantom.grid(protocol)
        wm, gm, csf = (on_grid(grid.image(t), sim.affine) for t in ("wm", "gm", "csf"))
        mask = wm + gm + csf >= 0.5
        for kind, name, data in (("masks", "brain_mask", mask.astype(np.uint8)),
                                 ("truth", "wm_fraction", wm)):
            target = args.out / "derivatives" / kind / f"sub-{sub}" / "dwi" / f"sub-{sub}_desc-{name}.nii.gz"
            target.parent.mkdir(parents=True, exist_ok=True)
            nib.Nifti1Image(data, sim.affine).to_filename(target)
        print(f"sub-{sub} ({group}): {len(bvals)} volumes", flush=True)


if __name__ == "__main__":
    main()
