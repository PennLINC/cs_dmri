# SPDX-License-Identifier: MIT OR Apache-2.0
"""Which frame a fit's gradient directions (and so its SH output) live in."""

from __future__ import annotations

import numpy as np

from . import _cs_dmri

FRAMES = ("auto", "world", "image")


def fit_frame(bvec_frame: str, affine) -> tuple[str, np.ndarray]:
    """Resolve ``bvec_frame`` to ``("world" | "image", R)`` where ``R`` rotates
    image-axis vectors into the fit frame (identity for "image").

    ``"auto"`` means world RAS when an affine is known, as the CLI does by
    default, so SH coefficients and ODX output are in world space. ``"image"``
    keeps dipy's convention (directions in the image's voxel axes).
    """
    if bvec_frame not in FRAMES:
        raise ValueError(f"bvec_frame must be one of {FRAMES}, got {bvec_frame!r}")
    if bvec_frame == "auto":
        bvec_frame = "world" if affine is not None else "image"
    if bvec_frame == "world":
        if affine is None:
            raise ValueError("bvec_frame='world' needs an affine (fit a DWI, or pass affine=)")
        return "world", _cs_dmri.affine_rotation(np.asarray(affine, dtype=np.float64))
    return "image", np.eye(3)


def rotate(vectors: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply ``R`` to row vectors (..., 3)."""
    return np.ascontiguousarray(np.asarray(vectors, dtype=np.float64) @ R.T)
