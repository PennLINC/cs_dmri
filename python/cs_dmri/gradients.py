# SPDX-License-Identifier: MIT OR Apache-2.0
"""Gradient tables."""

from __future__ import annotations

from os import PathLike

import numpy as np

__all__ = ["GradientTable", "as_gradient_table", "read_bvals_bvecs"]


def read_bvals_bvecs(bval_file: str | PathLike, bvec_file: str | PathLike):
    """Read FSL-style ``.bval`` / ``.bvec`` files.

    Returns ``(bvals (N,), bvecs (N, 3))``. bvecs files may be 3×N (FSL) or N×3.
    """
    bvals = np.loadtxt(bval_file, dtype=np.float64, ndmin=1).ravel()
    bvecs = np.loadtxt(bvec_file, dtype=np.float64, ndmin=2)
    return bvals, _bvecs_n3(bvecs, len(bvals))


def _bvecs_n3(bvecs, n: int) -> np.ndarray:
    bvecs = np.asarray(bvecs, dtype=np.float64)
    if bvecs.shape == (n, 3):
        return np.ascontiguousarray(bvecs)
    if bvecs.shape == (3, n):
        return np.ascontiguousarray(bvecs.T)
    raise ValueError(f"bvecs must have shape ({n}, 3) or (3, {n}), got {bvecs.shape}")


class GradientTable:
    """b-values and gradient directions of a DWI series.

    Directions are taken as given; cs_dmri never flips or rotates them. For
    data loaded with nibabel, that means the image's voxel-axis frame (dipy's
    and FSL's convention).

    Parameters
    ----------
    bvals : array (N,)
    bvecs : array (N, 3) or (3, N)
    b0_threshold : float
        Volumes with ``b <= b0_threshold`` are b=0.
    big_delta, small_delta : float, optional
        Diffusion times in seconds. Only the SHORE fits use them; if either is
        missing both are estimated from the maximum b-value (TORTOISE's
        heuristic).
    """

    def __init__(self, bvals, bvecs, *, b0_threshold: float = 50.0,
                 big_delta: float | None = None, small_delta: float | None = None):
        bvals = np.ascontiguousarray(np.asarray(bvals, dtype=np.float64).ravel())
        self._bvals = bvals
        self._bvecs = _bvecs_n3(bvecs, len(bvals))
        self._bvals.setflags(write=False)
        self._bvecs.setflags(write=False)
        self.b0_threshold = float(b0_threshold)
        self.big_delta = big_delta
        self.small_delta = small_delta

    @classmethod
    def from_files(cls, bval_file, bvec_file, **kwargs) -> "GradientTable":
        """Build from FSL ``.bval`` / ``.bvec`` files."""
        return cls(*read_bvals_bvecs(bval_file, bvec_file), **kwargs)

    @classmethod
    def from_dipy(cls, gtab, **kwargs) -> "GradientTable":
        """Build from a ``dipy.core.gradients.GradientTable`` (or anything with
        ``bvals`` / ``bvecs`` attributes)."""
        kwargs.setdefault("b0_threshold", getattr(gtab, "b0_threshold", 50.0))
        return cls(gtab.bvals, gtab.bvecs, **kwargs)

    @property
    def bvals(self) -> np.ndarray:
        return self._bvals

    @property
    def bvecs(self) -> np.ndarray:
        return self._bvecs

    @property
    def b0s_mask(self) -> np.ndarray:
        return self._bvals <= self.b0_threshold

    @property
    def max_b(self) -> float:
        return float(self._bvals.max()) if len(self._bvals) else 0.0

    def __len__(self) -> int:
        return len(self._bvals)

    def __repr__(self) -> str:
        n_b0 = int(self.b0s_mask.sum())
        return (f"GradientTable({len(self)} volumes: {n_b0} b=0, {len(self) - n_b0} DWI, "
                f"max b={self.max_b:g}, b0_threshold={self.b0_threshold:g})")


def as_gradient_table(gtab, **kwargs) -> GradientTable:
    """Accept a cs_dmri or dipy gradient table, or a ``(bvals, bvecs)`` pair."""
    if isinstance(gtab, GradientTable):
        return gtab
    if isinstance(gtab, tuple) and len(gtab) == 2:
        return GradientTable(*gtab, **kwargs)
    if hasattr(gtab, "bvals") and hasattr(gtab, "bvecs"):
        return GradientTable.from_dipy(gtab, **kwargs)
    raise TypeError(f"expected a GradientTable, a dipy gradient table or (bvals, bvecs); got {type(gtab)!r}")
