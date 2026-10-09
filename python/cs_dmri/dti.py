# SPDX-License-Identifier: MIT OR Apache-2.0
"""Robust diffusion-tensor fitting (RESTORE)."""

from __future__ import annotations

import numpy as np

from . import _cs_dmri
from ._inputs import resolve
from .gradients import as_gradient_table

__all__ = ["RestoreModel", "RestoreFit"]


class RestoreModel:
    """RESTORE diffusion-tensor model (Chang, Jones & Pierpaoli 2005).

    Iteratively reweighted tensor fit that detects and down-weights outlier
    measurements per voxel.

    Parameters
    ----------
    gtab : GradientTable or dipy gradient table
    max_iter, tol : int, float
        Iteratively reweighted least-squares limits.
    outlier_threshold : float
        Geman-McClure weight below which a measurement counts as an outlier
        (0.04 ≈ a 2σ residual). Only affects ``outlier_fraction``, not the fit.
    min_signal : float
        Signals are clipped to at least this before taking logs.
    """

    def __init__(self, gtab, *, max_iter: int = 50, tol: float = 1e-6,
                 outlier_threshold: float = 0.04, min_signal: float = 1e-6):
        self.gtab = as_gradient_table(gtab)
        self.max_iter = max_iter
        self.tol = tol
        self.outlier_threshold = outlier_threshold
        self.min_signal = min_signal

    def fit(self, data, mask=None, *, diagnostics: bool = False, n_threads: int | None = None) -> "RestoreFit":
        """Fit every voxel in ``mask``.

        ``data`` is a :class:`~cs_dmri.DWI` or a 4-D array; without a mask the
        DWI's effective mask (or, for an array, the b=0 fallback mask) is used.
        """
        vol, m, affine, _ = resolve(data, mask, self.gtab)
        out = _cs_dmri.dti_fit_restore(
            vol, self.gtab.bvals, self.gtab.bvecs, m,
            b0_threshold=self.gtab.b0_threshold, max_iter=self.max_iter, tol=self.tol,
            outlier_threshold=self.outlier_threshold, min_signal=self.min_signal,
            diagnostics=diagnostics, n_threads=n_threads,
        )
        return RestoreFit(self, m, affine, out)


class RestoreFit:
    """Result of :meth:`RestoreModel.fit`. Maps are zero outside the mask.

    Attributes
    ----------
    fa, md, s0, outlier_fraction : (X, Y, Z) float32
    tensor : (X, Y, Z, 6) float32, ``[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]`` in mm²/s
    principal_dir : (X, Y, Z, 3) float32, in the gradient table's frame
    iterations, converged : (X, Y, Z), only with ``diagnostics=True``
    """

    def __init__(self, model: RestoreModel, mask, affine, arrays: dict):
        self.model = model
        self.mask = mask
        self.affine = affine
        self.fa = arrays["fa"]
        self.md = arrays["md"]
        self.s0 = arrays["s0"]
        self.outlier_fraction = arrays["outlier_fraction"]
        self.tensor = arrays["tensor"]
        self.principal_dir = arrays["principal_dir"]
        self.iterations = arrays.get("iterations")
        self.converged = arrays.get("converged")

    @property
    def quadratic_form(self) -> np.ndarray:
        """Full symmetric tensors, (X, Y, Z, 3, 3)."""
        t = self.tensor
        xx, xy, xz, yy, yz, zz = (t[..., i] for i in range(6))
        return np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1),
                         np.stack([xz, yz, zz], -1)], -2)

    @property
    def evals(self) -> np.ndarray:
        """Eigenvalues, descending, (X, Y, Z, 3)."""
        return np.linalg.eigvalsh(self.quadratic_form)[..., ::-1]

    def __repr__(self) -> str:
        n = int(self.mask.sum())
        fa = float(self.fa[self.mask].mean()) if n else float("nan")
        return f"RestoreFit({n} voxels, mean FA {fa:.3f})"
