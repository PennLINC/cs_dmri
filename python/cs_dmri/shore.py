# SPDX-License-Identifier: MIT OR Apache-2.0
"""3D-SHORE fitting with L1 (compressed-sensing), L2 or non-negative-ODF
regularization, and everything derived from the coefficients."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

import numpy as np

from . import _cs_dmri
from ._frame import fit_frame, rotate
from ._inputs import resolve
from .gradients import as_gradient_table

__all__ = ["ShoreModel", "ShoreFit"]

_REGULARIZATIONS = ("l1", "l2", "nonneg")
_ALPHA_MODES = ("l2-anchored", "path-bic", "alpha-ratio", "fixed")


class ShoreModel:
    """3D-SHORE model (Merlet & Deriche 2013; Özarslan et al. 2013).

    Parameters
    ----------
    gtab : GradientTable or dipy gradient table
    radial_order : int
        Even; 6 gives 72 coefficients. Default 6.
    zeta : float
        Scale parameter. Default 700.
    regularization : {"l1", "l2", "nonneg"}
        ``"l1"``: sparse FISTA fit with a per-voxel α (the compressed-sensing
        path). ``"l2"``: closed-form Tikhonov. ``"nonneg"``: ICLS with
        non-negative ODF amplitudes on a dense sphere.
    alpha_mode : {"l2-anchored", "path-bic", "alpha-ratio", "fixed"}
        L1 α selection. ``"l2-anchored"`` (default) picks per voxel the largest
        α whose RSS is within ``(1 + slack)`` of an L2 fit's.
    alpha, alpha_ratio, path_n_alphas, path_eps, slack, max_iter, tol, non_negative
        L1 settings; ``path_eps=None`` takes the mode's default (1e-4 for
        l2-anchored, 1e-3 for path-bic).
    lambda_n, lambda_l : float
        Tikhonov radial / angular weights (L2, and L2-anchored's reference).
    bvec_frame : {"auto", "world", "image"}
        Frame of the fit and of the SH coefficients. ``"auto"`` uses world RAS
        when an affine is known and the image's voxel axes otherwise.
    """

    def __init__(self, gtab, *, radial_order: int = 6, zeta: float = 700.0, regularization: str = "l1",
                 alpha_mode: str = "l2-anchored", alpha: float = 1.0, alpha_ratio: float = 1e-3,
                 path_n_alphas: int = 20, path_eps: float | None = None, slack: float = 0.05,
                 max_iter: int = 1000, tol: float = 1e-6, non_negative: bool = False,
                 lambda_n: float = 1e-8, lambda_l: float = 1e-8, nonneg_max_iter: int = 200,
                 nonneg_tol: float = 1e-9, nonneg_epsilon: float = 1e-10, bvec_frame: str = "auto"):
        if regularization not in _REGULARIZATIONS:
            raise ValueError(f"regularization must be one of {_REGULARIZATIONS}, got {regularization!r}")
        if alpha_mode not in _ALPHA_MODES:
            raise ValueError(f"alpha_mode must be one of {_ALPHA_MODES}, got {alpha_mode!r}")
        self.gtab = as_gradient_table(gtab)
        self.radial_order = radial_order
        self.zeta = zeta
        self.regularization = regularization
        self.bvec_frame = bvec_frame
        self._solver = dict(
            alpha_mode=alpha_mode, alpha=alpha, alpha_ratio=alpha_ratio, path_n_alphas=path_n_alphas,
            path_eps=path_eps, slack=slack, max_iter=max_iter, tol=tol, non_negative=non_negative,
            lambda_n=lambda_n, lambda_l=lambda_l, nonneg_max_iter=nonneg_max_iter, nonneg_tol=nonneg_tol,
            nonneg_epsilon=nonneg_epsilon)

    def fit(self, data, mask=None, *, affine=None, diagnostics: bool = True,
            n_threads: int | None = None) -> "ShoreFit":
        """Fit every voxel in the mask (a DWI's effective mask by default)."""
        vol, m, dwi_affine, dwi = resolve(data, mask, self.gtab)
        affine = dwi_affine if affine is None else np.asarray(affine, dtype=np.float64)
        frame, R = fit_frame(self.bvec_frame, affine)
        out = _cs_dmri.shore_fit(
            vol, self.gtab.bvals, rotate(self.gtab.bvecs, R), m,
            big_delta=self.gtab.big_delta, small_delta=self.gtab.small_delta,
            b0_threshold=self.gtab.b0_threshold,
            bvec_frame="world-ras" if frame == "world" else "image-axis",
            radial_order=self.radial_order, zeta=self.zeta, regularization=self.regularization,
            diagnostics=diagnostics, n_threads=n_threads, **self._solver)
        return ShoreFit(out, model=self, mask=m, affine=affine, frame=frame, rotation=R, dwi=dwi)


class ShoreFit:
    """Result of :meth:`ShoreModel.fit` (or :meth:`load`).

    Attributes
    ----------
    coefficients : (X, Y, Z, K) float32
    r2, rmse, residual_l2, iterations, regularization_kind : (X, Y, Z)
        Per-voxel diagnostics (``None`` unless fit with ``diagnostics=True``).
    alpha, bic, rss_l2 : (X, Y, Z) or None
        Present for the L1 strategies that produce them.
    sidecar : dict
        Fit metadata, as written next to the coefficient NIfTI.
    frame : {"world", "image"}
        Frame of the coefficients' angular part.
    """

    def __init__(self, arrays: dict, *, model=None, mask=None, affine=None, frame="image", rotation=None,
                 dwi=None):
        self.coefficients = arrays["coefficients"]
        self.sidecar = arrays["sidecar"]
        self.alpha_distribution = arrays.get("alpha_distribution")
        for k in ("r2", "rmse", "residual_l2", "iterations", "regularization_kind", "alpha", "bic", "rss_l2"):
            setattr(self, k, arrays.get(k))
        self.model = model
        self.mask = mask if mask is not None else np.any(self.coefficients != 0, axis=-1)
        self.affine = affine
        self.frame = frame
        self._rotation = np.eye(3) if rotation is None else rotation
        self._dwi = dwi
        self.header = dwi.header if dwi is not None else None
        basis = self.sidecar["basis"]
        self.radial_order = int(basis["radial_order"])
        self.zeta = float(basis["zeta"])

    @property
    def sparsity(self) -> np.ndarray:
        """Fraction of nonzero coefficients per voxel (L1 sparsity)."""
        out = np.zeros(self.coefficients.shape[:3], np.float32)
        out[self.mask] = (self.coefficients[self.mask] != 0).mean(-1)
        return out

    def odf_sh(self, lmax: int | None = None, *, n_threads=None) -> np.ndarray:
        """Tournier (MRtrix) SH coefficients of the ODF, (X, Y, Z, n_sh).
        ``lmax`` defaults to the largest even ≤ radial order."""
        lmax = self.radial_order - self.radial_order % 2 if lmax is None else lmax
        return _cs_dmri.shore_odf_sh(self.coefficients, self.radial_order, self.zeta, lmax, self.mask,
                                     n_threads=n_threads)

    def predict(self, gtab=None, *, clip: bool = True, n_threads=None) -> np.ndarray:
        """Signal for ``gtab`` (default: the fit's own), (X, Y, Z, N).

        ``gtab`` is in the image's voxel-axis frame, like the one the model was
        built with; it is rotated into the fit frame internally.
        """
        if gtab is None:
            if self.model is None:
                raise ValueError("pass a gradient table: this fit has no model")
            gtab = self.model.gtab
        gtab = as_gradient_table(gtab)
        out = _cs_dmri.shore_predict(
            self.coefficients, self.radial_order, self.zeta, gtab.bvals, rotate(gtab.bvecs, self._rotation),
            big_delta=self.sidecar["big_delta_seconds"], small_delta=self.sidecar["small_delta_seconds"],
            n_threads=n_threads)
        return np.maximum(out, 0) if clip else out

    def microstructure(self, directions=None, *, units: str = "um", outlier_factor: float | None = 10.0,
                       n_threads=None) -> dict:
        """RTOP, RTAP, RTPP, MSD, QIV and NG maps.

        RTAP/RTPP need a fiber direction per voxel: ``directions`` (X, Y, Z, 3)
        in the image's voxel-axis frame, or by default the RESTORE principal
        direction of the DWI this was fit on. Without either they are NaN.
        ``units="um"`` follows TORTOISE (q in 1/µm); ``"mm"`` follows dipy.
        """
        if directions is None and self._dwi is not None:
            directions = self._dwi.tensor.principal_dir
        if directions is not None:
            directions = rotate(directions, self._rotation).astype(np.float32)
        return _cs_dmri.shore_microstructure(
            self.coefficients, self.radial_order, self.zeta, self.mask, directions, units=units,
            outlier_factor=outlier_factor, n_threads=n_threads)

    def export(self, path: str | PathLike, *, format: str | None = None, fixel_container: str = "nifti",
               lmax: int | None = None, peaks: bool = True, microstructure: bool = True,
               include_diagnostics: bool = True, overwrite: bool = False, n_threads=None,
               **odx_options) -> None:
        """Write the ODF SH, peaks, anisotropic power, GFA, microstructure and
        diagnostics (as per-voxel values), like ``cs-odf``.

        ``format`` is one of :data:`cs_dmri.EXPORT_FORMATS`; when omitted it is
        inferred from the extension (``.odx``, ``.fz``, ``.fib.gz``, ``.pam5``,
        ``.mif``/``.mif.gz``/``.nii``/``.nii.gz`` for MRtrix3 SH images).
        Directory outputs (``"odx-directory"``, ``"mrtrix-fixel-dir"``) must be
        named explicitly. ``fixel_container`` (``"nifti"`` or ``"mif"``) sets
        the image format inside an MRtrix3 fixel directory. Formats other than
        ODX keep the subset of the data they can represent.
        """
        if self.affine is None:
            raise ValueError("exporting needs an affine")
        if self.frame != "world":
            raise ValueError("exports are world-space: fit with bvec_frame='world' "
                             "(the default when an affine is known)")
        dpvs = {}
        if include_diagnostics:
            for name in ("r2", "rmse", "alpha", "bic"):
                if getattr(self, name) is not None:
                    dpvs[name] = np.asarray(getattr(self, name), dtype=np.float32)
            if self.r2 is not None:
                dpvs["sparsity"] = self.sparsity
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass overwrite=True")
        _cs_dmri.shore_export(
            Path(path), self.coefficients, self.affine, self.radial_order, self.zeta, self.mask, format=format,
            fixel_container=fixel_container, lmax=lmax, dpvs=dpvs, peaks=peaks, microstructure=microstructure,
            overwrite=overwrite, n_threads=n_threads, **odx_options)

    def to_odx(self, path: str | PathLike, *, directory: bool = False, **options) -> None:
        """Write an ODX archive, or an ODX directory with ``directory=True``.
        Other keyword arguments are as for :meth:`export`."""
        self.export(path, format="odx-directory" if directory else "odx-archive", **options)

    def save(self, path: str | PathLike, *, overwrite: bool = False) -> None:
        """Write the coefficient NIfTI, its JSON sidecar and diagnostic siblings
        (``_r2``, ``_rmse``, ...) in the layout ``cs-fit`` uses, so ``cs-odf``
        and ``cs-synth`` can read them. NIfTIs inherit the source DWI's header."""
        from ._spatial import make_image

        if self.affine is None:
            raise ValueError("saving needs an affine")
        path = Path(path)
        stem = str(path)[: -len(".nii.gz")] if str(path).endswith(".nii.gz") else str(path.with_suffix(""))
        outputs = {path: self.coefficients, Path(stem + ".json"): None}
        for name in ("r2", "rmse", "alpha", "bic", "rss_l2"):
            if getattr(self, name) is not None:
                outputs[Path(f"{stem}_{name}.nii.gz")] = np.asarray(getattr(self, name), dtype=np.float32)
        if self.r2 is not None:
            outputs[Path(f"{stem}_sparsity.nii.gz")] = self.sparsity
        if not overwrite:
            existing = [str(p) for p in outputs if p.exists()]
            if existing:
                raise FileExistsError(f"refusing to overwrite {existing}; pass overwrite=True")
        for p, arr in outputs.items():
            if arr is None:
                p.write_text(json.dumps(self.sidecar, indent=2))
            else:
                make_image(arr, self.affine, self.header).to_filename(p)

    @classmethod
    def load(cls, path: str | PathLike, mask=None) -> "ShoreFit":
        """Read a coefficient NIfTI written by ``cs-fit`` or :meth:`save`."""
        import nibabel as nib

        path = Path(path)
        stem = str(path)[: -len(".nii.gz")] if str(path).endswith(".nii.gz") else str(path.with_suffix(""))
        img = nib.load(path)
        sidecar = json.loads(Path(stem + ".json").read_text())
        arrays = {"coefficients": np.asarray(img.dataobj, dtype=np.float32), "sidecar": sidecar}
        for name in ("r2", "rmse", "alpha", "bic", "rss_l2"):
            p = Path(f"{stem}_{name}.nii.gz")
            if p.exists():
                arrays[name] = np.asarray(nib.load(p).dataobj, dtype=np.float32)
        frame = "world" if sidecar.get("bvec_frame") == "world-ras" else "image"
        rotation = _cs_dmri.affine_rotation(img.affine) if frame == "world" else None
        fit = cls(arrays, mask=mask, affine=img.affine, frame=frame, rotation=rotation)
        fit.header = img.header.copy()
        return fit

    def __repr__(self) -> str:
        r2 = "" if self.r2 is None else f", mean R² {float(self.r2[self.mask].mean()):.3f}"
        return f"ShoreFit(radial_order={self.radial_order}, {int(self.mask.sum())} voxels{r2}, frame={self.frame})"
