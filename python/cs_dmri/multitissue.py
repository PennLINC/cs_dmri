# SPDX-License-Identifier: MIT OR Apache-2.0
"""Single-shell three-tissue CSD: responses, SS3T, multi-tissue normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from os import PathLike
from pathlib import Path

import numpy as np

from . import _cs_dmri
from ._frame import fit_frame, rotate
from ._inputs import as_mask, resolve
from .gradients import as_gradient_table

__all__ = ["TissueResponse", "ResponseSet", "estimate_responses", "SS3TModel", "SS3TFit", "mtnormalise",
           "ss3t_pipeline"]

MTNORMALISE_TARGET = 1.0 / math.sqrt(4.0 * math.pi)


@dataclass(frozen=True)
class TissueResponse:
    """One tissue's response: zonal SH coefficients per shell
    (``coeffs`` (n_shells, n_coef), first row the b=0 shell)."""

    coeffs: np.ndarray
    lmax: int

    @classmethod
    def from_text(cls, text: str) -> "TissueResponse":
        return cls(*_cs_dmri.response_from_text(text))

    @classmethod
    def from_mrtrix_txt(cls, path: str | PathLike) -> "TissueResponse":
        return cls.from_text(Path(path).read_text())

    def to_text(self) -> str:
        return _cs_dmri.response_to_text(np.asarray(self.coeffs, dtype=np.float64), self.lmax)

    def to_mrtrix_txt(self, path: str | PathLike) -> None:
        Path(path).write_text(self.to_text())

    def _pair(self):
        return np.ascontiguousarray(self.coeffs, dtype=np.float64), int(self.lmax)


@dataclass(frozen=True)
class ResponseSet:
    """WM, GM and CSF responses, plus the voxel masks and diagnostics of the
    estimate when there was one."""

    wm: TissueResponse
    gm: TissueResponse
    csf: TissueResponse
    masks: dict = field(default_factory=dict, repr=False)
    diagnostics: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load_mrtrix(cls, wm, gm, csf) -> "ResponseSet":
        """From three MRtrix ``.txt`` files."""
        return cls(*(TissueResponse.from_mrtrix_txt(p) for p in (wm, gm, csf)))

    def save_mrtrix(self, directory: str | PathLike, prefix: str = "") -> dict:
        """Write ``{prefix}{wm,gm,csf}_response.txt`` (``cs-ss3t-full
        --write-responses-to`` layout); returns the paths."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in ("wm", "gm", "csf"):
            p = d / f"{prefix}{name}_response.txt"
            getattr(self, name).to_mrtrix_txt(p)
            paths[name] = p
        return paths


def estimate_responses(data, gtab, mask=None, *, tensor_fit=None, lmax_wm: int = 8,
                       legacy_selection: bool = False, erode: int = 3, fa: float = 0.2, sfwm_pct: float = 0.5,
                       gm_pct: float = 2.0, csf_pct: float = 10.0, fa_wm_threshold: float = 0.7,
                       fiber_dominance_ratio: float = 2.0, md_csf_pct: float = 2.5,
                       n_threads: int | None = None) -> ResponseSet:
    """Dhollander (2016) WM/GM/CSF responses from a single-shell series.

    The voxel selection follows MRtrix3's ``dwi2response dhollander``
    (``erode``, ``fa``, ``sfwm_pct``, ``gm_pct``, ``csf_pct``);
    ``legacy_selection`` switches to the older FA/MD threshold triple. Needs a
    RESTORE fit; :meth:`cs_dmri.DWI.estimate_responses` reuses the DWI's cached
    one.
    """
    gtab = as_gradient_table(gtab)
    vol, m, _, dwi = resolve(data, mask, gtab)
    if tensor_fit is None:
        tensor_fit = dwi.tensor if dwi is not None and mask is None else None
    if tensor_fit is None:
        from .dti import RestoreModel
        tensor_fit = RestoreModel(gtab).fit(vol, m, n_threads=n_threads)
    dti = {k: np.asarray(getattr(tensor_fit, k), dtype=np.float32)
           for k in ("s0", "fa", "md", "outlier_fraction", "tensor", "principal_dir")}
    r = _cs_dmri.estimate_responses(
        vol, gtab.bvals, gtab.bvecs, m, dti, b0_threshold=gtab.b0_threshold, lmax_wm=lmax_wm,
        legacy_selection=legacy_selection, erode=erode, fa=fa, sfwm_pct=sfwm_pct, gm_pct=gm_pct,
        csf_pct=csf_pct, fa_wm_threshold=fa_wm_threshold, fiber_dominance_ratio=fiber_dominance_ratio,
        md_csf_pct=md_csf_pct, n_threads=n_threads)
    return ResponseSet(
        wm=TissueResponse(*r["wm"]), gm=TissueResponse(*r["gm"]), csf=TissueResponse(*r["csf"]),
        masks={"wm": r["wm_mask"], "gm": r["gm_mask"], "csf": r["csf_mask"]},
        diagnostics=r["diagnostics"])


class SS3TModel:
    """Single-Shell 3-Tissue CSD (Dhollander & Connelly 2016).

    Parameters
    ----------
    gtab : GradientTable or dipy gradient table
        One b=0 shell plus one DWI shell.
    responses : ResponseSet
    lmax_wm : int
        WM FOD order (clamped to the WM response's lmax).
    lmax_candidates : list of int, optional
        Choose the WM lmax per voxel by BIC among these instead.
    niter, bzero_pct : int, float
        SS3T iterations and b=0 weighting (percent).
    bvec_frame : {"auto", "world", "image"}
        Frame of the WM FOD SH. ``"auto"``: world RAS when an affine is known,
        as ``cs-ss3t`` does.
    """

    def __init__(self, gtab, responses: ResponseSet, *, lmax_wm: int = 8, lmax_candidates=None,
                 niter: int = 3, bzero_pct: float = 10.0, icls_max_iter: int = 200, icls_tol: float = 1e-10,
                 icls_epsilon: float = 1e-10, bvec_frame: str = "auto"):
        self.gtab = as_gradient_table(gtab)
        self.responses = responses
        self.lmax_wm = lmax_wm
        self.lmax_candidates = None if lmax_candidates is None else [int(x) for x in lmax_candidates]
        self.niter = niter
        self.bzero_pct = bzero_pct
        self.icls = dict(icls_max_iter=icls_max_iter, icls_tol=icls_tol, icls_epsilon=icls_epsilon)
        self.bvec_frame = bvec_frame

    def fit(self, data, mask=None, *, affine=None, diagnostics: bool = False,
            n_threads: int | None = None) -> "SS3TFit":
        vol, m, dwi_affine, _ = resolve(data, mask, self.gtab)
        affine = dwi_affine if affine is None else np.asarray(affine, dtype=np.float64)
        frame, R = fit_frame(self.bvec_frame, affine)
        r = _cs_dmri.ss3t_fit(
            vol, self.gtab.bvals, rotate(self.gtab.bvecs, R), m, self.responses.wm._pair(),
            self.responses.gm._pair(), self.responses.csf._pair(), b0_threshold=self.gtab.b0_threshold,
            niter=self.niter, bzero_pct=self.bzero_pct, lmax_wm=self.lmax_wm,
            lmax_candidates=self.lmax_candidates, diagnostics=diagnostics, n_threads=n_threads, **self.icls)
        diag = {k: r[k] for k in ("iterations", "residual_l2", "converged", "chosen_lmax", "bic") if k in r}
        return SS3TFit(wm=r["wm"], gm=r["gm"], csf=r["csf"], lmax_wm=r["lmax_wm"], mask=m, affine=affine,
                       frame=frame, responses=self.responses, diagnostics=diag, warnings=tuple(r["warnings"]))


@dataclass(frozen=True)
class SS3TFit:
    """Tissue maps from :class:`SS3TModel`: ``wm`` (X, Y, Z, n_sh) Tournier SH,
    ``gm`` and ``csf`` (X, Y, Z). ``normalization`` holds mtnormalise's
    diagnostics once applied."""

    wm: np.ndarray
    gm: np.ndarray
    csf: np.ndarray
    lmax_wm: int
    mask: np.ndarray = field(repr=False)
    affine: np.ndarray | None = field(repr=False)
    frame: str
    responses: ResponseSet = field(repr=False)
    diagnostics: dict = field(default_factory=dict, repr=False)
    warnings: tuple[str, ...] = ()
    normalization: dict | None = field(default=None, repr=False)

    def mtnormalise(self, *, target_sum: float | str = MTNORMALISE_TARGET, poly_order: int = 3,
                    niter: int = 15, balance_maxiter: int = 7, balanced: bool = False) -> "SS3TFit":
        """Multi-tissue intensity normalization (MRtrix3 ``mtnormalise``);
        returns a new fit. ``target_sum="median"`` keeps the median observed sum."""
        wm, gm, csf, diag = mtnormalise(self.wm, self.gm, self.csf, self.mask, target_sum=target_sum,
                                        poly_order=poly_order, niter=niter, balance_maxiter=balance_maxiter,
                                        balanced=balanced)
        return replace(self, wm=wm, gm=gm, csf=csf, normalization=diag)

    def to_odx(self, path: str | PathLike, *, directory: bool = False, overwrite: bool = False) -> None:
        """Write an ODX (WM SH glyphs, GM/CSF, mask, WM peaks), like ``cs-ss3t-full --odx``."""
        if self.affine is None:
            raise ValueError("writing an ODX needs an affine")
        if self.frame != "world":
            raise ValueError("ODX is world-space: fit with bvec_frame='world' (the default when an affine is known)")
        r = self.responses
        _cs_dmri.ss3t_write_odx(Path(path), self.affine, self.mask, self.wm, self.gm, self.csf, self.lmax_wm,
                                r.wm._pair(), r.gm._pair(), r.csf._pair(), directory=directory,
                                overwrite=overwrite)


def mtnormalise(wm, gm, csf, mask, *, target_sum: float | str = MTNORMALISE_TARGET, poly_order: int = 3,
                niter: int = 15, balance_maxiter: int = 7, balanced: bool = False):
    """Multi-tissue log-domain intensity normalization, a port of MRtrix3's
    ``mtnormalise``. ``wm`` is (X, Y, Z, n_sh); ``gm``/``csf`` are (X, Y, Z).
    Returns new ``(wm, gm, csf, diagnostics)``."""
    if isinstance(target_sum, str):
        if target_sum != "median":
            raise ValueError("target_sum must be a number or 'median'")
        target_sum = None
    wm = np.asarray(wm, dtype=np.float32)
    gm = np.asarray(gm, dtype=np.float32)
    csf = np.asarray(csf, dtype=np.float32)
    if gm.ndim == 4:
        gm = gm[..., 0]
    if csf.ndim == 4:
        csf = csf[..., 0]
    return _cs_dmri.mtnormalise(wm, gm, csf, as_mask(mask, wm.shape), target_sum=target_sum,
                                poly_order=poly_order, niter=niter, balance_maxiter=balance_maxiter,
                                balanced=balanced)


def ss3t_pipeline(dwi, *, responses: ResponseSet | None = None, lmax_wm: int = 8, niter: int = 3,
                  bzero_pct: float = 10.0, normalize: bool = True, mtnorm_target: float | str = MTNORMALISE_TARGET,
                  mtnorm_poly_order: int = 3, mtnorm_balanced: bool = False, diagnostics: bool = False,
                  n_threads: int | None = None, **response_options) -> SS3TFit:
    """Responses (estimated from ``dwi`` unless given) → SS3T → mtnormalise:
    the ``cs-ss3t-full`` pipeline. ``response_options`` go to
    :func:`estimate_responses`."""
    if responses is None:
        responses = dwi.estimate_responses(lmax_wm=lmax_wm, n_threads=n_threads, **response_options)
    fit = SS3TModel(dwi.gtab, responses, lmax_wm=lmax_wm, niter=niter, bzero_pct=bzero_pct).fit(
        dwi, diagnostics=diagnostics, n_threads=n_threads)
    if normalize:
        fit = fit.mtnormalise(target_sum=mtnorm_target, poly_order=mtnorm_poly_order, balanced=mtnorm_balanced)
    return fit
