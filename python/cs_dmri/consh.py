# SPDX-License-Identifier: MIT OR Apache-2.0
"""Continuous-b multi-tissue CSD (CONSH) for non-shelled q-space.

CONSH fits a white-matter fODF and gray-matter and CSF compartments to
diffusion data whose b-values do not form shells, against tissue responses
defined on a dense grid of b-values (row ``k`` is the response at
``b = k * step``). On shelled data it reduces to MSMT-CSD.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from os import PathLike
from pathlib import Path

import numpy as np

from . import _cs_dmri
from ._frame import fit_frame
from ._inputs import as_mask, resolve
from .gradients import as_gradient_table
from .multitissue import MTNORMALISE_TARGET, TissueResponse, mtnormalise

__all__ = ["DenseBResponses", "estimate_dense_b_responses", "ConshModel", "ConshFit", "consh_pipeline"]

_ESTIMATORS = ("auto", "kernel", "smooth", "dki")
_TISSUES = ("wm", "gm", "csf")


@dataclass(frozen=True)
class DenseBResponses:
    """WM, GM and CSF dense-b responses. Each tissue's ``coeffs`` has one row
    per dense-b grid point; row ``k`` is the response at ``b = k * step``.

    ``estimator`` names the row estimator when the responses were estimated
    from a series (``None`` for loaded or pooled responses); ``diagnostics``
    holds the tissue-selection counts and, for ``"dki"``, the fitted
    parameters.
    """

    wm: TissueResponse
    gm: TissueResponse
    csf: TissueResponse
    step: float = 50.0
    estimator: str | None = None
    diagnostics: dict = field(default_factory=dict, repr=False)

    @property
    def b(self) -> np.ndarray:
        """b-value of each row, in s/mm²."""
        return self.step * np.arange(len(self.wm.coeffs))

    @classmethod
    def load(cls, directory: str | PathLike, step: float = 50.0) -> "DenseBResponses":
        """Read ``{wm,gm,csf}_response_denseb.txt`` (the layout of
        ``consh --write-responses-to``)."""
        d = Path(directory)
        return cls(*(TissueResponse.from_mrtrix_txt(d / f"{t}_response_denseb.txt") for t in _TISSUES),
                   step=float(step))

    def save(self, directory: str | PathLike, *, overwrite: bool = False) -> dict:
        """Write ``{wm,gm,csf}_response_denseb.txt`` into ``directory``, readable
        by ``consh --read-responses-from``; returns the paths."""
        d = Path(directory)
        paths = {t: d / f"{t}_response_denseb.txt" for t in _TISSUES}
        existing = [p for p in paths.values() if p.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"refusing to overwrite {existing[0]}; pass overwrite=True")
        d.mkdir(parents=True, exist_ok=True)
        for t, p in paths.items():
            getattr(self, t).to_mrtrix_txt(p)
        return paths

    @staticmethod
    def pool(responses, *, statistic: str = "mean", legacy: bool = False) -> "DenseBResponses":
        """Pool per-scan responses into one group response (``consh-response
        mean``). Each input is rescaled by a magnitude multiplier before pooling,
        as in MRtrix3's ``responsemean``; ``legacy=True`` pools without
        rescaling. ``statistic`` is ``"mean"`` or ``"median"``. All inputs must
        share the dense-b grid (maximum b and step) and the WM SH order. The
        multipliers and each input's deviation from the pool are in
        ``diagnostics``."""
        responses = list(responses)
        steps = {r.step for r in responses}
        if len(steps) != 1:
            raise ValueError(f"responses have different grid steps {sorted(steps)}")
        r = _cs_dmri.consh_pool_responses([x._triples() for x in responses], statistic=statistic,
                                          legacy=legacy)
        diagnostics = {t: r[f"{t}_pooling"] for t in _TISSUES}
        return DenseBResponses(*(TissueResponse(*r[t]) for t in _TISSUES), step=steps.pop(),
                               diagnostics=diagnostics)

    def _triples(self):
        return tuple(getattr(self, t)._pair() for t in _TISSUES)


def _check_estimator(estimator: str) -> str:
    if estimator not in _ESTIMATORS:
        raise ValueError(f"estimator must be one of {_ESTIMATORS}, got {estimator!r}")
    return estimator


def _graddev_array(graddev, shape, affine):
    """A gradient-deviation field as a float32 (X, Y, Z, 9) array on the DWI grid.

    ``graddev`` may be a path, a nibabel image (checked against the DWI's grid)
    or an array already on that grid, of shape (X, Y, Z, 9) or (X, Y, Z, 3, 3).
    """
    if graddev is None:
        return None
    if affine is None:
        raise ValueError("a gradient-deviation field needs the series' affine (fit a DWI, or pass affine=)")
    if isinstance(graddev, (str, PathLike)) or hasattr(graddev, "dataobj"):
        import nibabel as nib

        img = graddev if hasattr(graddev, "dataobj") else nib.load(graddev)
        if img.shape[:3] != tuple(shape) or not np.allclose(img.affine, affine, atol=1e-3):
            raise ValueError(
                f"the gradient-deviation field (shape {img.shape}) is not on the DWI grid (shape "
                f"{tuple(shape)}); resample it to the DWI first")
        graddev = np.asarray(img.dataobj, dtype=np.float32)
    a = np.asarray(graddev, dtype=np.float32)
    if a.shape[:3] != tuple(shape):
        raise ValueError(f"graddev spatial shape {a.shape[:3]} does not match the DWI's {tuple(shape)}")
    a = a.reshape(a.shape[:3] + (-1,))
    if a.shape[3] != 9:
        raise ValueError(f"graddev must have 9 components per voxel, got {a.shape[3]}")
    return np.ascontiguousarray(a)


def _series(data, mask, gtab, affine, bvec_frame):
    """Volume, mask, affine, fit frame and rotation for the CONSH bindings."""
    vol, m, dwi_affine, dwi = resolve(data, mask, gtab)
    affine = dwi_affine if affine is None else np.asarray(affine, dtype=np.float64)
    frame, R = fit_frame(bvec_frame, affine)
    return vol, m, affine, frame, np.asarray(R, dtype=np.float64), dwi


def estimate_dense_b_responses(dwi, gtab=None, mask=None, *, affine=None, graddev=None,
                               graddev_identity: str = "auto", estimator: str = "auto",
                               b_interp: bool = False, step: float = 50.0, sigma_b: float = 150.0,
                               lmax_wm: int = 8, legacy_selection: bool = False, erode: int = 3,
                               fa: float = 0.2, sfwm_pct: float = 0.5, gm_pct: float = 2.0,
                               csf_pct: float = 10.0, fa_wm_threshold: float = 0.7,
                               fiber_dominance_ratio: float = 2.0, md_csf_pct: float = 2.5,
                               restore_max_iter: int = 50, restore_tol: float = 1e-6,
                               bvec_frame: str = "auto", n_threads: int | None = None) -> DenseBResponses:
    """Dense-b WM, GM and CSF responses estimated from a series, as ``consh``
    estimates them: RESTORE tensor fit, Dhollander tissue selection (MRtrix3's
    staged algorithm, or the earlier threshold-based one with
    ``legacy_selection``), and one of three row estimators.

    ``estimator``: ``"kernel"`` fits each row from a Gaussian window in b of
    width ``sigma_b``; ``"smooth"`` fits all rows jointly, smooth in b and
    constrained; ``"dki"`` fits the parametric DKI-plus-offset response of
    Morez et al. (2021); ``"auto"`` chooses between the first two as ``consh``
    does. With a gradient-deviation field (``graddev``) every voxel is pooled at
    its effective b-values and directions, and the grid reaches the largest
    effective b. ``dwi`` is a :class:`~cs_dmri.DWI` or an array with ``gtab``.
    """
    gtab = as_gradient_table(gtab if gtab is not None else dwi.gtab)
    vol, m, affine, frame, R, _ = _series(dwi, mask, gtab, affine, bvec_frame)
    r = _cs_dmri.consh_estimate_responses(
        vol, gtab.bvals, gtab.bvecs, m, R, world=frame == "world", affine=affine,
        b0_threshold=gtab.b0_threshold, graddev=_graddev_array(graddev, vol.shape[:3], affine),
        graddev_identity=graddev_identity, estimator=_check_estimator(estimator), b_interp=b_interp,
        step=step, sigma_b=sigma_b, lmax_wm=lmax_wm, legacy_selection=legacy_selection, erode=erode,
        fa=fa, sfwm_pct=sfwm_pct, gm_pct=gm_pct, csf_pct=csf_pct, fa_wm_threshold=fa_wm_threshold,
        fiber_dominance_ratio=fiber_dominance_ratio, md_csf_pct=md_csf_pct,
        restore_max_iter=restore_max_iter, restore_tol=restore_tol, n_threads=n_threads)
    return DenseBResponses(*(TissueResponse(*r[t]) for t in _TISSUES), step=float(step),
                           estimator=r["estimator"], diagnostics=r["diagnostics"])


class ConshModel:
    """Continuous-b multi-tissue CSD with fixed dense-b responses.

    Parameters
    ----------
    gtab : GradientTable or dipy gradient table
        Any sampling of q-space; shells are not required.
    responses : DenseBResponses
        From :func:`estimate_dense_b_responses`, :meth:`DenseBResponses.load`
        or :meth:`DenseBResponses.pool`. Their rows must cover the series'
        dense-b grid.
    lmax_wm : int
        WM fODF order (clamped to the WM response's).
    b_interp : bool
        Interpolate the response linearly in b between grid rows instead of
        using the nearest row. Implied by a gradient-deviation field.
    noise_floor : bool
        Fit a constant compartment that absorbs the magnitude noise floor.
    quantitative : bool
        Scale each voxel's coefficients by its b=0 signal (absolute signal
        units) instead of S/S0.
    wm_ridge, wm_ridge_anchor, wm_ridge_relative
        L2 penalty on the WM fODF coefficients with l ≥ 2 (see ``consh
        --help``).
    bvec_frame : {"auto", "world", "image"}
        Frame of the WM fODF SH. ``"auto"``: world RAS when an affine is known.
    """

    def __init__(self, gtab, responses: DenseBResponses, *, lmax_wm: int = 8, b_interp: bool = False,
                 noise_floor: bool = True, quantitative: bool = False, wm_ridge: float = 0.0,
                 wm_ridge_anchor: float = 0.0, wm_ridge_relative: bool = False, icls_max_iter: int = 2000,
                 bvec_frame: str = "auto"):
        self.gtab = as_gradient_table(gtab)
        self.responses = responses
        self.lmax_wm = lmax_wm
        self.b_interp = b_interp
        self.noise_floor = noise_floor
        self.quantitative = quantitative
        self.ridge = dict(wm_ridge=wm_ridge, wm_ridge_anchor=wm_ridge_anchor,
                          wm_ridge_relative=wm_ridge_relative)
        self.icls_max_iter = icls_max_iter
        self.bvec_frame = bvec_frame

    def fit(self, data, mask=None, *, affine=None, graddev=None, graddev_identity: str = "auto",
            reference_mask=None, diagnostics: bool = False, n_threads: int | None = None) -> "ConshFit":
        """Fit every voxel in the mask. With ``graddev`` (a path, nibabel image or
        (X, Y, Z, 9) array on the DWI grid) each voxel uses its own effective
        gradient table."""
        return _run(self.gtab, data, mask, affine, self.bvec_frame, responses=self.responses,
                    graddev=graddev, graddev_identity=graddev_identity, lmax_wm=self.lmax_wm,
                    b_interp=self.b_interp, noise_floor=self.noise_floor, quantitative=self.quantitative,
                    icls_max_iter=self.icls_max_iter, normalize=False, reference_mask=reference_mask,
                    diagnostics=diagnostics, n_threads=n_threads, step=self.responses.step, **self.ridge)


@dataclass(frozen=True)
class ConshFit:
    """Tissue maps from CONSH: ``wm`` (X, Y, Z, n_sh) Tournier SH, ``gm``,
    ``csf`` and ``noise_floor`` (X, Y, Z; ``noise_floor`` is ``None`` when not
    fitted). ``normalization`` holds mtnormalise's diagnostics once applied;
    ``reference_b0`` is the median b=0 in white matter and where it came from,
    recorded in exported sidecars."""

    wm: np.ndarray
    gm: np.ndarray
    csf: np.ndarray
    noise_floor: np.ndarray | None
    lmax_wm: int
    mask: np.ndarray = field(repr=False)
    affine: np.ndarray | None = field(repr=False)
    frame: str
    responses: DenseBResponses = field(repr=False)
    quantitative: bool = False
    b_interp: bool = False
    external_responses: bool = True
    reference_b0: tuple | None = None
    source: str | None = None
    diagnostics: dict = field(default_factory=dict, repr=False)
    warnings: tuple[str, ...] = ()
    normalization: dict | None = field(default=None, repr=False)

    def mtnormalise(self, *, target_sum: float | str = MTNORMALISE_TARGET, poly_order: int = 3,
                    niter: int = 15, balance_maxiter: int = 7, balanced: bool = False) -> "ConshFit":
        """Multi-tissue intensity normalization (MRtrix3 ``mtnormalise``); returns
        a new fit. The noise-floor map is divided by the same bias field, so it
        stays on the scale of the GM and CSF amplitudes."""
        wm, gm, csf, diag = mtnormalise(self.wm, self.gm, self.csf, self.mask, target_sum=target_sum,
                                        poly_order=poly_order, niter=niter,
                                        balance_maxiter=balance_maxiter, balanced=balanced)
        floor = self.noise_floor
        if floor is not None:
            field_ = np.asarray(diag["bias_field"])
            floor = np.where(self.mask & (field_ > 0), floor / np.where(field_ > 0, field_, 1), floor)
            floor = floor.astype(np.float32)
        return replace(self, wm=wm, gm=gm, csf=csf, noise_floor=floor, normalization=diag)

    def export(self, path: str | PathLike, *, format: str | None = None, fixel_container: str = "nifti",
               overwrite: bool = False, peak_min_amplitude: float = 0.0,
               peak_min_amplitude_frac: float = 0.0) -> None:
        """Write the WM fODF SH, GM, CSF, the noise floor, the mask and WM peaks,
        like ``consh --odx``, with the dense-b responses and a quantitative
        sidecar in the ODX header. ``format`` is one of
        :data:`cs_dmri.EXPORT_FORMATS`; when omitted it is inferred from the
        extension. Peaks below ``peak_min_amplitude`` (or below
        ``peak_min_amplitude_frac`` of the median dominant-peak amplitude) are
        dropped."""
        if self.affine is None:
            raise ValueError("exporting needs an affine")
        if self.frame != "world":
            raise ValueError("exports are world-space: fit with bvec_frame='world' "
                             "(the default when an affine is known)")
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass overwrite=True")
        n = self.normalization or {}
        lognorm = n.get("lognorm_scale")
        _cs_dmri.consh_export(
            path, np.asarray(self.affine, dtype=np.float64), self.mask, self.wm, self.gm, self.csf,
            self.lmax_wm, *self.responses._triples(), self.responses.step, noise_floor=self.noise_floor,
            format=format, fixel_container=fixel_container, overwrite=overwrite,
            peak_min_amplitude=peak_min_amplitude, peak_min_amplitude_frac=peak_min_amplitude_frac,
            quantitative=self.quantitative, reference_b0=self.reference_b0,
            mtnorm_target=n.get("target_sum"),
            mtnorm_scale=(1.0 / lognorm) if lognorm else None,
            external_responses=self.external_responses, b_interp=self.b_interp,
            estimator=self.responses.estimator if not self.external_responses else None,
            source=self.source)

    def to_odx(self, path: str | PathLike, *, directory: bool = False, overwrite: bool = False) -> None:
        """Write an ODX archive, or an ODX directory with ``directory=True``."""
        self.export(path, format="odx-directory" if directory else "odx-archive", overwrite=overwrite)


def _run(gtab, data, mask, affine, bvec_frame, *, responses: DenseBResponses | None, graddev,
         graddev_identity, reference_mask, n_threads, **options) -> ConshFit:
    vol, m, affine, frame, R, dwi = _series(data, mask, gtab, affine, bvec_frame)
    ref = None if reference_mask is None else as_mask(reference_mask, vol.shape)
    r = _cs_dmri.consh_run(
        vol, gtab.bvals, gtab.bvecs, m, R, world=frame == "world", affine=affine,
        b0_threshold=gtab.b0_threshold, responses=None if responses is None else responses._triples(),
        graddev=_graddev_array(graddev, vol.shape[:3], affine), graddev_identity=graddev_identity,
        reference_mask=ref, n_threads=n_threads, **options)
    if responses is None:
        responses = DenseBResponses(*(TissueResponse(*r["responses"][t]) for t in _TISSUES),
                                    step=float(options["step"]), estimator=r["estimator"],
                                    diagnostics=r.get("response_diagnostics", {}))
    for message in r["warnings"]:
        warnings.warn(message, stacklevel=3)
    diag = {k: r[k] for k in ("converged", "iterations", "grid_bmax", "b_shift_p95") if r.get(k) is not None}
    return ConshFit(
        wm=r["wm"], gm=r["gm"], csf=r["csf"], noise_floor=r.get("noise_floor"), lmax_wm=options["lmax_wm"],
        mask=m, affine=affine, frame=frame, responses=responses, quantitative=options["quantitative"],
        b_interp=r["b_interp"], external_responses=r["estimator"] is None,
        reference_b0=r["reference_b0"], source=getattr(dwi, "source", None), diagnostics=diag,
        warnings=tuple(r["warnings"]),
        normalization=r.get("normalization"))


def consh_pipeline(dwi, *, responses: DenseBResponses | None = None, graddev=None,
                   graddev_identity: str = "auto", estimator: str = "auto", step: float = 50.0,
                   sigma_b: float = 150.0, lmax_wm: int = 8, b_interp: bool = False, noise_floor: bool = True,
                   quantitative: bool = False, normalize: bool = True,
                   mtnorm_target: float | str = MTNORMALISE_TARGET, mtnorm_poly_order: int = 3,
                   mtnorm_balanced: bool = False, wm_ridge: float = 0.0, wm_ridge_anchor: float = 0.0,
                   wm_ridge_relative: bool = False, icls_max_iter: int = 2000, reference_mask=None,
                   diagnostics: bool = False, n_threads: int | None = None, **response_options) -> ConshFit:
    """Dense-b responses (estimated from ``dwi`` unless given) → CONSH →
    mtnormalise: the ``consh`` pipeline in one call, with identical results.

    ``response_options`` go to the tissue selection and tensor fit (see
    :func:`estimate_dense_b_responses`). ``quantitative=True`` writes fODFs in
    absolute signal units and requires ``normalize``; for comparisons across
    subjects, pass one group-average response (:meth:`DenseBResponses.pool`)
    for every scan. ``mtnorm_target="median"`` keeps the median observed sum.
    """
    if quantitative and not normalize:
        raise ValueError("quantitative=True requires normalize=True")
    if isinstance(mtnorm_target, str):
        if mtnorm_target != "median":
            raise ValueError("mtnorm_target must be a number or 'median'")
        mtnorm_target = None
    if responses is not None:
        step = responses.step
    allowed = {"legacy_selection", "erode", "fa", "sfwm_pct", "gm_pct", "csf_pct", "fa_wm_threshold",
               "fiber_dominance_ratio", "md_csf_pct", "restore_max_iter", "restore_tol"}
    unknown = set(response_options) - allowed
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
    return _run(dwi.gtab, dwi, None, None, "auto", responses=responses, graddev=graddev,
                graddev_identity=graddev_identity, reference_mask=reference_mask, n_threads=n_threads,
                estimator=_check_estimator(estimator), step=step, sigma_b=sigma_b, lmax_wm=lmax_wm,
                b_interp=b_interp, noise_floor=noise_floor, quantitative=quantitative, wm_ridge=wm_ridge,
                wm_ridge_anchor=wm_ridge_anchor, wm_ridge_relative=wm_ridge_relative,
                icls_max_iter=icls_max_iter, normalize=normalize, mtnorm_target=mtnorm_target,
                mtnorm_poly_order=mtnorm_poly_order, mtnorm_balanced=mtnorm_balanced,
                diagnostics=diagnostics, **response_options)
