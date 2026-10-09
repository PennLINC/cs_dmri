# SPDX-License-Identifier: MIT OR Apache-2.0
"""Image-quality metrics for DWI series.

The usual entry point is :meth:`cs_dmri.DWI.qc`, which returns a
:class:`QCReport`. The functions here are the same metrics on plain arrays.

Metrics (see :func:`column_descriptions` for full definitions):

``ndc`` / ``ndc_masked``
    Neighboring DWI correlation: mean correlation of each b>0 volume with its
    nearest q-space neighbour. Below 0.4 flags a low-quality image.
``dwi_contrast_ratio`` / ``dwi_contrast_ratio_masked``
    Neighbour correlation ÷ correlation with the most nearly perpendicular
    volume. Below 1.1 poor, 1.1–1.3 fair, above 1.3 good. Mask-sensitive.
``n_outlier_slices``
    Slices that don't lie between their two adjacent slices in the same volume.
``fixel_coherence``
    FA-weighted share of voxels whose principal direction continues coherently
    into a neighbour (0–1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _cs_dmri
from ._inputs import as_mask, as_volume
from .gradients import as_gradient_table

__all__ = [
    "QCReport",
    "assess",
    "column_descriptions",
    "columns",
    "dwi_contrast_ratio",
    "fixel_coherence",
    "neighboring_dwi_correlation",
    "outlier_slices",
]

NDC_LOW_THRESHOLD = 0.4
DWI_CONTRAST_POOR_THRESHOLD = 1.1
DWI_CONTRAST_GOOD_THRESHOLD = 1.3


def columns() -> list[str]:
    """QC table column names, in output order."""
    return [c["name"] for c in _cs_dmri.qc_columns()]


def column_descriptions(prefix: str = "") -> dict:
    """BIDS-style data dictionary for the QC table: ``{column: {"LongName",
    "Description", "Units"?, "Replaces"?}}``, ready to dump as the TSV's JSON
    sidecar. ``Replaces`` names the DSI Studio column (as named in qsiprep) a
    renamed column supersedes."""
    out = {}
    for c in _cs_dmri.qc_columns():
        c = dict(c)
        name = c.pop("name")
        if "Replaces" in c and prefix:
            c["Replaces"] = prefix + c["Replaces"]
        out[prefix + name] = c
    return out


@dataclass(frozen=True)
class QCReport:
    """QC metrics for one series. Absent metrics are ``None``."""

    dimensions: tuple[int, int, int]
    voxel_size: tuple[float, float, float] | None
    max_b: float
    n_dwi_volumes: int
    n_b0_volumes: int
    ndc: float | None
    ndc_masked: float | None
    dwi_contrast_ratio: float | None
    dwi_contrast_ratio_masked: float | None
    outlier_slices: np.ndarray = field(repr=False)
    """(n_volumes, n_slices) bool: flagged slices."""
    outlier_ratio: np.ndarray = field(repr=False)
    """(n_volumes, n_slices): each slice's inconsistency ratio (NaN where unscored)."""
    fixel_coherence: float | None = None
    coherence_elasticity: float | None = None
    mask_source: str | None = None
    mask_voxels: int | None = None
    warnings: tuple[str, ...] = ()
    options: dict = field(default_factory=dict, repr=False)

    @property
    def n_outlier_slices(self) -> int:
        return int(self.outlier_slices.sum())

    @property
    def flagged_slices(self) -> list[tuple[int, int]]:
        """``(volume, slice)`` pairs of the flagged slices."""
        return [tuple(int(i) for i in p) for p in np.argwhere(self.outlier_slices)]

    @property
    def contrast_grade(self) -> str | None:
        v = self.dwi_contrast_ratio_masked if self.dwi_contrast_ratio_masked is not None else self.dwi_contrast_ratio
        if v is None or not math.isfinite(v):
            return None
        return "poor" if v < DWI_CONTRAST_POOR_THRESHOLD else "fair" if v <= DWI_CONTRAST_GOOD_THRESHOLD else "good"

    def to_dict(self, prefix: str = "") -> dict:
        """Flat row keyed by :func:`columns` (with ``prefix``); absent values are
        ``None``. This is the row qsiprep writes to ``desc-image_qc.tsv``."""
        vs = self.voxel_size or (None, None, None)
        values = {
            "dimension_x": self.dimensions[0], "dimension_y": self.dimensions[1],
            "dimension_z": self.dimensions[2],
            "voxel_size_x": vs[0], "voxel_size_y": vs[1], "voxel_size_z": vs[2],
            "max_b": self.max_b, "n_dwi_volumes": self.n_dwi_volumes, "n_b0_volumes": self.n_b0_volumes,
            "ndc": self.ndc, "ndc_masked": self.ndc_masked,
            "dwi_contrast_ratio": self.dwi_contrast_ratio,
            "dwi_contrast_ratio_masked": self.dwi_contrast_ratio_masked,
            "n_outlier_slices": self.n_outlier_slices, "fixel_coherence": self.fixel_coherence,
        }
        return {prefix + c: values[c] for c in columns()}

    column_descriptions = staticmethod(column_descriptions)

    def __str__(self) -> str:
        f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        grade = f", {self.contrast_grade}" if self.contrast_grade else ""
        lines = [
            f"NDC {f(self.ndc)} (masked {f(self.ndc_masked)})",
            f"DWI contrast ratio {f(self.dwi_contrast_ratio)} (masked {f(self.dwi_contrast_ratio_masked)}{grade})",
            f"outlier slices {self.n_outlier_slices}",
            f"fixel coherence {f(self.fixel_coherence)}",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _b0(gtab, b0_threshold):
    return gtab.b0_threshold if b0_threshold is None else b0_threshold


def neighboring_dwi_correlation(data, gtab, mask=None, *, b0_threshold=None, n_threads=None):
    """Mean correlation of each b>0 volume with its nearest q-space neighbour.

    Order-invariant (every volume counts) and antipodally symmetric: dipy's
    definition. ``None`` if no volume has a neighbour.
    """
    gtab = as_gradient_table(gtab)
    vol = as_volume(data)
    return _cs_dmri.qc_neighboring_dwi_correlation(
        vol, gtab.bvals, gtab.bvecs, as_mask(mask, vol.shape),
        b0_threshold=_b0(gtab, b0_threshold), n_threads=n_threads)


def dwi_contrast_ratio(data, gtab, mask=None, *, b0_threshold=None, n_threads=None):
    """Mean neighbour correlation ÷ mean correlation with each volume's most
    nearly perpendicular q-space volume. ``None`` if undefined."""
    gtab = as_gradient_table(gtab)
    vol = as_volume(data)
    return _cs_dmri.qc_dwi_contrast_ratio(
        vol, gtab.bvals, gtab.bvecs, as_mask(mask, vol.shape),
        b0_threshold=_b0(gtab, b0_threshold), n_threads=n_threads)


def outlier_slices(data, mask=None, *, axis=2, min_voxels=100, smoothing_sigma=2.0, threshold=2.5,
                   n_threads=None):
    """Within-volume outlier slices: ``(flags, ratio)``, each (n_volumes, n_slices).

    After in-plane Gaussian smoothing, a slice's mean absolute deviation from
    the average of its two adjacent slices is divided by half their mean
    absolute difference; above ``threshold`` it is flagged. No other volume is
    consulted.
    """
    vol = as_volume(data)
    return _cs_dmri.qc_outlier_slices(
        vol, as_mask(mask, vol.shape), axis=axis, min_voxels=min_voxels,
        smoothing_sigma=smoothing_sigma, threshold=threshold, n_threads=n_threads)


def fixel_coherence(principal_dir, fa, mask, affine, *, world_directions=False, quantile=0.1,
                    angle_degrees=15.0, n_threads=None) -> dict:
    """odx-rs primary coherence of a principal-direction field, weighted and
    thresholded by FA. Directions are in the image's voxel-axis frame unless
    ``world_directions``. Returns a dict with ``coherence``,
    ``threshold_elasticity``, ``evaluated_voxels``, ``connected_voxels``,
    ``fa_threshold``."""
    return _cs_dmri.qc_fixel_coherence(
        np.asarray(principal_dir, dtype=np.float32), np.asarray(fa, dtype=np.float32),
        np.asarray(mask, dtype=bool), np.asarray(affine, dtype=np.float64),
        world_directions=world_directions, quantile=quantile, angle_degrees=angle_degrees,
        n_threads=n_threads)


def assess(data, gtab, mask=None, *, affine=None, coherence=True, tensor_fit=None, b0_threshold=None,
           slice_axis=2, min_slice_voxels=100, slice_smoothing_sigma=2.0, slice_threshold=2.5,
           mask_source=None, voxel_size=None, extra_warnings=(), n_threads=None) -> QCReport:
    """All QC metrics for an array. :meth:`cs_dmri.DWI.qc` is the convenient
    form; it reuses the DWI's cached tensor fit for ``fixel_coherence``.

    ``coherence`` needs an ``affine`` and a RESTORE fit (``tensor_fit``, else
    one is computed inside ``mask``).
    """
    gtab = as_gradient_table(gtab)
    vol = as_volume(data)
    m = as_mask(mask, vol.shape)
    b0 = _b0(gtab, b0_threshold)
    r = _cs_dmri.qc_assess(
        vol, gtab.bvals, gtab.bvecs, m, b0_threshold=b0, slice_axis=slice_axis,
        min_slice_voxels=min_slice_voxels, slice_smoothing_sigma=slice_smoothing_sigma,
        slice_threshold=slice_threshold, n_threads=n_threads)
    warnings = list(extra_warnings) + list(r["warnings"])

    coh = elasticity = None
    if coherence:
        if affine is None:
            warnings.append("fixel coherence skipped: no affine")
        elif m is None:
            warnings.append("fixel coherence skipped: no mask")
        else:
            if tensor_fit is None:
                from .dti import RestoreModel
                tensor_fit = RestoreModel(gtab).fit(vol, m, n_threads=n_threads)
            c = fixel_coherence(tensor_fit.principal_dir, tensor_fit.fa, m, affine, n_threads=n_threads)
            coh, elasticity = c["coherence"], c["threshold_elasticity"]

    if voxel_size is not None:
        voxel_size = tuple(float(x) for x in voxel_size)
    elif affine is not None:
        a = np.asarray(affine, dtype=np.float64)
        voxel_size = tuple(float(x) for x in np.linalg.norm(a[:3, :3], axis=0))

    return QCReport(
        dimensions=tuple(r["dimensions"]),
        voxel_size=voxel_size,
        max_b=r["max_b"],
        n_dwi_volumes=r["n_dwi_volumes"],
        n_b0_volumes=r["n_b0_volumes"],
        ndc=r["ndc"],
        ndc_masked=r["ndc_masked"],
        dwi_contrast_ratio=r["dwi_contrast_ratio"],
        dwi_contrast_ratio_masked=r["dwi_contrast_ratio_masked"],
        outlier_slices=r["outlier_flags"],
        outlier_ratio=r["outlier_ratio"],
        fixel_coherence=coh,
        coherence_elasticity=elasticity,
        mask_source=mask_source if mask_source is not None else ("provided" if m is not None else None),
        mask_voxels=r["mask_voxels"],
        warnings=tuple(warnings),
        options=r["options"],
    )
