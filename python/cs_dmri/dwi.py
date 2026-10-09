# SPDX-License-Identifier: MIT OR Apache-2.0
"""The :class:`DWI` series object."""

from __future__ import annotations

from functools import cached_property
from os import PathLike

import numpy as np

from . import _cs_dmri
from . import _spatial
from ._inputs import as_mask, as_volume
from .gradients import GradientTable, as_gradient_table

__all__ = ["DWI"]


class DWI:
    """A diffusion-weighted series: data, gradient table, affine and mask.

    Work several features share is computed once and cached: the fallback
    mask (:attr:`b0_mask`) and the RESTORE tensor fit (:attr:`tensor`), which
    QC and response estimation both use. The object is effectively immutable;
    :meth:`with_mask` returns a new one sharing the same data.

    Loading through nibabel also checks the grid: masks must be on the DWI's
    grid, and qform/sform disagreement, header zooms that do not match the
    affine, and oblique grids are reported in :attr:`warnings` and as
    :class:`~cs_dmri.SpatialWarning`.

    Parameters
    ----------
    data : array (X, Y, Z, N)
        Stored as float32 (no copy if it already is).
    gtab : GradientTable, dipy gradient table, or ``(bvals, bvecs)``
    affine : array (4, 4), optional
        Voxel-to-world transform. Needed for anything world-space (voxel size,
        fixel coherence, ODX output).
    mask : array (X, Y, Z), optional
        Brain mask. Without one, :attr:`effective_mask` falls back to
        :attr:`b0_mask`.
    header : nibabel header, optional
        The source NIfTI header. Everything cs_dmri writes for this series
        uses it as the template, so qform/sform codes and units survive.
        :meth:`from_files` / :meth:`from_nibabel` set it.
    """

    def __init__(self, data, gtab, *, affine=None, mask=None, header=None):
        self.data = as_volume(data)
        self.gtab = as_gradient_table(gtab)
        if len(self.gtab) != self.data.shape[3]:
            raise ValueError(f"data has {self.data.shape[3]} volumes but the gradient table has {len(self.gtab)}")
        self.affine = None if affine is None else np.asarray(affine, dtype=np.float64)
        if self.affine is not None and self.affine.shape != (4, 4):
            raise ValueError(f"affine must be 4x4, got {self.affine.shape}")
        self.mask = as_mask(mask, self.data.shape)
        self.header = header
        self.warnings: tuple[str, ...] = ()

    # ------------------------------------------------------------ construction

    @classmethod
    def from_files(cls, dwi: str | PathLike, bval: str | PathLike, bvec: str | PathLike, *,
                   mask: str | PathLike | np.ndarray | None = None, b0_threshold: float = 50.0,
                   big_delta: float | None = None, small_delta: float | None = None) -> "DWI":
        """Load a NIfTI series with FSL ``.bval`` / ``.bvec`` (and optional mask
        NIfTI) via nibabel."""
        import nibabel as nib

        gtab = GradientTable.from_files(bval, bvec, b0_threshold=b0_threshold,
                                        big_delta=big_delta, small_delta=small_delta)
        return cls.from_nibabel(nib.load(dwi), gtab, mask=mask)

    @classmethod
    def from_nibabel(cls, img, gtab, *, mask=None) -> "DWI":
        """From a nibabel image; ``mask`` may be an array, a nibabel image or a
        path. Image masks must be on the DWI's grid (shape and affine)."""
        import nibabel as nib

        if len(img.shape) != 4:
            raise ValueError(f"expected a 4-D DWI series, got shape {img.shape}")
        found = _spatial.image_warnings(img)
        if mask is not None and not isinstance(mask, np.ndarray):
            mask_img = mask if hasattr(mask, "dataobj") else nib.load(mask)
            if len(mask_img.shape) == 4 and mask_img.shape[3] == 1:
                mask_img = mask_img.slicer[..., 0]
            if len(mask_img.shape) != 3:
                raise ValueError(f"mask must be 3-D, got shape {mask_img.shape}")
            _spatial.check_same_grid(mask_img, img, "mask")
            mask = np.asarray(mask_img.dataobj) > 0
        _spatial.emit(found, "DWI")
        dwi = cls(np.asarray(img.dataobj, dtype=np.float32), gtab, affine=img.affine, mask=mask,
                  header=img.header.copy())
        dwi.warnings = tuple(found)
        return dwi

    def with_mask(self, mask) -> "DWI":
        """Same data and gradients, different mask (mask-dependent caches reset)."""
        out = DWI.__new__(DWI)
        out.data, out.gtab, out.affine = self.data, self.gtab, self.affine
        out.header, out.warnings = self.header, self.warnings
        if mask is not None and hasattr(mask, "dataobj"):
            _spatial.check_same_grid(mask, self.to_image(self.data[..., 0]), "mask")
            mask = np.asarray(mask.dataobj) > 0
        out.mask = as_mask(mask, self.data.shape)
        if "b0_mask" in self.__dict__:
            out.__dict__["b0_mask"] = self.__dict__["b0_mask"]
        return out

    # -------------------------------------------------------------- properties

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.data.shape

    @property
    def zooms(self) -> tuple[float, float, float] | None:
        """Voxel size in mm: the header's zooms when there is a header, else
        the affine's column norms."""
        if self.header is not None:
            return tuple(float(x) for x in self.header.get_zooms()[:3])
        if self.affine is None:
            return None
        return tuple(float(x) for x in np.linalg.norm(self.affine[:3, :3], axis=0))

    def to_image(self, data):
        """A nibabel image of a map on this DWI's grid (3-D or 4-D), inheriting
        the source header (qform/sform codes, units)."""
        if self.affine is None:
            raise ValueError("this DWI has no affine")
        data = np.asarray(data)
        if data.shape[:3] != self.shape[:3]:
            raise ValueError(f"data shape {data.shape} is not on the DWI grid {self.shape[:3]}")
        return _spatial.make_image(data, self.affine, self.header)

    @cached_property
    def b0_mask(self) -> np.ndarray:
        """Fallback mask: mean b=0 above 1% of its maximum."""
        return _cs_dmri.b0_mask(self.data, self.gtab.bvals, b0_threshold=self.gtab.b0_threshold)

    @property
    def effective_mask(self) -> np.ndarray:
        """The mask models and QC use: :attr:`mask` if given, else :attr:`b0_mask`."""
        return self.mask if self.mask is not None else self.b0_mask

    @property
    def mask_source(self) -> str:
        return "provided" if self.mask is not None else "auto-b0"

    @cached_property
    def tensor(self):
        """RESTORE fit inside :attr:`effective_mask` (default settings), computed once."""
        from .dti import RestoreModel

        return RestoreModel(self.gtab).fit(self)

    # ---------------------------------------------------------------- analyses

    def qc(self, *, coherence: bool = True, slice_axis: int = 2, min_slice_voxels: int = 100,
           slice_smoothing_sigma: float = 2.0, slice_threshold: float = 2.5,
           n_threads: int | None = None):
        """All QC metrics (see :mod:`cs_dmri.qc`). Masked metrics, outlier slices
        and coherence use :attr:`effective_mask`; coherence reuses
        :attr:`tensor`."""
        from .qc import assess

        tensor = self.tensor if (coherence and self.affine is not None) else None
        return assess(
            self.data, self.gtab, self.effective_mask, affine=self.affine, coherence=coherence,
            tensor_fit=tensor, slice_axis=slice_axis, min_slice_voxels=min_slice_voxels,
            slice_smoothing_sigma=slice_smoothing_sigma, slice_threshold=slice_threshold,
            mask_source=self.mask_source, voxel_size=self.zooms, extra_warnings=self.warnings,
            n_threads=n_threads)

    def estimate_responses(self, **options):
        """Dhollander WM/GM/CSF responses inside :attr:`effective_mask`, reusing
        :attr:`tensor`. See :func:`cs_dmri.multitissue.estimate_responses`."""
        from .multitissue import estimate_responses

        return estimate_responses(self, self.gtab, tensor_fit=self.tensor, **options)

    def __repr__(self) -> str:
        return f"DWI(shape={self.shape}, {self.gtab!r}, mask={self.mask_source})"
