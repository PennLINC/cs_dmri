# SPDX-License-Identifier: MIT OR Apache-2.0
"""NIfTI grid checks and header-preserving output, via nibabel."""

from __future__ import annotations

import warnings

import numpy as np

AFFINE_ATOL_MM = 1e-3
OBLIQUE_TOL_DEG = 0.1


class SpatialWarning(UserWarning):
    """A NIfTI header or grid condition worth knowing about (not fatal)."""


def image_warnings(img) -> list[str]:
    """Header conditions that don't stop processing but may matter downstream:
    qform/sform disagreement, header zooms that don't match the affine, and
    oblique acquisition grids."""
    import nibabel as nib

    out = []
    hdr = img.header
    qform, qcode = hdr.get_qform(coded=True)
    sform, scode = hdr.get_sform(coded=True)
    if qcode and scode and not np.allclose(qform, sform, atol=AFFINE_ATOL_MM):
        diff = float(np.abs(qform - sform).max())
        out.append(f"qform and sform differ (max {diff:.3g} mm); nibabel uses the "
                   f"{'sform' if scode else 'qform'} as the affine")
    zooms = np.asarray(hdr.get_zooms()[:3], dtype=np.float64)
    affine_zooms = np.linalg.norm(img.affine[:3, :3], axis=0)
    if not np.allclose(zooms, affine_zooms, rtol=1e-3, atol=1e-4):
        out.append(f"header voxel sizes {tuple(np.round(zooms, 4))} disagree with the affine's "
                   f"{tuple(np.round(affine_zooms, 4))}")
    obl = np.degrees(nib.affines.obliquity(img.affine))
    if np.max(np.abs(obl)) > OBLIQUE_TOL_DEG:
        out.append(f"oblique acquisition grid ({np.max(np.abs(obl)):.2f}° off the closest axes); "
                   "directions are interpreted on the voxel grid")
    return out


def describe_grid(img) -> str:
    import nibabel as nib

    return f"shape {img.shape[:3]}, axes {''.join(nib.aff2axcodes(img.affine))}, zooms " \
           f"{tuple(round(float(z), 4) for z in np.linalg.norm(img.affine[:3, :3], axis=0))}"


def check_same_grid(img, ref, what: str = "mask") -> None:
    """Raise if ``img`` is not on ``ref``'s voxel grid (3-D shape and affine)."""
    if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine, atol=AFFINE_ATOL_MM):
        diff = float(np.abs(img.affine - ref.affine).max()) if img.affine.shape == ref.affine.shape else float("nan")
        raise ValueError(
            f"{what} is not on the DWI's grid: {what} has {describe_grid(img)}; DWI has {describe_grid(ref)}"
            + (f"; affines differ by up to {diff:.3g} mm" if np.isfinite(diff) and diff > AFFINE_ATOL_MM else ""))


def emit(messages: list[str], source: str) -> None:
    for m in messages:
        warnings.warn(f"{source}: {m}", SpatialWarning, stacklevel=3)


def make_image(data: np.ndarray, affine: np.ndarray, header=None):
    """A Nifti1Image of ``data`` that inherits ``header`` (qform/sform codes,
    units, orientation) when given. Data are stored as float32 (bool/ints as
    uint8 or as given) with scaling reset."""
    import nibabel as nib

    data = np.asarray(data)
    if data.dtype == bool:
        data = data.astype(np.uint8)
    elif data.dtype.kind == "f":
        data = data.astype(np.float32, copy=False)
    if header is None:
        return nib.Nifti1Image(data, affine)
    hdr = header.copy()
    hdr.set_data_dtype(data.dtype)
    hdr.set_slope_inter(1, 0)
    hdr.set_intent("none")
    hdr["cal_min"] = hdr["cal_max"] = 0
    # Spatial zooms from the source; a 4th axis that isn't the source's time
    # axis (SH or model coefficients) gets unit spacing.
    zooms = tuple(header.get_zooms()[:3])
    hdr.set_data_shape(data.shape)
    hdr.set_zooms(zooms + (1.0,) * (data.ndim - 3))
    img = nib.Nifti1Image(data, affine, header=hdr)
    # Keep the source's qform/sform codes (nibabel resets them otherwise).
    _, qcode = header.get_qform(coded=True)
    _, scode = header.get_sform(coded=True)
    img.set_qform(affine, int(qcode) if qcode else 1)
    img.set_sform(affine, int(scode) if scode else 1)
    return img
