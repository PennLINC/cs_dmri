# SPDX-License-Identifier: MIT OR Apache-2.0
"""Normalise the ``(data | DWI, mask)`` arguments every model accepts."""

from __future__ import annotations

import numpy as np

from . import _cs_dmri


def as_volume(data) -> np.ndarray:
    """4-D float32 view of ``data`` (copies only if the dtype differs)."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"expected a 4-D (X, Y, Z, N) array, got shape {arr.shape}")
    return arr


def as_mask(mask, shape) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype != bool:
        m = m > 0
    if m.shape != tuple(shape[:3]):
        raise ValueError(f"mask shape {m.shape} does not match data spatial shape {tuple(shape[:3])}")
    return m


def resolve(data, mask, gtab=None):
    """``(data float32, mask bool, affine or None, DWI or None)``.

    ``data`` may be a :class:`~cs_dmri.DWI` (its effective mask and affine are
    used unless ``mask`` overrides) or an array; for an array without a mask the
    b=0 fallback mask is computed.
    """
    from .dwi import DWI

    if isinstance(data, DWI):
        m = as_mask(mask, data.shape) if mask is not None else data.effective_mask
        return data.data, m, data.affine, data
    vol = as_volume(data)
    m = as_mask(mask, vol.shape)
    if m is None:
        if gtab is None:
            raise ValueError("a mask (or a DWI) is required")
        m = _cs_dmri.b0_mask(vol, gtab.bvals, b0_threshold=gtab.b0_threshold)
    return vol, m, None, None
