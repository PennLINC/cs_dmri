import numpy as np
import pytest


def fibre_series(n_dirs=30, shape=(12, 12, 8), seed=7, b=1000.0, n_b0=1):
    """Synthetic single-shell series: per-voxel fibre angle and a shared
    anatomical baseline, so neighbours correlate strongly and perpendicular
    volumes less. Returns ``(data, bvals, bvecs)``."""
    rng = np.random.default_rng(seed)
    t = np.pi * np.arange(n_dirs) / n_dirs
    dirs = np.stack([np.cos(t), np.sin(t), 0.1 * np.ones_like(t)], 1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((n_b0, 3)), dirs])
    bvals = np.concatenate([np.zeros(n_b0), np.full(n_dirs, b)])
    angle = rng.random(shape) * np.pi
    base = 0.5 + rng.random(shape)
    fib = np.stack([np.cos(angle), np.sin(angle), np.zeros(shape)], -1)
    c = np.einsum("xyzk,nk->xyzn", fib, bvecs)
    data = base[..., None] * (0.6 + 0.4 * np.exp(-3 * c**2))
    data += 0.01 * (rng.random(data.shape) - 0.5)
    data[..., :n_b0] = base[..., None]
    return data.astype(np.float32), bvals, bvecs


@pytest.fixture
def series():
    return fibre_series()
