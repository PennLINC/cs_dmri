# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate scikit-learn LASSO reference fixtures for cs_dmri's
optimization-correctness tests.

    python scripts/generate_lasso_fixtures.py

Writes one ``.bin`` per case under ``tests/data/lasso_fixtures/`` and a
sidecar ``manifest.json`` listing the fixture filenames in load order. Commit
the result. CI is pure Rust and never runs this script — the Rust test reads
the .bin files directly with no extra crates.

File format (little-endian throughout):

    magic         u32      0x434C4153  ("CLAS")
    version       u32      1
    m             u32      number of rows in design / signal length
    n             u32      number of columns / coefficient length
    positive      u8       0 (plain LASSO) or 1 (non-negative)
    _padding      u8[7]    align next field to 8 bytes
    alpha         f64      regularization weight (sklearn convention)
    alpha_max     f64      ||M^T y||_inf / m
    expected_objective f64
    design        f64[m*n] row-major
    signal        f64[m]
    expected_coef f64[n]

Total per fixture: 48-byte header + 8 * (m*n + m + n) bytes. Uncompressed —
Gaussian iid floats don't compress meaningfully, so we trade ~0% size for
zero deps and stdlib-only loaders on both sides.

f64 throughout: our test tolerances (coef rel L2 < 1e-3, objective rel < 1e-6)
are tight enough that quantizing to f32 would introduce comparable noise.
"""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso


MAGIC = 0x434C4153  # "CLAS" little-endian
VERSION = 1


# (name, m, n, k, sigma, alpha_ratio, positive)
CASES = [
    ("small_dense", 50, 20, 4, 0.05, 0.1, False),
    ("medium_dense", 200, 50, 8, 0.05, 0.05, False),
    ("underdetermined", 30, 80, 5, 0.05, 0.1, False),
    ("ill_conditioned", 150, 40, 6, 0.10, 0.1, False),
    ("nonneg_small", 80, 30, 5, 0.05, 0.1, True),
    ("nonneg_medium", 200, 60, 8, 0.05, 0.05, True),
    ("near_alpha_max", 100, 40, 3, 0.02, 0.95, False),
    ("very_low_alpha", 150, 50, 6, 0.05, 0.001, False),
]


def gen_design(m: int, n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    M = rng.standard_normal((m, n)) / np.sqrt(m)
    if kind == "ill_conditioned":
        # Push condition number to ~1e3 via SVD rescaling.
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        s = np.geomspace(1.0, 1e-3, num=min(m, n))
        M = (U * s) @ Vt
    return M


def alpha_max(M: np.ndarray, y: np.ndarray) -> float:
    # cs_dmri / sklearn convention: loss = (1/(2m)) ||M beta - y||^2 + alpha ||beta||_1
    return float(np.max(np.abs(M.T @ y)) / M.shape[0])


def write_fixture(
    path: Path,
    *,
    m: int,
    n: int,
    positive: bool,
    alpha: float,
    amax: float,
    expected_objective: float,
    design: np.ndarray,
    signal: np.ndarray,
    expected_coef: np.ndarray,
) -> None:
    assert design.shape == (m, n) and design.dtype == np.float64
    assert signal.shape == (m,) and signal.dtype == np.float64
    assert expected_coef.shape == (n,) and expected_coef.dtype == np.float64

    # 32-byte header: <I I I I B 7x d d d
    header = struct.pack(
        "<IIIIB7xddd",
        MAGIC,
        VERSION,
        m,
        n,
        1 if positive else 0,
        alpha,
        amax,
        expected_objective,
    )
    assert len(header) == 48, f"header length {len(header)}, expected 48"

    with path.open("wb") as f:
        f.write(header)
        # numpy is row-major (C order) by default; tobytes() honors that.
        f.write(np.ascontiguousarray(design).tobytes())
        f.write(np.ascontiguousarray(signal).tobytes())
        f.write(np.ascontiguousarray(expected_coef).tobytes())


def main() -> None:
    out_dir = Path(__file__).parent.parent / "tests" / "data" / "lasso_fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe stale fixtures so renamed/removed cases don't linger.
    for stale in list(out_dir.glob("*.npz")) + list(out_dir.glob("*.bin")):
        stale.unlink()

    fixture_files: list[str] = []
    for name, m, n, k, sigma, ratio, positive in CASES:
        # Deterministic seed across Python interpreters / PYTHONHASHSEED
        # settings. `hash(name)` would salt per process and diff every run.
        seed = int.from_bytes(
            hashlib.sha256(name.encode()).digest()[:4], "little"
        )
        rng = np.random.default_rng(seed)

        kind = "ill_conditioned" if "ill" in name else "gaussian"
        M = gen_design(m, n, kind, rng).astype(np.float64)

        beta_star = np.zeros(n)
        idx = rng.choice(n, size=k, replace=False)
        signs = np.ones(k) if positive else rng.choice([-1.0, 1.0], size=k)
        beta_star[idx] = signs * rng.uniform(0.5, 2.0, size=k)
        y = (M @ beta_star + sigma * rng.standard_normal(m)).astype(np.float64)

        amax = alpha_max(M, y)
        alpha = ratio * amax

        model = Lasso(
            alpha=alpha,
            fit_intercept=False,
            positive=positive,
            max_iter=200_000,
            tol=1e-12,
        )
        model.fit(M, y)
        beta_ref = model.coef_.astype(np.float64)

        residual = M @ beta_ref - y
        obj_ref = float(
            0.5 / m * np.sum(residual ** 2) + alpha * np.sum(np.abs(beta_ref))
        )

        bin_path = out_dir / f"{name}.bin"
        write_fixture(
            bin_path,
            m=m,
            n=n,
            positive=positive,
            alpha=alpha,
            amax=amax,
            expected_objective=obj_ref,
            design=M,
            signal=y,
            expected_coef=beta_ref,
        )
        fixture_files.append(bin_path.name)
        nnz = int(np.sum(np.abs(beta_ref) > 1e-10))
        size_kb = bin_path.stat().st_size / 1024
        print(
            f"  {name:<20} m={m:>4} n={n:>3} alpha={alpha:.4e} "
            f"nnz={nnz:>3}/{n} obj={obj_ref:.6e}  ({size_kb:.1f} KB)"
        )

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(fixture_files, indent=2))

    total_kb = sum((out_dir / f).stat().st_size for f in fixture_files) / 1024
    print(
        f"\nwrote {len(fixture_files)} fixtures to {out_dir} "
        f"(total {total_kb:.1f} KB) + manifest.json"
    )


if __name__ == "__main__":
    main()
