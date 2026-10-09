// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cross-check cs_dmri's FISTA against scikit-learn's coordinate-descent
//! Lasso on a battery of synthetic problems.
//!
//! Fixtures live under `tests/data/lasso_fixtures/` as one `.bin` per case
//! (custom binary format, see `scripts/generate_lasso_fixtures.py` for the
//! exact layout) listed in `manifest.json`. Generated offline (no Python at
//! CI time) by:
//!
//! ```bash
//! mamba activate trx
//! python scripts/generate_lasso_fixtures.py
//! ```
//!
//! Per-fixture asserts:
//! - **coefficient agreement** with sklearn (relative L₂ < 1e-3),
//! - **objective agreement** (relative < 1e-6),
//! - **KKT residual** on cs_dmri's solution (< 1e-5) — proves we converged
//!   to the true minimum, not just sklearn's iterate.

mod common;

use std::fs;
use std::path::PathBuf;

use cs_dmri::solver::fista::FistaSolver;
use cs_dmri::solver::{Problem, Solver};
use nalgebra::{DMatrix, DVector};

use common::{kkt_residual_lasso, kkt_residual_nonneg_lasso, lasso_objective};

const MAGIC: u32 = 0x434C4153; // "CLAS" little-endian
const VERSION: u32 = 1;
// struct layout: <I I I I B 7x d d d  →  4+4+4+4+1+7+8+8+8 = 48 bytes.
const HEADER_LEN: usize = 48;

struct Fixture {
    name: String,
    m: usize,
    n: usize,
    alpha: f64,
    positive: bool,
    design: DMatrix<f64>,
    signal: DVector<f64>,
    expected_coef: DVector<f64>,
    expected_objective: f64,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("lasso_fixtures")
}

fn load_manifest() -> Vec<String> {
    let path = fixtures_dir().join("manifest.json");
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing or unreadable fixture manifest at {:?}: {e}\n\
             Regenerate via:\n  mamba activate trx\n  python scripts/generate_lasso_fixtures.py",
            path
        )
    });
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {:?}: {e}", path))
}

fn load_fixture(file_name: &str) -> Fixture {
    let path = fixtures_dir().join(file_name);
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {:?}: {e}", path));
    if bytes.len() < HEADER_LEN {
        panic!("{:?}: file too short ({} bytes)", path, bytes.len());
    }

    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != MAGIC {
        panic!("{:?}: bad magic 0x{magic:08x}, expected 0x{MAGIC:08x}", path);
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        panic!("{:?}: unsupported version {version}, expected {VERSION}", path);
    }
    let m = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let positive = bytes[16] != 0;
    // bytes[17..24] is 7-byte padding aligning the f64 fields to 8 bytes.
    let alpha = f64::from_le_bytes(bytes[24..32].try_into().unwrap());
    let _alpha_max = f64::from_le_bytes(bytes[32..40].try_into().unwrap());
    let expected_objective = f64::from_le_bytes(bytes[40..48].try_into().unwrap());

    let design_bytes = m * n * 8;
    let signal_bytes = m * 8;
    let coef_bytes = n * 8;
    let expected_total = HEADER_LEN + design_bytes + signal_bytes + coef_bytes;
    if bytes.len() != expected_total {
        panic!(
            "{:?}: size {} != expected {} (m={m}, n={n})",
            path,
            bytes.len(),
            expected_total
        );
    }

    let mut cursor = HEADER_LEN;
    let design_flat = read_f64_slice(&bytes[cursor..cursor + design_bytes], m * n);
    cursor += design_bytes;
    let signal_vec = read_f64_slice(&bytes[cursor..cursor + signal_bytes], m);
    cursor += signal_bytes;
    let expected_coef_vec = read_f64_slice(&bytes[cursor..cursor + coef_bytes], n);

    // Python writes design row-major; nalgebra is column-major.
    let design = DMatrix::<f64>::from_row_iterator(m, n, design_flat.into_iter());
    let signal = DVector::<f64>::from_vec(signal_vec);
    let expected_coef = DVector::<f64>::from_vec(expected_coef_vec);

    Fixture {
        name: file_name.trim_end_matches(".bin").to_string(),
        m,
        n,
        alpha,
        positive,
        design,
        signal,
        expected_coef,
        expected_objective,
    }
}

fn read_f64_slice(bytes: &[u8], expected_count: usize) -> Vec<f64> {
    debug_assert_eq!(bytes.len(), expected_count * 8);
    let mut out = Vec::with_capacity(expected_count);
    for chunk in bytes.chunks_exact(8) {
        out.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    out
}

#[test]
fn matches_sklearn_lasso_on_all_fixtures() {
    let manifest = load_manifest();
    assert!(!manifest.is_empty(), "no fixtures listed in manifest");

    let mut failures: Vec<String> = Vec::new();
    for file in &manifest {
        let fx = load_fixture(file);
        if let Err(msg) = check_fixture(&fx) {
            failures.push(format!("[{}] {msg}", fx.name));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} fixtures failed:\n  {}",
            failures.len(),
            manifest.len(),
            failures.join("\n  ")
        );
    }
}

fn check_fixture(fx: &Fixture) -> Result<(), String> {
    let _ = (fx.m, fx.n); // shapes already validated in load_fixture
    let solver = FistaSolver::new(fx.design.clone(), fx.alpha, 200_000, 1e-12, fx.positive);
    let problem = Problem {
        design: &fx.design,
        signal: &fx.signal,
    };
    let (coef, _diag) = solver.fit(&problem);

    // Coefficient agreement (relative L₂ with floor of 1).
    let coef_diff = (&coef - &fx.expected_coef).norm();
    let ref_norm = fx.expected_coef.norm().max(1.0);
    let coef_rel = coef_diff / ref_norm;
    if coef_rel >= 1e-3 {
        return Err(format!(
            "coefficient gap ‖β̂ − β_ref‖/max(1,‖β_ref‖) = {coef_rel:.3e} ≥ 1e-3"
        ));
    }

    // Objective agreement.
    let our_obj = lasso_objective(&fx.design, &fx.signal, &coef, fx.alpha);
    let obj_diff = (our_obj - fx.expected_objective).abs();
    let obj_rel = obj_diff / fx.expected_objective.abs().max(1.0);
    if obj_rel >= 1e-6 {
        return Err(format!(
            "objective gap |F(β̂) − F_ref|/max(1,|F_ref|) = {obj_rel:.3e} ≥ 1e-6 \
             (ours={our_obj:.6e}, ref={:.6e})",
            fx.expected_objective
        ));
    }

    // KKT residual on cs_dmri's solution.
    let (active, inactive) = if fx.positive {
        kkt_residual_nonneg_lasso(&fx.design, &fx.signal, &coef, fx.alpha)
    } else {
        kkt_residual_lasso(&fx.design, &fx.signal, &coef, fx.alpha)
    };
    if active >= 1e-5 || inactive >= 1e-5 {
        return Err(format!(
            "KKT residual too large: active={active:.3e}, inactive={inactive:.3e}"
        ));
    }

    Ok(())
}
