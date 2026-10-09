// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compressed-sensing reconstruction of diffusion MRI data.
//!
//! Loads a 4D DWI + bval/bvec/mask, fits a regularized basis-coefficient field
//! (3D-SHORE in v1), writes coefficients as a 4D NIfTI plus a JSON sidecar, and
//! supports the reverse pass: synthesize a new 4D DWI from coefficients and a
//! new gradient table.

// The allocator is a process-wide choice, so it is opt-out: the Python
// extension builds without it and leaves allocation to the host interpreter.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod basis;
pub mod dti;
pub mod fit;
pub mod graddev;
pub mod io;
pub mod multitissue;
pub mod odf;
pub mod progress;
pub mod qc;
pub mod qspace;
pub mod scalars;
pub mod sh;
pub mod solver;
pub mod synth;
pub mod voxel_loop;

mod math;

pub use basis::{Basis, BasisMetadata, RegularizationDiagonals, shore::ShoreBasis};
pub use fit::{
    AlphaStrategyFit, FitConfig, fit_volume, fit_volume_reporting,
    fit_volume_with_alpha_strategy, fit_volume_with_alpha_strategy_reporting,
    fit_volume_with_alpha_strategy_l2_anchored_reporting,
};
pub use io::coeffs::{CoefficientsFile, SidecarMetadata};
pub use io::dwi::{DwiData, load_dwi, load_dwi_mrtrix_grad};
pub use io::provenance::{Provenance, ProvenanceBuilder, ProvenanceMode};
pub use io::{atomic_write, atomic_write_pair};
pub use progress::Heartbeat;
pub use qspace::GradientTable;
pub use solver::{
    AlphaConfigurable, AlphaPath, AlphaStrategy, FitDiagnostics, Problem, Solver,
    VoxelAlphaResult, alpha_max, fista::FistaSolver, tikhonov::TikhonovSolver,
};
pub use synth::{synthesize_volume, synthesize_volume_reporting};

/// Crate version (`Cargo.toml` `[package].version`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git SHA captured by `build.rs` at compile time. `"unknown"` when built
/// outside a git checkout; suffix `-dirty` when the working tree had
/// uncommitted changes.
pub const GIT_SHA: &str = env!("CS_DMRI_GIT_SHA");

/// Build timestamp (UTC, ISO 8601) captured by `build.rs`.
pub const BUILD_TIMESTAMP: &str = env!("CS_DMRI_BUILD_TS");

#[derive(Debug, thiserror::Error)]
pub enum CsDmriError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("NIfTI error: {0}")]
    Nifti(#[from] nifti::NiftiError),

    #[error("dimension mismatch: {0}")]
    Dimension(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("fit error: {0}")]
    Fit(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, CsDmriError>;

/// Where the rayon thread cap came from, for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadSource {
    /// User passed `--threads N`.
    Cli,
    /// Picked up from `$SLURM_CPUS_PER_TASK` (cgroup-aware on Slurm clusters).
    Slurm,
    /// Picked up from `$RAYON_NUM_THREADS`.
    Rayon,
    /// No cap set; rayon uses one worker per logical CPU.
    DefaultAllCpus,
}

impl ThreadSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "--threads",
            Self::Slurm => "SLURM_CPUS_PER_TASK",
            Self::Rayon => "RAYON_NUM_THREADS",
            Self::DefaultAllCpus => "default-all-cpus",
        }
    }
}

/// Pin rayon's global thread pool. Resolves the cap in this order:
///
/// 1. Explicit `--threads N` (`cli`).
/// 2. `$SLURM_CPUS_PER_TASK` if set and parseable.
/// 3. `$RAYON_NUM_THREADS` if set and parseable.
/// 4. Default — rayon uses one worker per logical CPU.
///
/// Returns `(threads, source)` so the caller can log which knob was honored.
/// `threads == 0` means "rayon default" (case 4 only).
///
/// `rayon::ThreadPoolBuilder::build_global` may only be called once per
/// process; calling this twice (or after rayon has already been used) returns
/// `CsDmriError::Other`.
pub fn configure_rayon_threads(cli: Option<usize>) -> Result<(usize, ThreadSource)> {
    let (threads, source) = if let Some(n) = cli {
        if n == 0 {
            return Err(CsDmriError::Other(
                "--threads must be ≥ 1; omit the flag to keep the rayon default".into(),
            ));
        }
        (n, ThreadSource::Cli)
    } else if let Some(n) = parse_positive_env("SLURM_CPUS_PER_TASK") {
        (n, ThreadSource::Slurm)
    } else if let Some(n) = parse_positive_env("RAYON_NUM_THREADS") {
        (n, ThreadSource::Rayon)
    } else {
        return Ok((0, ThreadSource::DefaultAllCpus));
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| CsDmriError::Other(format!("failed to set rayon thread count: {e}")))?;
    Ok((threads, source))
}

fn parse_positive_env(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// Effective number of rayon worker threads, after `configure_rayon_threads`
/// has run. Useful for the provenance block.
pub fn effective_thread_count() -> usize {
    rayon::current_num_threads()
}
