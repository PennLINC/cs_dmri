// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-voxel scalar maps derived from BrainSuiteSHORE coefficients.
//!
//! See `scripts/microstructure_math.md` for the closed-form derivations
//! (RTOP, RTAP, RTPP, MSD, QIV, NG) and the BrainSuiteSHORE ↔ dipy
//! iso-MAPMRI basis equivalence the formulas rely on.

pub mod microstructure;
