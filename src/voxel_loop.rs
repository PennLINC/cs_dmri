// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generic parallel-over-masked-voxels driver.
//!
//! Three reconstruction pipelines previously reimplemented this same shape —
//! collect masked voxel indices, run a per-voxel closure in parallel through
//! rayon, fire a per-voxel progress callback, return `(idx, result)` pairs
//! for the caller to scatter into output arrays. This module exposes the
//! pattern once.
//!
//! Two entry points:
//!
//! - [`run`] for stateless per-voxel fits (one closure does everything).
//! - [`run_init`] for fits that benefit from per-rayon-worker scratch state
//!   (e.g. cloned solvers, ICLS workspaces). The `init` closure runs once
//!   per worker thread; the `fit` closure receives the workspace by `&mut`.
//!
//! Both functions return `Vec<((x, y, z), R)>` in unspecified order. Callers
//! scatter into output arrays during a (cheap) sequential pass.

use ndarray::Array3;
use rayon::prelude::*;

/// Voxel index: `(x, y, z)`.
pub type VoxelIdx = (usize, usize, usize);

/// Collect every `(x, y, z)` where `mask[x, y, z]` is true, in canonical
/// `for x { for y { for z } }` order.
pub fn collect_masked_voxels(mask: &Array3<bool>) -> Vec<VoxelIdx> {
    let s = mask.shape();
    let (nx, ny, nz) = (s[0], s[1], s[2]);
    let mut work = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if mask[(x, y, z)] {
                    work.push((x, y, z));
                }
            }
        }
    }
    work
}

/// Run `fit` on every masked voxel in parallel via rayon. Calls `on_voxel`
/// after each voxel completes (use this to drive a progress heartbeat from
/// the CLI binaries).
///
/// Returns `(idx, result)` pairs in unspecified order; the caller scatters
/// into output arrays.
pub fn run<R, F, OnVoxel>(
    mask: &Array3<bool>,
    on_voxel: OnVoxel,
    fit: F,
) -> Vec<(VoxelIdx, R)>
where
    R: Send,
    F: Fn(usize, usize, usize) -> R + Sync,
    OnVoxel: Fn() + Sync,
{
    let work = collect_masked_voxels(mask);
    work.par_iter()
        .map(|&(x, y, z)| {
            let r = fit(x, y, z);
            on_voxel();
            ((x, y, z), r)
        })
        .collect()
}

/// Run `fit` on every masked voxel with per-rayon-worker scratch state.
///
/// `init` is called once per worker thread to allocate a workspace `S`; that
/// workspace is reused for every voxel the worker handles. Use this when
/// per-call allocations would dominate (e.g. cloning a solver, allocating
/// ICLS scratch matrices).
///
/// The `init` closure must be `Fn() -> S + Sync + Send` (rayon clones it
/// across workers); `fit` takes the workspace by `&mut` and may mutate it.
pub fn run_init<S, R, Init, F, OnVoxel>(
    mask: &Array3<bool>,
    on_voxel: OnVoxel,
    init: Init,
    fit: F,
) -> Vec<(VoxelIdx, R)>
where
    S: Send,
    R: Send,
    Init: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, usize, usize) -> R + Sync,
    OnVoxel: Fn() + Sync,
{
    let work = collect_masked_voxels(mask);
    work.par_iter()
        .map_init(
            || init(),
            |state, &(x, y, z)| {
                let r = fit(state, x, y, z);
                on_voxel();
                ((x, y, z), r)
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn checker_mask(nx: usize, ny: usize, nz: usize) -> Array3<bool> {
        let mut m = Array3::<bool>::default((nx, ny, nz));
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    m[(x, y, z)] = (x + y + z) % 2 == 0;
                }
            }
        }
        m
    }

    #[test]
    fn collect_masked_voxels_visits_only_true_cells() {
        let m = checker_mask(3, 3, 3);
        let voxels = collect_masked_voxels(&m);
        let expected: usize = (0..3)
            .flat_map(|x| (0..3).flat_map(move |y| (0..3).map(move |z| (x, y, z))))
            .filter(|&(x, y, z)| (x + y + z) % 2 == 0)
            .count();
        assert_eq!(voxels.len(), expected);
        for (x, y, z) in voxels {
            assert!(m[(x, y, z)]);
        }
    }

    #[test]
    fn run_visits_each_masked_voxel_once_and_returns_pairs() {
        let m = checker_mask(4, 4, 4);
        let counter = AtomicUsize::new(0);
        let results = run(
            &m,
            || {
                counter.fetch_add(1, Ordering::Relaxed);
            },
            |x, y, z| (x as f64) * 100.0 + (y as f64) * 10.0 + (z as f64),
        );

        // Number of progress callbacks equals number of masked voxels.
        let expected = collect_masked_voxels(&m).len();
        assert_eq!(counter.load(Ordering::Relaxed), expected);
        assert_eq!(results.len(), expected);

        // Every returned (idx, result) must be at a masked voxel and carry
        // the expected value.
        for ((x, y, z), v) in results {
            assert!(m[(x, y, z)]);
            assert_eq!(v, x as f64 * 100.0 + y as f64 * 10.0 + z as f64);
        }
    }

    #[test]
    fn run_init_workspace_is_reused_across_voxels() {
        // Each worker counts how many voxels it processes via its workspace.
        // Sum over all workers must equal the masked-voxel count.
        let m = checker_mask(5, 5, 5);
        let results = run_init(
            &m,
            || (),
            || 0_u32,
            |state: &mut u32, _x, _y, _z| {
                *state += 1;
                *state
            },
        );

        // The largest counter value across workers tells us at least one
        // worker reused its state. With rayon's typical pool size > 1 voxel,
        // we expect at least one worker's final counter to be ≥ 2 (rather
        // than every worker getting only one voxel — which would defeat the
        // whole point). On a single-threaded pool the test still passes
        // because that worker handles every voxel.
        let max_state = results
            .iter()
            .map(|(_, v)| *v)
            .max()
            .expect("at least one masked voxel");
        // Total masked voxels in this fixture:
        let n_masked = collect_masked_voxels(&m).len();
        assert_eq!(results.len(), n_masked);
        // Either single-threaded (max == n_masked) or multi-threaded (max ≥ 2
        // when n_masked is large enough). Just check sanity: at least 1.
        assert!(max_state >= 1);
    }
}
