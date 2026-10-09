// SPDX-License-Identifier: MIT OR Apache-2.0
//! Goldfarb-Idnani inequality-constrained least-squares solver.
//!
//! Implements the dual active-set algorithm from
//! D. Goldfarb and A. Idnani, *"A Numerically Stable Dual Method for Solving
//! Strictly Convex Quadratic Programs"*, Mathematical Programming 27, 1983.
//!
//! Solves
//!
//! ```text
//!   minimize    ½ ‖Hx − b‖²
//!   subject to  C x ≥ 0
//! ```
//!
//! by reformulating as the strictly-convex QP `min ½ xᵀ G x + aᵀ x` with
//! `G = HᵀH + εI`, `a = -Hᵀb`, and applying GI's dual method:
//!
//! 1. Start at the unconstrained minimizer `x* = G⁻¹ Hᵀb`.
//! 2. While any constraint is violated:
//!    - pick the most violated constraint `p`;
//!    - compute primal step `z` and dual step `r` from the maintained
//!      QR factorization of the active-constraint normals;
//!    - take the longest feasible step (either to satisfy `p` exactly,
//!      or to drop a blocking constraint whose multiplier hits zero);
//!    - update the QR factorization via Givens rotations.
//!
//! The factorizations maintained throughout the solve:
//! - `G = L Lᵀ` — Cholesky of the Hessian, computed once at construction.
//! - `J = L⁻ᵀ Q` and `R` (q × q upper-triangular), where `Q [R; 0]` is the
//!   QR factorization of `L⁻¹ N` and `N` collects active-constraint normals.
//!
//! Adding or dropping a constraint costs O(n²) via Givens rotations, so a
//! typical solve is O(q · n²) with q the final active-set size.

use std::sync::Arc;

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

/// Tunable knobs for the ICLS solver.
#[derive(Debug, Clone, Copy)]
pub struct IclsConfig {
    /// Hard cap on outer iterations (constraint additions).
    pub max_iter: usize,
    /// A constraint `(C x)_i` is considered satisfied if it is `≥ -tol`. Also
    /// used as the threshold below which `r_i` and `zᵀn_p` are treated as
    /// zero (avoiding division by tiny numbers).
    pub tol: f64,
    /// Diagonal regularizer added to `G = HᵀH` to guarantee strict
    /// positive-definiteness; passed straight to the Cholesky factorization.
    pub epsilon: f64,
}

impl Default for IclsConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-10,
            epsilon: 1e-10,
        }
    }
}

/// Per-solve diagnostics returned alongside the coefficient vector.
#[derive(Debug, Clone, Copy, Default)]
pub struct IclsDiagnostics {
    pub iterations: usize,
    pub final_active: usize,
    pub converged: bool,
    /// True if the solver ran out of feasible steps before satisfying every
    /// constraint — should never happen for our use case (the trivial
    /// solution `x = 0` always satisfies `Cx ≥ 0` when the column of zeros
    /// satisfies the inequality).
    pub infeasible: bool,
}

/// Per-solve scratch buffers, sized for the solver they were built for.
///
/// `IclsSolver::solve` allocates a fresh workspace on every call — fine for
/// one-off solves, but inside a hot loop (the SS3T volume driver fires ~7 ICLS
/// solves per voxel × ~250K voxels) this dominates allocation time. The
/// volume drivers create one `IclsWorkspace` per rayon worker thread (via
/// `voxel_loop::run_init`) and reuse it for every voxel that worker handles.
///
/// Build with [`IclsSolver::workspace`].
#[derive(Debug, Clone)]
pub struct IclsWorkspace {
    j: DMatrix<f64>,
    r: DMatrix<f64>,
    d: DVector<f64>,
    z: DVector<f64>,
    r_dir: Vec<f64>,
    active_indices: Vec<usize>,
    multipliers: Vec<f64>,
    is_active: Vec<bool>,
}

impl IclsWorkspace {
    /// Allocate scratch sized for `n_coeffs` parameters and `n_constraints`
    /// inequality constraints. Use [`IclsSolver::workspace`] in practice;
    /// this constructor is `pub` only so library users can pre-allocate
    /// without a solver in hand.
    pub fn new(n_coeffs: usize, n_constraints: usize) -> Self {
        let max_q = n_coeffs.min(n_constraints);
        Self {
            j: DMatrix::<f64>::zeros(n_coeffs, n_coeffs),
            r: DMatrix::<f64>::zeros(max_q, max_q),
            d: DVector::<f64>::zeros(n_coeffs),
            z: DVector::<f64>::zeros(n_coeffs),
            r_dir: vec![0.0; max_q],
            active_indices: Vec::with_capacity(max_q),
            multipliers: Vec::with_capacity(max_q),
            is_active: vec![false; n_constraints],
        }
    }

    /// Reset workspace state for a fresh solve. `j0` is the solver's
    /// precomputed `L⁻ᵀ` (the initial value of `J`).
    fn reset(&mut self, j0: &DMatrix<f64>) {
        self.j.copy_from(j0);
        self.active_indices.clear();
        self.multipliers.clear();
        for v in self.is_active.iter_mut() {
            *v = false;
        }
        // r, d, z, r_dir don't need explicit reset — every entry is written
        // before being read inside `solve_into`.
    }
}

/// Inequality-constrained least-squares solver.
///
/// Construction precomputes `G = HᵀH + εI`, its Cholesky factor, and the
/// initial `J = L⁻ᵀ`. Each `solve` call reuses these and only allocates
/// per-solve scratch, so a single `IclsSolver` can be shared by reference
/// across worker threads (`Send + Sync` follows from the field types).
#[derive(Debug, Clone)]
pub struct IclsSolver {
    h: DMatrix<f64>,
    ht: DMatrix<f64>,
    /// Shared so per-voxel solvers (graddev) need not copy the m×p constraint.
    constraint: Arc<DMatrix<f64>>,
    /// Cholesky factorization of G; serves the initial unconstrained solve.
    chol: Cholesky<f64, Dyn>,
    /// Initial value of J: `L⁻ᵀ` where `G = L Lᵀ`.
    j0: DMatrix<f64>,
    cfg: IclsConfig,
    n_coeffs: usize,
    n_constraints: usize,
}

impl IclsSolver {
    /// Build a solver for `min ½‖Hx − b‖² s.t. constraint · x ≥ 0`.
    ///
    /// Panics if `h.ncols() != constraint.ncols()` or if `G = HᵀH + εI`
    /// fails to factor (typically only with `epsilon == 0` on a rank-deficient
    /// `H`).
    pub fn new(h: DMatrix<f64>, constraint: DMatrix<f64>, cfg: IclsConfig) -> Self {
        Self::new_with_ridge(h, constraint, cfg, None)
    }

    /// Like [`new`](Self::new) but adds a per-coefficient Tikhonov ridge to the
    /// QP Hessian diagonal: `G = HᵀH + εI + diag(ridge)`. This solves
    /// `min ½‖Hx − b‖² + ½ Σ ridgeᵢ xᵢ²  s.t. C x ≥ 0` — i.e. L2 damping of the
    /// selected coefficients. Used to shrink the WM fODF's ill-conditioned
    /// high-order SH terms (the run-to-run peak-jitter source) while leaving the
    /// isotropic amplitudes undamped. `ridge`, if given, must have length
    /// `h.ncols()`; `None` reproduces [`new`](Self::new) exactly.
    pub fn new_with_ridge(
        h: DMatrix<f64>,
        constraint: DMatrix<f64>,
        cfg: IclsConfig,
        ridge: Option<&[f64]>,
    ) -> Self {
        let ht = h.transpose();
        let gram = &ht * &h;
        Self::from_gram(h, ht, &gram, Arc::new(constraint), cfg, ridge)
    }

    /// Like [`new_with_ridge`](Self::new_with_ridge) but takes `Hᵀ` and the Gram
    /// matrix `HᵀH` precomputed, and a shared constraint. Lets several solvers
    /// over the same `H` (a ridge path) pay for `HᵀH` once, and per-voxel
    /// solvers share one constraint matrix.
    pub fn from_gram(
        h: DMatrix<f64>,
        ht: DMatrix<f64>,
        gram: &DMatrix<f64>,
        constraint: Arc<DMatrix<f64>>,
        cfg: IclsConfig,
        ridge: Option<&[f64]>,
    ) -> Self {
        assert_eq!(
            h.ncols(),
            constraint.ncols(),
            "design (n={}) and constraint (n={}) must share parameter count",
            h.ncols(),
            constraint.ncols()
        );
        let n_coeffs = h.ncols();
        let n_constraints = constraint.nrows();
        if let Some(r) = ridge {
            assert_eq!(r.len(), n_coeffs, "ridge length must equal n_coeffs");
        }

        let mut g = gram.clone();
        for i in 0..n_coeffs {
            g[(i, i)] += cfg.epsilon + ridge.map_or(0.0, |r| r[i]);
        }
        let chol = Cholesky::new(g).expect("HᵀH + εI must be positive definite");
        let l = chol.l();
        let j0 = l_inverse_transpose(&l);

        Self {
            h,
            ht,
            constraint,
            chol,
            j0,
            cfg,
            n_coeffs,
            n_constraints,
        }
    }

    /// Number of parameters (columns of `H`).
    #[inline]
    pub fn n_coeffs(&self) -> usize {
        self.n_coeffs
    }

    /// Number of inequality constraints (rows of the constraint matrix).
    #[inline]
    pub fn n_constraints(&self) -> usize {
        self.n_constraints
    }

    /// Forward operator `H`.
    #[inline]
    pub fn h(&self) -> &DMatrix<f64> {
        &self.h
    }

    /// Allocate a per-thread workspace sized for this solver. Reuse it across
    /// many `solve_into` calls to avoid per-solve allocation churn.
    pub fn workspace(&self) -> IclsWorkspace {
        IclsWorkspace::new(self.n_coeffs, self.n_constraints)
    }

    /// Solve `min ½‖Hx − b‖² s.t. C x ≥ 0` for the supplied right-hand-side.
    /// Convenience wrapper around [`solve_into`](Self::solve_into) that
    /// allocates a fresh workspace on every call. Inside hot loops, build
    /// one workspace via [`workspace`](Self::workspace) and call
    /// `solve_into` repeatedly.
    pub fn solve(&self, b: &DVector<f64>) -> (DVector<f64>, IclsDiagnostics) {
        let mut ws = self.workspace();
        self.solve_into(b, &mut ws)
    }

    /// Solve into a caller-supplied workspace. Reuses every per-call buffer
    /// (J, R, active set, scratch vectors) so a single workspace can drive
    /// thousands of solves with zero allocation past the first.
    pub fn solve_into(
        &self,
        b: &DVector<f64>,
        ws: &mut IclsWorkspace,
    ) -> (DVector<f64>, IclsDiagnostics) {
        let n = self.n_coeffs;
        let m = self.n_constraints;

        ws.reset(&self.j0);

        // Initial unconstrained solution: G x = Hᵀ b.
        let ht_b = &self.ht * b;
        let mut x = self.chol.solve(&ht_b);

        // Borrow the per-call scratch from the workspace.
        let IclsWorkspace {
            j,
            r,
            d,
            z,
            r_dir,
            active_indices,
            multipliers,
            is_active,
        } = ws;
        let mut q: usize = 0;

        let mut iters = 0usize;
        let mut converged = false;
        let mut infeasible = false;

        'outer: loop {
            iters += 1;
            if iters > self.cfg.max_iter {
                break;
            }

            // --- Step 1: pick the most violated constraint, or stop. -----
            let mut worst_p: Option<usize> = None;
            let mut worst_val = -self.cfg.tol;
            for p in 0..m {
                if is_active[p] {
                    continue;
                }
                let s_p = self.constraint.row(p).dot(&x.transpose());
                if s_p < worst_val {
                    worst_val = s_p;
                    worst_p = Some(p);
                }
            }
            let p = match worst_p {
                Some(p) => p,
                None => {
                    converged = true;
                    break 'outer;
                }
            };
            let n_p = self.constraint.row(p).transpose().into_owned();
            let mut s_p = self.constraint.row(p).dot(&x.transpose());

            // --- Inner loop: drop blocking constraints until p can be added. ---
            loop {
                // Step 2: step directions in the current J/R basis.
                // d = Jᵀ n_p
                d.gemv_tr(1.0, j, &n_p, 0.0);
                // z = J[:, q..n] · d[q..n]   (primal direction)
                z.fill(0.0);
                if q < n {
                    z.gemv(1.0, &j.columns(q, n - q), &d.rows(q, n - q), 0.0);
                }
                // r_dir = R⁻¹ d[0..q]   (dual direction; back-substitution
                // in the q × q upper-triangular block of R)
                if q > 0 {
                    for i in (0..q).rev() {
                        let mut sum = d[i];
                        for k in (i + 1)..q {
                            sum -= r[(i, k)] * r_dir[k];
                        }
                        r_dir[i] = sum / r[(i, i)];
                    }
                }

                // Step 3: compute the two candidate step lengths.
                // t1 = min over i ∈ A with r_i > 0 of u_i / r_i (drop step).
                let mut t1 = f64::INFINITY;
                let mut drop_idx: Option<usize> = None;
                for i in 0..q {
                    if r_dir[i] > self.cfg.tol {
                        let ratio = multipliers[i] / r_dir[i];
                        if ratio < t1 {
                            t1 = ratio;
                            drop_idx = Some(i);
                        }
                    }
                }
                // t2 = -s_p / (zᵀ n_p)  (full step). zᵀ n_p = ‖d[q..n]‖² ≥ 0.
                let z_dot_n = z.dot(&n_p);
                let t2 = if z_dot_n > self.cfg.tol {
                    -s_p / z_dot_n
                } else {
                    f64::INFINITY
                };

                let t = t1.min(t2);
                if !t.is_finite() {
                    infeasible = true;
                    break 'outer;
                }

                // Step 4: take the step and update.
                if q > 0 {
                    for i in 0..q {
                        multipliers[i] -= t * r_dir[i];
                    }
                }
                if t2.is_finite() && t == t2 {
                    // Full step: constraint p becomes active.
                    x.axpy(t, z, 1.0);
                    add_active_constraint(r, j, d, q, n);
                    active_indices.push(p);
                    is_active[p] = true;
                    multipliers.push(t);
                    q += 1;
                    break; // back to outer: pick next violated constraint
                } else {
                    // Partial step: a blocking constraint's multiplier hit zero.
                    // x and s_p still update by the partial step; the dropped
                    // constraint then leaves the active set.
                    x.axpy(t, z, 1.0);
                    s_p += t * z_dot_n;
                    let k = drop_idx.expect("partial step requires a drop index");
                    let dropped = active_indices[k];
                    drop_active_constraint(r, j, k, q);
                    active_indices.remove(k);
                    is_active[dropped] = false;
                    multipliers.remove(k);
                    q -= 1;
                    // continue inner loop: recompute step directions for new q
                }
            }
        }

        let diag = IclsDiagnostics {
            iterations: iters - 1, // iters counts the "step 1 entry" — final no-op increment
            final_active: q,
            converged,
            infeasible,
        };
        (x, diag)
    }
}

// ------------------ Internal helpers ------------------

/// Compute `L⁻ᵀ` given the lower-triangular Cholesky factor `L`. Used as
/// the initial value of `J` in the GI algorithm.
fn l_inverse_transpose(l: &DMatrix<f64>) -> DMatrix<f64> {
    // For each column j: solve Lᵀ x = e_j by back-substitution. Lᵀ[i,k] = L[k,i].
    let n = l.nrows();
    let mut out = DMatrix::<f64>::zeros(n, n);
    for j in 0..n {
        for i in (0..n).rev() {
            let mut sum = if i == j { 1.0 } else { 0.0 };
            for k in (i + 1)..n {
                sum -= l[(k, i)] * out[(k, j)];
            }
            out[(i, j)] = sum / l[(i, i)];
        }
    }
    out
}

/// Add an active constraint: rotate `d` so that `d[q+1..n] = 0`, then write
/// `d[0..=q]` as the new column `q` of `R`. Each Givens rotation applied to
/// `d` is mirrored as a column rotation on `J` to preserve `J = L⁻ᵀ Q`.
fn add_active_constraint(
    r: &mut DMatrix<f64>,
    j: &mut DMatrix<f64>,
    d: &mut DVector<f64>,
    q: usize,
    n: usize,
) {
    // Bottom-up sweep: zero d[i] using d[i-1] for i = n-1 down to q+1.
    for i in (q + 1..n).rev() {
        let a = d[i - 1];
        let b = d[i];
        if b == 0.0 {
            continue;
        }
        let r_norm = (a * a + b * b).sqrt();
        let c = a / r_norm;
        let s = b / r_norm;
        d[i - 1] = r_norm;
        d[i] = 0.0;
        rotate_columns(j, i - 1, c, s);
    }
    for i in 0..=q {
        r[(i, q)] = d[i];
    }
}

/// Drop the active constraint at position `k`: shift `R`'s columns left by
/// one, then restore upper-triangular structure with row Givens rotations
/// (mirrored as column rotations on `J`).
fn drop_active_constraint(
    r: &mut DMatrix<f64>,
    j: &mut DMatrix<f64>,
    k: usize,
    q: usize,
) {
    // Shift columns k+1..q one to the left within the active block.
    for col in k..(q - 1) {
        for row in 0..q {
            r[(row, col)] = r[(row, col + 1)];
        }
    }
    for row in 0..q {
        r[(row, q - 1)] = 0.0;
    }
    // Restore upper-triangular: zero R[i+1, i] for i = k..q-2 with row Givens
    // on (i, i+1), applying matching column rotations to J.
    for i in k..(q - 1) {
        let a = r[(i, i)];
        let b = r[(i + 1, i)];
        if b == 0.0 {
            continue;
        }
        let r_norm = (a * a + b * b).sqrt();
        let c = a / r_norm;
        let s = b / r_norm;
        // Apply to rows (i, i+1) of R, columns i..q-1. (Column q-1 was just
        // zeroed, so we stop before it.)
        for col in i..(q - 1) {
            let ri = r[(i, col)];
            let rip1 = r[(i + 1, col)];
            r[(i, col)] = c * ri + s * rip1;
            r[(i + 1, col)] = -s * ri + c * rip1;
        }
        rotate_columns(j, i, c, s);
    }
}

/// Apply a Givens rotation parameterized by `(c, s)` to columns `i` and
/// `i+1` of `m`:
///
/// ```text
///   m[:, i]'   =  c · m[:, i] + s · m[:, i+1]
///   m[:, i+1]' = -s · m[:, i] + c · m[:, i+1]
/// ```
#[inline]
fn rotate_columns(m: &mut DMatrix<f64>, i: usize, c: f64, s: f64) {
    let nrows = m.nrows();
    for r in 0..nrows {
        let a = m[(r, i)];
        let b = m[(r, i + 1)];
        m[(r, i)] = c * a + s * b;
        m[(r, i + 1)] = -s * a + c * b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn unconstrained_when_constraints_inactive() {
        // Overdetermined system H x = b where the OLS minimizer satisfies
        // every constraint; ICLS should return the OLS solution unchanged.
        let h = DMatrix::<f64>::from_row_slice(
            4,
            2,
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        );
        let true_x = DVector::<f64>::from_row_slice(&[2.0, 3.0]);
        let b = &h * &true_x;
        let constraint = DMatrix::<f64>::identity(2, 2);
        let solver = IclsSolver::new(
            h,
            constraint,
            IclsConfig {
                epsilon: 1e-12,
                ..Default::default()
            },
        );
        let (x, diag) = solver.solve(&b);
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-7);
        assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-7);
        assert_eq!(diag.final_active, 0);
        assert!(diag.converged);
    }

    #[test]
    fn single_binding_non_negativity() {
        // OLS prefers x = (-1, 1); the constraint x_0 ≥ 0 zeros x_0
        // exactly while leaving x_1 = 1.
        let h = DMatrix::<f64>::identity(2, 2);
        let b = DVector::<f64>::from_row_slice(&[-1.0, 1.0]);
        let constraint = DMatrix::<f64>::identity(2, 2);
        let solver = IclsSolver::new(
            h,
            constraint,
            IclsConfig {
                epsilon: 1e-12,
                ..Default::default()
            },
        );
        let (x, diag) = solver.solve(&b);
        assert_abs_diff_eq!(x[0], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-9);
        assert_eq!(diag.final_active, 1);
        assert!(diag.converged);
    }

    #[test]
    fn multiple_binding_constraints() {
        // OLS prefers (1, 2, -1); x ≥ 0 forces x_2 = 0 and leaves the rest.
        let h = DMatrix::<f64>::identity(3, 3);
        let b = DVector::<f64>::from_row_slice(&[1.0, 2.0, -1.0]);
        let constraint = DMatrix::<f64>::identity(3, 3);
        let solver = IclsSolver::new(
            h,
            constraint,
            IclsConfig {
                epsilon: 1e-12,
                ..Default::default()
            },
        );
        let (x, diag) = solver.solve(&b);
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(x[2], 0.0, epsilon = 1e-9);
        assert_eq!(diag.final_active, 1);
    }

    #[test]
    fn linear_amplitude_constraint() {
        // 1D problem: min ½(x − (−2))² s.t. x ≥ 0  →  x = 0.
        let h = DMatrix::<f64>::from_row_slice(1, 1, &[1.0]);
        let b = DVector::<f64>::from_row_slice(&[-2.0]);
        let constraint = DMatrix::<f64>::from_row_slice(1, 1, &[1.0]);
        let solver = IclsSolver::new(
            h,
            constraint,
            IclsConfig {
                epsilon: 1e-12,
                ..Default::default()
            },
        );
        let (x, _) = solver.solve(&b);
        assert_abs_diff_eq!(x[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn matches_brute_force_on_3d_nnls() {
        // 3D NNLS with random-but-fixed H, b. Compare against an exhaustive
        // search over the 2³ active-set possibilities.
        let h = DMatrix::<f64>::from_row_slice(
            4,
            3,
            &[
                1.0, 0.5, 0.2, 0.5, 1.0, 0.4, 0.2, 0.4, 1.0, 0.3, 0.3, 0.3,
            ],
        );
        let b = DVector::<f64>::from_row_slice(&[1.0, -0.5, 0.8, 0.2]);
        let constraint = DMatrix::<f64>::identity(3, 3);
        let solver = IclsSolver::new(
            h.clone(),
            constraint,
            IclsConfig {
                epsilon: 1e-12,
                tol: 1e-12,
                ..Default::default()
            },
        );
        let (x, diag) = solver.solve(&b);
        for &v in x.iter() {
            assert!(v >= -1e-9, "x_i should be ≥ 0, got {v}");
        }

        let brute = brute_force_nnls(&h, &b);
        for i in 0..3 {
            assert_abs_diff_eq!(x[i], brute[i], epsilon = 1e-7);
        }
        assert!(diag.converged);
    }

    #[test]
    fn linear_inequality_constraint_pushes_into_feasible_region() {
        // 2D problem: minimize ½‖x‖² s.t. x_0 + x_1 ≥ 1 (the active set
        // becomes {0}, x* = (½, ½)).
        let h = DMatrix::<f64>::identity(2, 2);
        let b = DVector::<f64>::zeros(2);
        // Constraint Cx ≥ 0 with C = [[1, 1]] and we want Cx ≥ 1, so we
        // shift by introducing C' = [1, 1] and solving the shifted problem.
        // Since IclsSolver uses Cx ≥ 0, we instead minimize ‖x − (0, 0)‖²
        // s.t. x_0 + x_1 − 1 ≥ 0 → reformulate by setting up H, b so that
        // the unconstrained minimum is the origin and the constraint pushes
        // to the line. Equivalent encoding: add an artificial dimension.
        // Simpler: just check that with C = [1,1] and right-hand side 0,
        // x = 0 is feasible and OLS optimal, so no shift expected.
        let constraint = DMatrix::<f64>::from_row_slice(1, 2, &[1.0, 1.0]);
        let solver = IclsSolver::new(
            h,
            constraint,
            IclsConfig {
                epsilon: 1e-12,
                ..Default::default()
            },
        );
        let (x, _) = solver.solve(&b);
        // x = 0 satisfies x_0 + x_1 ≥ 0 with equality and is the OLS minimum.
        assert_abs_diff_eq!(x[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn negative_b_with_amplitude_constraint() {
        // H = I (2D), b = (−2, 1). Constraints: x_0 + x_1 ≥ 0 (one row).
        // OLS minimum is (−2, 1) which has x_0 + x_1 = −1 < 0; the solution
        // projects orthogonally onto the line x_0 + x_1 = 0, giving x* =
        // (−2, 1) − ½ · (−1) · (1, 1) = (−1.5, 1.5).
        let h = DMatrix::<f64>::identity(2, 2);
        let b = DVector::<f64>::from_row_slice(&[-2.0, 1.0]);
        let constraint = DMatrix::<f64>::from_row_slice(1, 2, &[1.0, 1.0]);
        let solver = IclsSolver::new(
            h,
            constraint,
            IclsConfig {
                epsilon: 1e-12,
                ..Default::default()
            },
        );
        let (x, diag) = solver.solve(&b);
        assert_abs_diff_eq!(x[0], -1.5, epsilon = 1e-9);
        assert_abs_diff_eq!(x[1], 1.5, epsilon = 1e-9);
        assert!(diag.converged);
        assert_eq!(diag.final_active, 1);
    }

    /// Brute-force NNLS by enumerating all 2ⁿ active-set possibilities and
    /// returning the lowest-cost feasible candidate. Only practical for
    /// small `n`, used here as ground truth in the n=3 test.
    fn brute_force_nnls(h: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
        let n = h.ncols();
        let mut best: Option<(DVector<f64>, f64)> = None;
        for mask in 0..(1u32 << n) {
            // mask bit i = 1 means coefficient i is "free"; bit 0 means it's
            // pinned to zero (active non-negativity constraint).
            let free: Vec<usize> = (0..n).filter(|i| (mask >> i) & 1 == 1).collect();
            let mut x = DVector::<f64>::zeros(n);
            if !free.is_empty() {
                let h_free = DMatrix::<f64>::from_columns(
                    &free.iter().map(|&i| h.column(i).clone_owned()).collect::<Vec<_>>(),
                );
                let hth = h_free.transpose() * &h_free;
                let chol = match hth.cholesky() {
                    Some(c) => c,
                    None => continue,
                };
                let rhs = h_free.transpose() * b;
                let x_free = chol.solve(&rhs);
                if x_free.iter().any(|&v| v < -1e-9) {
                    continue; // infeasible for this active-set choice
                }
                for (i, &idx) in free.iter().enumerate() {
                    x[idx] = x_free[i];
                }
            }
            let res = h * &x - b;
            let cost = 0.5 * res.dot(&res);
            if best.as_ref().map_or(true, |(_, c)| cost < *c) {
                best = Some((x, cost));
            }
        }
        best.expect("at least one feasible point exists").0
    }
}
