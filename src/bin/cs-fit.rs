// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-fit`: load DWI + bval/bvec/mask → write coefficient NIfTI + JSON sidecar.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use cs_dmri::basis::Basis;
use cs_dmri::io::aux::{ensure_nifti_extension, precheck_writable, sibling_path, write_3d_f32};
use cs_dmri::io::coeffs::{ChosenAlpha, CoefficientsFile, SidecarMetadata, SolverMetadata};
use cs_dmri::io::dwi::load_dwi;
use cs_dmri::qspace::{BvecFrame, TORTOISE_DEFAULT_GMAX};
use cs_dmri::solver::fista::FistaSolver;
use cs_dmri::fit::fit_volume_with_alpha_strategy_l2_anchored_reporting;
use cs_dmri::solver::tikhonov::TikhonovSolver;
use cs_dmri::solver::{AlphaPath, AlphaStrategy};
use cs_dmri::{
    FitConfig, Heartbeat, ProvenanceBuilder, ProvenanceMode, ShoreBasis,
    effective_thread_count, fit_volume_reporting, fit_volume_with_alpha_strategy_reporting,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Regularization {
    /// L1 sparse fit via FISTA. (Default — matches qsirecon's CS path.)
    L1,
    /// L2 closed-form Tikhonov. Fast; useful for parity / smoke tests.
    L2,
    /// Goldfarb-Idnani ICLS with hard non-negativity on the *projected ODF
    /// amplitudes* on a dense sphere. Use when downstream tools (peak
    /// extraction, fixel viewers) need a guaranteed-positive ODF and L2's
    /// post-hoc clamping is too lossy.
    AmpNn,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AlphaMode {
    /// Use a fixed α for every voxel (escape hatch / smoke tests).
    Fixed,
    /// Per voxel: α = ratio · α_max(voxel). Cheap, dimensionless, dataset-agnostic.
    AlphaRatio,
    /// Per voxel: sweep a log-spaced α path α_max → α_max·eps, pick the α
    /// minimizing BIC. Path-BIC is the historical default but mis-fires at
    /// high b-value or low SNR / low FA (over- or under-sparsifies).
    PathBic,
    /// Per voxel: walk the same α path as PathBic, but pick the largest α
    /// whose RSS is within `(1+slack) · RSS_L2`, where the L2 reference is
    /// a per-voxel Tikhonov fit using `--lambda-n` / `--lambda-l`. Falls
    /// back to argmin(RSS) if no α meets the slack. The robust default for
    /// high-b CS-DSI, infant scans, and other low-SNR regimes; matches L2
    /// reliability while keeping L1 sparsity.
    L2Anchored,
}

#[derive(Parser, Debug)]
#[command(version, about = "Compressed-sensing dMRI fit (3D-SHORE basis)")]
struct Cli {
    /// 4D DWI NIfTI input.
    #[arg(long)]
    dwi: PathBuf,
    /// FSL bval file.
    #[arg(long)]
    bval: PathBuf,
    /// FSL bvec file.
    #[arg(long)]
    bvec: PathBuf,
    /// Optional brain mask NIfTI. Auto-generated from b0 mean if not provided.
    #[arg(long)]
    mask: Option<PathBuf>,
    /// Big delta Δ (seconds). If either delta is missing, both are estimated
    /// via TORTOISE's max-bval heuristic.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Small delta δ (seconds).
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude (T/m), used only when deltas are estimated.
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,
    /// Output coefficient NIfTI path.
    #[arg(long)]
    output: PathBuf,

    /// SHORE radial order (must be even-friendly per qsirecon convention; 6 by default).
    #[arg(long, default_value_t = 6)]
    radial_order: u32,
    /// SHORE scale parameter ζ.
    #[arg(long, default_value_t = 700.0)]
    zeta: f64,

    /// Regularization mode.
    #[arg(long, value_enum, default_value_t = Regularization::L1)]
    reg: Regularization,

    /// L1 α-selection strategy. `l2-anchored` is the recommended default —
    /// per voxel it picks the largest α whose RSS is within `(1+slack) ·
    /// RSS_L2`, where the L2 reference is a Tikhonov fit. This caps fit
    /// looseness against L2 directly and avoids the failure modes of
    /// `path-bic` on high-b CS-DSI, infant scans, and other low-SNR
    /// regimes (where BIC over- or under-sparsifies). Use `path-bic` for
    /// qsirecon-parity / historical-default behavior.
    #[arg(long, value_enum, default_value_t = AlphaMode::L2Anchored)]
    alpha_mode: AlphaMode,
    /// L1 sparsity weight α (only used when `--alpha-mode=fixed`).
    #[arg(long, default_value_t = 1.0)]
    alpha: f64,
    /// α / α_max ratio (only used when `--alpha-mode=alpha-ratio`). Typical 1e-3 .. 1e-2.
    #[arg(long, default_value_t = 1e-3)]
    alpha_ratio: f64,
    /// Number of α grid points along the regularization path
    /// (path-bic / l2-anchored).
    #[arg(long, default_value_t = 20)]
    path_n_alphas: usize,
    /// α_min / α_max ratio along the regularization path
    /// (path-bic / l2-anchored). Mode-dependent default: 1e-3 for path-bic
    /// (backward-compatible) and 1e-4 for l2-anchored (lets the slack
    /// constraint bind on every voxel — without this the path doesn't
    /// reach an L2-good fit on high-b CS-DSI data and the selector falls
    /// back). Override explicitly to use a single value across modes.
    #[arg(long)]
    path_eps: Option<f64>,
    /// L2-residual slack for `--alpha-mode=l2-anchored`. Per voxel, the
    /// selected α has RSS ≤ (1 + slack) · RSS_L2. Tighter values (0.02)
    /// match L2's ODF cleanliness; looser (0.10) preserves more sparsity.
    /// 0.05 is a good empirical default across infant + high-b regimes.
    #[arg(long, default_value_t = 0.05)]
    slack: f64,

    /// L1 max iterations (FISTA). Applies to every fit on the path.
    #[arg(long, default_value_t = 1000)]
    max_iter: u32,
    /// L1 convergence tolerance (relative coefficient change).
    #[arg(long, default_value_t = 1e-6)]
    tol: f64,
    /// Enforce non-negative coefficients during the L1 fit.
    #[arg(long)]
    non_negative: bool,

    /// L2 radial regularization weight λ_N.
    #[arg(long, default_value_t = 1e-8)]
    lambda_n: f64,
    /// L2 angular regularization weight λ_L.
    #[arg(long, default_value_t = 1e-8)]
    lambda_l: f64,

    /// `--reg amp-nn` only: Goldfarb-Idnani ICLS active-set max iterations.
    #[arg(long, default_value_t = 200)]
    amp_nn_max_iter: usize,
    /// `--reg amp-nn` only: constraint-satisfaction tolerance. A sphere
    /// direction's amplitude is "satisfied" if it is ≥ -tol.
    #[arg(long, default_value_t = 1e-9)]
    amp_nn_tol: f64,
    /// `--reg amp-nn` only: Tikhonov stabilizer added to HᵀH diagonal.
    /// Increase if Cholesky fails on rank-deficient designs.
    #[arg(long, default_value_t = 1e-10)]
    amp_nn_epsilon: f64,

    /// Also write per-voxel R², residual, iterations, regularization kind, and
    /// (for L1 with a per-voxel α strategy) the chosen α map next to the
    /// coefficient NIfTI.
    #[arg(long)]
    diagnostics: bool,

    /// Keep bvecs in their FSL/image-axis frame instead of rotating them into
    /// world-RAS. The default rotation is what TRXViz and most ODF viewers
    /// assume; pass this flag for byte-for-byte parity with qsirecon /
    /// dipy's `BrainSuiteShoreModel`, which fit in image-axis.
    #[arg(long)]
    no_bvec_rotation: bool,

    /// Cap rayon's worker threads. If unset, picks up `$SLURM_CPUS_PER_TASK`,
    /// then `$RAYON_NUM_THREADS`, else uses one worker per logical CPU.
    #[arg(long)]
    threads: Option<usize>,

    /// Allow overwriting existing output files (default: refuse). Applies to
    /// the coefficient NIfTI, sidecar JSON, and every `--diagnostics` sibling.
    #[arg(long)]
    overwrite: bool,

    /// Suppress the periodic progress heartbeat and per-step summary lines.
    /// Errors still go to stderr.
    #[arg(long)]
    quiet: bool,

    /// Seconds between heartbeat lines during the per-voxel fit. Default 30.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance captured into the sidecar JSON. `minimal` (default) keeps
    /// no PHI surface — version, git SHA, build timestamp, threads, runtime
    /// only. `full` adds argv, hostname, and wall-clock start; opt-in only
    /// when input paths and host info are safe to retain. `none` skips it.
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,

    /// Also project the fit to a Tournier-ordered ODX file at `PATH`
    /// (one-step alternative to `cs-fit … && cs-odf …`). Defaults match
    /// `cs-odf` — DSI-Studio ODF8 peak finder, brain-wide ODF normalization,
    /// anisotropic-power DPV, lmax = largest even ≤ radial_order. For
    /// custom peak settings, microstructure scalars, or alternative lmax,
    /// run `cs-odf` against the coefficient NIfTI instead — the fit's most
    /// expensive step is preserved either way.
    #[arg(long)]
    odx_output: Option<PathBuf>,

    /// Emit `--odx-output` as a directory tree instead of a `.odx` zip
    /// archive. Mirrors `cs-odf --directory`.
    #[arg(long)]
    odx_directory: bool,
}

fn main() -> Result<()> {
    let mut args = Cli::parse();

    // Pre-flight: cheap path checks before the multi-hour fit. The nifti
    // crate infers format from the extension; passing `--output prefix`
    // (no `.nii.gz`/`.nii`) used to error out at the very end of the fit.
    // We also validate the parent directory and the overwrite guard up
    // front so any path mistake surfaces in seconds.
    precheck_outputs(&mut args)?;

    let provenance_builder = ProvenanceBuilder::new("cs-fit", args.provenance);
    let (threads, source) = cs_dmri::configure_rayon_threads(args.threads)?;
    if !args.quiet {
        let n = if threads == 0 { effective_thread_count() } else { threads };
        eprintln!("[cs-fit] threads={} source={}", n, source.as_str());
    }

    let bvec_frame = if args.no_bvec_rotation {
        BvecFrame::ImageAxis
    } else {
        BvecFrame::WorldRas
    };
    let mut dwi = load_dwi(
        &args.dwi,
        &args.bval,
        &args.bvec,
        args.mask.as_deref(),
        args.big_delta,
        args.small_delta,
        Some(args.gmax),
        bvec_frame,
    )
    .with_context(|| "failed to load DWI bundle")?;

    let mask_voxels = dwi.mask.iter().filter(|&&v| v).count();
    if !args.quiet {
        eprintln!(
            "[cs-fit] DWI shape {:?}, mask voxels: {}, deltas: Δ={:.4}s δ={:.4}s ({:?})",
            dwi.shape(),
            mask_voxels,
            dwi.gtab.big_delta,
            dwi.gtab.small_delta,
            dwi.gtab.delta_source,
        );
    }
    cs_dmri::qc::report_input_qc("cs-fit", &dwi, args.quiet);

    let basis = ShoreBasis::new(args.radial_order, args.zeta);
    let design = basis.design_matrix(&dwi.gtab);
    let regularization = basis.regularization();

    let cfg = FitConfig {
        compute_diagnostics: args.diagnostics,
    };

    let interval = Duration::from_secs(args.progress_interval_secs.max(1));
    let heartbeat = Heartbeat::new("cs-fit", mask_voxels, interval, args.quiet);

    let (coefficients, diagnostics, solver_meta) = match args.reg {
        Regularization::L1 => {
            let strategy = build_alpha_strategy(&args)?;
            // The base solver's `alpha` field is overwritten per-voxel; the
            // initial value just feeds the (alpha-independent) Lipschitz cache.
            let base = FistaSolver::new(
                design.clone(),
                args.alpha.max(1e-12),
                args.max_iter,
                args.tol,
                args.non_negative,
            );
            // PathL2Anchored needs a per-voxel L2 reference. Build the
            // Tikhonov solver once (cached Cholesky) and share read-only
            // across rayon workers.
            let fit = if matches!(strategy, AlphaStrategy::PathL2Anchored { .. }) {
                let l2 = TikhonovSolver::new(
                    design.clone(),
                    &regularization,
                    args.lambda_n,
                    args.lambda_l,
                );
                fit_volume_with_alpha_strategy_l2_anchored_reporting(
                    &dwi,
                    &design,
                    &base,
                    &l2,
                    &strategy,
                    basis.n_coeffs(),
                    cfg,
                    || heartbeat.tick(),
                )
            } else {
                fit_volume_with_alpha_strategy_reporting(
                    &dwi,
                    &design,
                    &base,
                    &strategy,
                    basis.n_coeffs(),
                    cfg,
                    || heartbeat.tick(),
                )
            };
            let chosen = match (&strategy, fit.alpha_distribution) {
                (AlphaStrategy::Fixed { alpha }, _) => ChosenAlpha::Global { alpha: *alpha },
                (_, Some((median, p10, p90))) => ChosenAlpha::PerVoxel { median, p10, p90 },
                (_, None) => ChosenAlpha::Global { alpha: fit.representative_alpha },
            };
            if !args.quiet {
                if let Some((m, p10, p90)) = fit.alpha_distribution {
                    eprintln!(
                        "[cs-fit] α distribution (per-voxel): p10={:.3e}, median={:.3e}, p90={:.3e}",
                        p10, m, p90
                    );
                } else {
                    eprintln!("[cs-fit] α (global): {:.3e}", fit.representative_alpha);
                }
            }
            let meta = SolverMetadata::Fista {
                alpha_strategy: strategy,
                chosen_alpha: chosen,
                non_negative: args.non_negative,
                max_iter: args.max_iter,
                tol: args.tol,
            };
            (fit.result.coefficients, fit.result.diagnostics, meta)
        }
        Regularization::L2 => {
            let solver =
                TikhonovSolver::new(design.clone(), &regularization, args.lambda_n, args.lambda_l);
            let r = fit_volume_reporting(
                &dwi,
                &design,
                &solver,
                basis.n_coeffs(),
                cfg,
                || heartbeat.tick(),
            );
            let meta = SolverMetadata::Tikhonov {
                lambda_n: args.lambda_n,
                lambda_l: args.lambda_l,
            };
            (r.coefficients, r.diagnostics, meta)
        }
        Regularization::AmpNn => {
            let lmax = cs_dmri::odf::default_lmax(args.radial_order);
            let icls_cfg = cs_dmri::solver::IclsConfig {
                max_iter: args.amp_nn_max_iter,
                tol: args.amp_nn_tol,
                epsilon: args.amp_nn_epsilon,
            };
            let solver = cs_dmri::solver::ShoreIclsSolver::new(
                design.clone(),
                &basis,
                lmax,
                icls_cfg,
            );
            let r = fit_volume_reporting(
                &dwi,
                &design,
                &solver,
                basis.n_coeffs(),
                cfg,
                || heartbeat.tick(),
            );
            let meta = SolverMetadata::ShoreIcls {
                lmax,
                n_constraint_dirs: solver.n_constraint_dirs(),
                max_iter: args.amp_nn_max_iter,
                tol: args.amp_nn_tol,
                epsilon: args.amp_nn_epsilon,
            };
            (r.coefficients, r.diagnostics, meta)
        }
    };

    heartbeat.finish();

    // The fit is done: from here on only the mask and gradient metadata are
    // used, so release the signal before writing outputs and building the ODX.
    dwi.data = ndarray::Array4::zeros((0, 0, 0, 0));

    let threads_used = effective_thread_count();
    let provenance = provenance_builder.map(|b| b.finish(threads_used));

    let metadata = SidecarMetadata {
        basis: basis.metadata(),
        big_delta_seconds: dwi.gtab.big_delta,
        small_delta_seconds: dwi.gtab.small_delta,
        tau_seconds: dwi.gtab.tau(),
        delta_source: dwi.gtab.delta_source,
        gmax_tesla_per_meter: Some(args.gmax),
        solver: solver_meta,
        n_coefficients: basis.n_coeffs(),
        bvec_frame: dwi.bvec_frame,
        provenance,
    };

    let coeffs_file = CoefficientsFile {
        coeffs: coefficients,
        metadata,
    };
    coeffs_file
        .write(&args.output, &args.dwi, args.overwrite)
        .with_context(|| format!("failed to write coefficients to {:?}", args.output))?;

    if !args.quiet {
        eprintln!("[cs-fit] wrote coefficients: {}", args.output.display());
    }

    // Diagnostic sibling NIfTIs (`_r2.nii.gz`, etc.). Held in `diag_dpvs` so
    // the same Array3s can be packaged into the ODX below without rebuilding
    // them from disk.
    let mut diag_dpvs: Vec<(&'static str, ndarray::Array3<f32>)> = Vec::new();
    if let Some(diag) = diagnostics {
        if !args.quiet {
            eprintln!(
                "[cs-fit] mean R² (masked) = {:.4}",
                mean_in_mask(&diag.r2, &dwi.mask)
            );
        }

        let r2_path = sibling_path(&args.output, "_r2.nii.gz");
        write_3d_f32(&r2_path, &args.dwi, &diag.r2, args.overwrite)?;
        if !args.quiet {
            eprintln!("[cs-fit] wrote R² map: {}", r2_path.display());
        }
        diag_dpvs.push(("r2", diag.r2));

        let rmse = rmse_from_residual_l2(&diag.residual_l2, dwi.gtab.n_grads());
        let rmse_path = sibling_path(&args.output, "_rmse.nii.gz");
        write_3d_f32(&rmse_path, &args.dwi, &rmse, args.overwrite)?;
        if !args.quiet {
            eprintln!("[cs-fit] wrote RMSE map: {}", rmse_path.display());
        }
        diag_dpvs.push(("rmse", rmse));

        if let Some(alpha_map) = diag.alpha {
            let alpha_path = sibling_path(&args.output, "_alpha.nii.gz");
            write_3d_f32(&alpha_path, &args.dwi, &alpha_map, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-fit] wrote α map: {}", alpha_path.display());
            }
            diag_dpvs.push(("alpha", alpha_map));
        }
        if let Some(bic_map) = diag.bic {
            let bic_path = sibling_path(&args.output, "_bic.nii.gz");
            write_3d_f32(&bic_path, &args.dwi, &bic_map, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-fit] wrote BIC map: {}", bic_path.display());
            }
            diag_dpvs.push(("bic", bic_map));
        }
        if let Some(rss_l2_map) = diag.rss_l2 {
            let rss_l2_path = sibling_path(&args.output, "_rss_l2.nii.gz");
            write_3d_f32(&rss_l2_path, &args.dwi, &rss_l2_map, args.overwrite)?;
            if !args.quiet {
                eprintln!("[cs-fit] wrote L2 reference RSS map: {}",
                          rss_l2_path.display());
            }
            // rss_l2 isn't a DPV cs-odf normally exposes; skip embedding.
            let _ = rss_l2_map;
        }

        let sparsity = sparsity_map(&coeffs_file.coeffs, &dwi.mask);
        let sparsity_path = sibling_path(&args.output, "_sparsity.nii.gz");
        write_3d_f32(&sparsity_path, &args.dwi, &sparsity, args.overwrite)?;
        if !args.quiet {
            eprintln!(
                "[cs-fit] wrote sparsity map: {} (mean nnz/n_coeffs in mask = {:.3})",
                sparsity_path.display(),
                mean_in_mask(&sparsity, &dwi.mask),
            );
        }
        diag_dpvs.push(("sparsity", sparsity));
    }

    if let Some(odx_path) = args.odx_output.as_deref() {
        write_odx_from_fit(
            odx_path,
            args.odx_directory,
            &args.dwi,
            &coeffs_file.coeffs,
            &dwi.mask,
            &basis,
            &diag_dpvs,
            args.overwrite,
            args.quiet,
            args.progress_interval_secs,
        )?;
        if !args.quiet {
            eprintln!("[cs-fit] wrote ODX: {}", odx_path.display());
        }
    }

    Ok(())
}

/// Project the in-memory fit to a Tournier ODX file using the same defaults
/// `cs-odf` exposes (DSI-Studio ODF8 peaks, brain-wide ODF normalization,
/// AP DPV, lmax = largest even ≤ radial_order, plus microstructure DPVs
/// winsorized at p99.5). Power-user re-exports go through `cs-odf` against
/// the on-disk coefficient NIfTI.
fn write_odx_from_fit(
    odx_path: &std::path::Path,
    directory: bool,
    dwi_ref: &std::path::Path,
    raw_coeffs: &ndarray::Array4<f32>,
    raw_mask: &ndarray::Array3<bool>,
    basis: &cs_dmri::ShoreBasis,
    raw_diag_dpvs: &[(&'static str, ndarray::Array3<f32>)],
    overwrite: bool,
    quiet: bool,
    progress_interval_secs: u64,
) -> Result<()> {
    use cs_dmri::io::microstructure::{
        MicrostructureOptions, compute_microstructure, embed_microstructure_dpvs,
    };
    use cs_dmri::io::odx_out::{
        ShoreOdxOptions, ShoreOdxPeakOpts, build_shore_odx, canonicalize_array3_f32,
        canonicalize_array4, canonicalize_bool_mask, finalize_and_write_odx,
    };
    use odx_rs::CanonTransform;
    use odx_rs::reference_affine::read_reference_affine;

    let raw_affine = read_reference_affine(dwi_ref)
        .map_err(|e| anyhow::anyhow!("read affine from {:?}: {e}", dwi_ref))?;
    let canon = CanonTransform::from_affine(raw_affine);
    let (coeffs, affine) = canonicalize_array4(raw_coeffs, raw_affine, &canon)?;
    let mask = canonicalize_bool_mask(raw_mask, raw_affine, &canon)?;

    let canon_dpvs: Vec<(&'static str, ndarray::Array3<f32>)> = raw_diag_dpvs
        .iter()
        .map(|(name, arr)| Ok((*name, canonicalize_array3_f32(arr, raw_affine, &canon)?)))
        .collect::<Result<_>>()?;
    let dpv_refs: Vec<(&str, &ndarray::Array3<f32>)> =
        canon_dpvs.iter().map(|(n, a)| (*n, a)).collect();

    let lmax = cs_dmri::odf::default_lmax(basis.radial_order);
    let opts = ShoreOdxOptions {
        peaks: Some(ShoreOdxPeakOpts::default()),
        progress_interval_secs,
        quiet,
        ..ShoreOdxOptions::default()
    };
    let mut built = build_shore_odx(&coeffs, affine, &mask, basis, lmax, &dpv_refs, &opts)?;

    let mopts = MicrostructureOptions { quiet, ..MicrostructureOptions::default() };
    let scalars = compute_microstructure(
        basis,
        &coeffs,
        &built.masked_indices,
        built.peak0_dirs.as_deref(),
        &mopts,
    );
    embed_microstructure_dpvs(&mut built.builder, &scalars);

    finalize_and_write_odx(built.builder, odx_path, overwrite, directory)
        .with_context(|| format!("write ODX {:?}", odx_path))?;
    Ok(())
}

/// Per-voxel sparsity: fraction of strictly-nonzero SH coefficients in the
/// fitted vector. FISTA's soft-threshold returns exact zeros, so no tolerance
/// is needed; for L2 this is essentially 1.0 everywhere (sanity check).
fn sparsity_map(
    coeffs: &ndarray::Array4<f32>,
    mask: &ndarray::Array3<bool>,
) -> ndarray::Array3<f32> {
    let s = coeffs.shape();
    let (nx, ny, nz, nk) = (s[0], s[1], s[2], s[3]);
    let denom = nk as f32;
    let mut out = ndarray::Array3::<f32>::zeros((nx, ny, nz));
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if !mask[(x, y, z)] {
                    continue;
                }
                let mut nnz = 0u32;
                for k in 0..nk {
                    if coeffs[(x, y, z, k)] != 0.0 {
                        nnz += 1;
                    }
                }
                out[(x, y, z)] = nnz as f32 / denom;
            }
        }
    }
    out
}

/// RMSE = ‖Mc − s‖₂ / √n_grads, derived from the per-voxel residual L2 norm.
fn rmse_from_residual_l2(
    residual_l2: &ndarray::Array3<f32>,
    n_grads: usize,
) -> ndarray::Array3<f32> {
    let denom = (n_grads as f32).sqrt();
    residual_l2.mapv(|r| r / denom)
}

fn build_alpha_strategy(args: &Cli) -> Result<AlphaStrategy> {
    Ok(match args.alpha_mode {
        AlphaMode::Fixed => AlphaStrategy::Fixed { alpha: args.alpha },
        AlphaMode::AlphaRatio => {
            if !(args.alpha_ratio > 0.0 && args.alpha_ratio < 1.0) {
                bail!("--alpha-ratio must lie in (0, 1), got {}", args.alpha_ratio);
            }
            AlphaStrategy::AlphaMaxRatio { ratio: args.alpha_ratio }
        }
        AlphaMode::PathBic => {
            if args.path_n_alphas < 2 {
                bail!("--path-n-alphas must be ≥ 2, got {}", args.path_n_alphas);
            }
            // Mode-dependent default: path-bic stays at 1e-3 for backward
            // compatibility.
            let eps = args.path_eps.unwrap_or(1e-3);
            if !(eps > 0.0 && eps < 1.0) {
                bail!("--path-eps must lie in (0, 1), got {}", eps);
            }
            AlphaStrategy::PathBic {
                path: AlphaPath { n: args.path_n_alphas, eps },
            }
        }
        AlphaMode::L2Anchored => {
            if args.path_n_alphas < 2 {
                bail!("--path-n-alphas must be ≥ 2, got {}", args.path_n_alphas);
            }
            // Mode-dependent default: l2-anchored uses 1e-4 so the slack
            // constraint binds on the full path. With eps=1e-3 (path-bic
            // default) high-b regimes fall back ~17% of voxels.
            let eps = args.path_eps.unwrap_or(1e-4);
            if !(eps > 0.0 && eps < 1.0) {
                bail!("--path-eps must lie in (0, 1), got {}", eps);
            }
            if args.slack < 0.0 {
                bail!("--slack must be ≥ 0, got {}", args.slack);
            }
            AlphaStrategy::PathL2Anchored {
                path: AlphaPath { n: args.path_n_alphas, eps },
                slack: args.slack,
            }
        }
    })
}

/// Pre-flight: validate every output path cs-fit will eventually write before
/// burning an hour on the fit.
///
/// - Auto-appends `.nii.gz` to `--output` when the user passed an
///   extensionless prefix (the bug that motivated this) — emitting a stderr
///   warning so the rewrite is observable.
/// - Refuses to clobber any of the eventual outputs when `--overwrite` is
///   off, with the same wording `atomic_write` uses post-fit.
/// - Probes the parent directory for write permissions so a read-only mount
///   or typo'd path fails immediately, not after the fit completes.
///
/// Conditional sibling paths (`_alpha`, `_bic`, `_rss_l2`) are gated on the
/// same alpha-mode predicates the emit code uses below, so we don't reject
/// pre-existing files that wouldn't have been touched anyway.
fn precheck_outputs(args: &mut Cli) -> Result<()> {
    ensure_nifti_extension(&mut args.output, "--output");

    precheck_writable(&args.output, args.overwrite)
        .with_context(|| format!("output {} not writable", args.output.display()))?;

    // Sidecar shares the parent dir — same writability check, separate
    // overwrite check (the user might have stale `.json` from a prior run).
    let sidecar = sibling_path(&args.output, ".json");
    precheck_writable(&sidecar, args.overwrite)
        .with_context(|| format!("sidecar {} not writable", sidecar.display()))?;

    if args.diagnostics {
        let mut diag_paths: Vec<PathBuf> = Vec::new();
        diag_paths.push(sibling_path(&args.output, "_r2.nii.gz"));
        diag_paths.push(sibling_path(&args.output, "_rmse.nii.gz"));
        diag_paths.push(sibling_path(&args.output, "_sparsity.nii.gz"));

        let per_voxel_alpha = !matches!(args.alpha_mode, AlphaMode::Fixed);
        if matches!(args.reg, Regularization::L1) && per_voxel_alpha {
            diag_paths.push(sibling_path(&args.output, "_alpha.nii.gz"));
        }
        if matches!(args.reg, Regularization::L1)
            && matches!(args.alpha_mode, AlphaMode::PathBic)
        {
            diag_paths.push(sibling_path(&args.output, "_bic.nii.gz"));
        }
        if matches!(args.reg, Regularization::L1)
            && matches!(args.alpha_mode, AlphaMode::L2Anchored)
        {
            diag_paths.push(sibling_path(&args.output, "_rss_l2.nii.gz"));
        }

        for p in &diag_paths {
            precheck_writable(p, args.overwrite)
                .with_context(|| format!("diagnostic {} not writable", p.display()))?;
        }
    }

    if let Some(odx) = args.odx_output.as_deref() {
        precheck_writable(odx, args.overwrite)
            .with_context(|| format!("--odx-output {} not writable", odx.display()))?;
    }

    Ok(())
}

fn mean_in_mask(arr: &ndarray::Array3<f32>, mask: &ndarray::Array3<bool>) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for ((idx, &v), &m) in arr.indexed_iter().zip(mask.iter()) {
        let _ = idx;
        if m {
            sum += v as f64;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

