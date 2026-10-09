// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-fit`: load DWI + bval/bvec/mask → write coefficient NIfTI + JSON sidecar.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use cs_dmri::basis::Basis;
use cs_dmri::fit::{
    ShoreFitSpec, ShoreRegularization, build_alpha_strategy, fit_shore, mean_in_mask,
    rmse_from_residual_l2, sparsity_map,
};
use cs_dmri::io::aux::{ensure_nifti_extension, precheck_writable, sibling_path, write_3d_f32};
use cs_dmri::io::coeffs::{CoefficientsFile, SidecarMetadata};
use cs_dmri::io::dwi::load_dwi;
use cs_dmri::qspace::{BvecFrame, TORTOISE_DEFAULT_GMAX};
use cs_dmri::{Heartbeat, ProvenanceBuilder, ProvenanceMode, effective_thread_count};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Regularization {
    /// L1-regularised (sparse) fit solved with FISTA. Default.
    L1,
    /// L2 (Tikhonov) fit with a closed-form solution.
    L2,
    /// Least-squares fit subject to non-negativity of the ODF amplitudes on
    /// a dense sphere, solved with the Goldfarb-Idnani inequality-constrained
    /// least-squares method.
    AmpNn,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AlphaMode {
    /// The same α for every voxel (--alpha).
    Fixed,
    /// α = ratio · α_max for each voxel, where α_max is the smallest α giving
    /// an all-zero solution (--alpha-ratio).
    AlphaRatio,
    /// Along a logarithmic path from α_max to α_max·eps, the α minimising the
    /// Bayesian information criterion.
    PathBic,
    /// Along the same path, the largest α whose residual sum of squares is
    /// within (1 + slack) of a Tikhonov (L2) fit's, using --lambda-n and
    /// --lambda-l. If no α satisfies the bound, the α with the smallest
    /// residual sum of squares is used. Default.
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
    /// Brain mask NIfTI. If omitted, a mask is computed from the mean b=0 image.
    #[arg(long)]
    mask: Option<PathBuf>,
    /// Diffusion time Δ (big delta), in seconds. If either Δ or δ is omitted,
    /// both are estimated from the maximum b-value and --gmax.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Gradient pulse duration δ (small delta), in seconds.
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude, in T/m. Used only when Δ and δ are estimated.
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,
    /// Output coefficient NIfTI path.
    #[arg(long)]
    output: PathBuf,

    /// SHORE radial order. Even values are expected.
    #[arg(long, default_value_t = 6)]
    radial_order: u32,
    /// SHORE scale parameter ζ.
    #[arg(long, default_value_t = 700.0)]
    zeta: f64,

    /// Regularisation of the coefficient fit.
    #[arg(long, value_enum, default_value_t = Regularization::L1)]
    reg: Regularization,

    /// Selection of the L1 sparsity weight α, made independently for each
    /// voxel except in `fixed` mode.
    #[arg(long, value_enum, default_value_t = AlphaMode::L2Anchored)]
    alpha_mode: AlphaMode,
    /// L1 sparsity weight α. Used only with `--alpha-mode fixed`.
    #[arg(long, default_value_t = 1.0)]
    alpha: f64,
    /// Ratio α / α_max, in (0, 1). Used only with `--alpha-mode alpha-ratio`.
    #[arg(long, default_value_t = 1e-3)]
    alpha_ratio: f64,
    /// Number of α values on the regularisation path (path-bic, l2-anchored).
    /// Must be at least 2.
    #[arg(long, default_value_t = 20)]
    path_n_alphas: usize,
    /// Ratio α_min / α_max of the regularisation path (path-bic,
    /// l2-anchored), in (0, 1). Default: 1e-3 for path-bic and 1e-4 for
    /// l2-anchored.
    #[arg(long)]
    path_eps: Option<f64>,
    /// Residual tolerance for `--alpha-mode l2-anchored`: the selected α
    /// satisfies RSS ≤ (1 + slack) · RSS_L2. Smaller values give fits closer
    /// to the L2 fit; larger values give sparser fits. Must be ≥ 0.
    #[arg(long, default_value_t = 0.05)]
    slack: f64,

    /// Maximum number of FISTA iterations per L1 fit, including each fit on
    /// the regularisation path.
    #[arg(long, default_value_t = 1000)]
    max_iter: u32,
    /// L1 convergence tolerance on the relative change in coefficients.
    #[arg(long, default_value_t = 1e-6)]
    tol: f64,
    /// Enforce non-negative coefficients during the L1 fit.
    #[arg(long)]
    non_negative: bool,

    /// L2 radial regularisation weight λ_N.
    #[arg(long, default_value_t = 1e-8)]
    lambda_n: f64,
    /// L2 angular regularisation weight λ_L.
    #[arg(long, default_value_t = 1e-8)]
    lambda_l: f64,

    /// Maximum number of active-set iterations for `--reg amp-nn`.
    #[arg(long, default_value_t = 200)]
    amp_nn_max_iter: usize,
    /// Constraint tolerance for `--reg amp-nn`: an ODF amplitude is treated
    /// as non-negative if it is ≥ -tol.
    #[arg(long, default_value_t = 1e-9)]
    amp_nn_tol: f64,
    /// Tikhonov term added to the diagonal of HᵀH for `--reg amp-nn`. Larger
    /// values are needed if the Cholesky factorisation fails on a
    /// rank-deficient design.
    #[arg(long, default_value_t = 1e-10)]
    amp_nn_epsilon: f64,

    /// Also write per-voxel maps of R², residual, iteration count and
    /// regularisation type, and, for L1 fits with per-voxel α selection, the
    /// selected α, next to the coefficient NIfTI.
    #[arg(long)]
    diagnostics: bool,

    /// Fit with b-vectors in the image-axis (FSL) frame. By default b-vectors
    /// are rotated into world (RAS) coordinates before fitting.
    #[arg(long)]
    no_bvec_rotation: bool,

    /// Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used,
    /// then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU.
    #[arg(long)]
    threads: Option<usize>,

    /// Overwrite existing output files. Without this flag, existing outputs
    /// cause an error. Applies to the coefficient NIfTI, the sidecar JSON and
    /// the `--diagnostics` maps.
    #[arg(long)]
    overwrite: bool,

    /// Suppress periodic progress and per-step summary messages. Errors are
    /// still written to stderr.
    #[arg(long)]
    quiet: bool,

    /// Interval between progress messages during the voxel-wise fit, in
    /// seconds.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,

    /// Provenance recorded in the sidecar JSON.
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,

    /// Also write an ODX file of ODF SH coefficients (MRtrix3/Tournier
    /// convention) to this path, using the `cs-odf` defaults: DSI Studio
    /// ODF8 peak finding, brain-wide ODF normalisation, an anisotropic power
    /// map, and lmax equal to the largest even integer
    /// ≤ --radial-order. For other settings, run `cs-odf` on the coefficient
    /// NIfTI.
    #[arg(long)]
    odx_output: Option<PathBuf>,

    /// Write `--odx-output` as a directory instead of a `.odx` zip archive,
    /// as with `cs-odf --directory`.
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

    let regularization = match args.reg {
        Regularization::L1 => ShoreRegularization::L1 {
            strategy: build_alpha_strategy(
                args.alpha_mode.into(),
                args.alpha,
                args.alpha_ratio,
                args.path_n_alphas,
                args.path_eps,
                args.slack,
            )?,
            max_iter: args.max_iter,
            tol: args.tol,
            non_negative: args.non_negative,
            seed_alpha: args.alpha,
        },
        Regularization::L2 => ShoreRegularization::L2,
        Regularization::AmpNn => ShoreRegularization::AmpNonNeg {
            icls: cs_dmri::solver::IclsConfig {
                max_iter: args.amp_nn_max_iter,
                tol: args.amp_nn_tol,
                epsilon: args.amp_nn_epsilon,
            },
        },
    };
    let spec = ShoreFitSpec {
        radial_order: args.radial_order,
        zeta: args.zeta,
        regularization,
        lambda_n: args.lambda_n,
        lambda_l: args.lambda_l,
        compute_diagnostics: args.diagnostics,
    };

    let interval = Duration::from_secs(args.progress_interval_secs.max(1));
    let heartbeat = Heartbeat::new("cs-fit", mask_voxels, interval, args.quiet);
    let fit = fit_shore(&dwi, &spec, || heartbeat.tick())?;
    if !args.quiet && matches!(args.reg, Regularization::L1) {
        match (fit.alpha_distribution, &fit.solver) {
            (Some((m, p10, p90)), _) => eprintln!(
                "[cs-fit] α distribution (per-voxel): p10={:.3e}, median={:.3e}, p90={:.3e}",
                p10, m, p90
            ),
            (
                None,
                cs_dmri::io::coeffs::SolverMetadata::Fista {
                    chosen_alpha: cs_dmri::io::coeffs::ChosenAlpha::Global { alpha },
                    ..
                },
            ) => eprintln!("[cs-fit] α (global): {:.3e}", alpha),
            _ => {}
        }
    }
    let basis = fit.basis;
    let (coefficients, diagnostics, solver_meta) =
        (fit.result.coefficients, fit.result.diagnostics, fit.solver);

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
    use cs_dmri::io::microstructure::MicrostructureOptions;
    use cs_dmri::io::odx_out::{
        ShoreOdxOptions, ShoreOdxPeakOpts, ShoreToOdxOptions, finalize_and_write_odx,
        shore_coeffs_to_odx,
    };
    use odx_rs::reference_affine::read_reference_affine;

    let raw_affine = read_reference_affine(dwi_ref)
        .map_err(|e| anyhow::anyhow!("read affine from {:?}: {e}", dwi_ref))?;
    let dpv_refs: Vec<(&str, &ndarray::Array3<f32>)> =
        raw_diag_dpvs.iter().map(|(n, a)| (*n, a)).collect();
    let opts = ShoreToOdxOptions {
        lmax: None,
        odx: ShoreOdxOptions {
            peaks: Some(ShoreOdxPeakOpts::default()),
            progress_interval_secs,
            quiet,
            ..ShoreOdxOptions::default()
        },
        microstructure: Some(MicrostructureOptions { quiet, ..MicrostructureOptions::default() }),
    };
    let out = shore_coeffs_to_odx(raw_coeffs, raw_affine, Some(raw_mask), basis, &dpv_refs, &opts)?;
    finalize_and_write_odx(out.build.builder, odx_path, overwrite, directory)
        .with_context(|| format!("write ODX {:?}", odx_path))?;
    Ok(())
}

impl From<AlphaMode> for cs_dmri::fit::AlphaMode {
    fn from(m: AlphaMode) -> Self {
        match m {
            AlphaMode::Fixed => Self::Fixed,
            AlphaMode::AlphaRatio => Self::AlphaRatio,
            AlphaMode::PathBic => Self::PathBic,
            AlphaMode::L2Anchored => Self::L2Anchored,
        }
    }
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
