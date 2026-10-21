// SPDX-License-Identifier: MIT OR Apache-2.0
//! `consh`: continuous-b spherical harmonic deconvolution (ConSH).
//!
//! Whole-brain continuous-b multi-tissue CSD.
//!
//! Reconstructs a white-matter fODF and gray-matter and CSF compartments from
//! non-shelled q-space. Unlike [`cs-ss3t-full`](cs-ss3t-full.rs) (single-shell,
//! alternating two-tissue), it uses every measurement at its own b-value: a
//! dense-b response, estimated continuously in b from single-tissue voxels,
//! drives one joint multi-tissue constrained least-squares fit per voxel. On a
//! shelled acquisition it reduces exactly to standard MSMT-CSD.
//!
//! Stages: DTI (RESTORE) → dense-b response estimation → per-voxel ConSH fit →
//! mtnormalise. Outputs the WM fODF (SH) and GM/CSF NIfTIs, or one ODX.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use std::sync::RwLock;

use cs_dmri::dti::RestoreConfig;
use cs_dmri::graddev::VoxelGradDev;
use cs_dmri::io::output::{sibling_path, write_3d_u32_as_f32, write_3d_u8, write_4d_f32};
use cs_dmri::io::dwi::{DwiData, load_dwi, load_dwi_mrtrix_grad, load_mask};
use cs_dmri::io::odx_out::{
    ContinuousBQuantMeta, GeneratedBy, ReferenceB0Source, write_continuous_b_odx,
};
use cs_dmri::multitissue::consh::{ConshConfig, ConshResponses};
use cs_dmri::multitissue::consh_pipeline::{
    ConshPipelineConfig, ConshPipelineResult, ConshResponseSource, REFERENCE_WM_FRACTION,
    consh_pipeline,
};
use cs_dmri::multitissue::pipeline::{PipelineObserver, PipelineStage};
use cs_dmri::multitissue::mtnormalise::{MtnormaliseConfig, target_sum_mrtrix_default};
use cs_dmri::multitissue::response::TissueResponse;
use cs_dmri::multitissue::response_estimation::{
    ConshResponseConfig, DhollanderConfig, DhollanderSelectConfig, ResponseEstimator,
    write_response_txt,
};
use cs_dmri::multitissue::response_smooth::SmoothResponseConfig;
use cs_dmri::qspace::{BvecFrame, TORTOISE_DEFAULT_GMAX};
use cs_dmri::solver::icls::IclsConfig;
use cs_dmri::{
    Heartbeat, ProvenanceBuilder, ProvenanceMode, configure_rayon_threads, effective_thread_count,
};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Continuous-b multi-tissue CSD (ConSH) for non-shelled diffusion MRI"
)]
struct Cli {
    /// 4D DWI NIfTI input (non-shelled CS-DSI or any multi-b scheme).
    #[arg(long)]
    dwi: PathBuf,
    /// FSL bval file.
    #[arg(long, required_unless_present = "grad", requires = "bvec")]
    bval: Option<PathBuf>,
    /// FSL bvec file. Components along the image's voxel axes (the dipy/qsiprep
    /// convention).
    #[arg(long, required_unless_present = "grad", requires = "bval")]
    bvec: Option<PathBuf>,
    /// MRtrix gradient table (`x y z b` per row, direction in world RAS), in
    /// place of --bval/--bvec. Unambiguous about the frame, which matters for
    /// --graddev.
    #[arg(long, conflicts_with_all = ["bval", "bvec"])]
    grad: Option<PathBuf>,

    /// Gradient-nonlinearity correction at fit time: a 9-volume deviation field
    /// on the DWI's exact grid (qsiprep `*_space-ACPC_graddev.nii.gz`, or an
    /// HCP/FSL `grad_dev.nii.gz`). Row-major `T` per voxel with `g_eff = Tᵀ g`
    /// in the image's voxel axes. Every voxel is deconvolved with its own
    /// effective gradient directions and b-values — correcting the direction-
    /// dependent b and apparent-shape distortion that rotating a fitted fODF
    /// afterwards (`odx graddev`) cannot. Implies --b-interp; the dense-b grid
    /// is extended to the largest effective b. Slower: each deviated voxel
    /// builds its own forward operator and factorization.
    #[arg(long, value_name = "NII")]
    graddev: Option<PathBuf>,
    /// Whether the --graddev file stores `T` (qsiprep/TORTOISE) or `T − I`
    /// (HCP/FSL). `auto` decides from the median diagonal.
    #[arg(long, value_enum, default_value_t = GraddevIdentity::Auto, requires = "graddev")]
    graddev_identity: GraddevIdentity,
    /// How dense-b response rows are estimated from the pooled single-tissue
    /// samples. `kernel`: each row on its own from a Gaussian-in-b window
    /// (`--sigma-b`); matches MRtrix's per-shell responses on shelled data,
    /// but is step-like there and noisy between the b-values of sparse
    /// non-shelled schemes. `smooth`: all rows jointly, linear in b between
    /// rows and smooth across rows over about `--sigma-b`, constrained (spherical
    /// mean ≥ 0 and non-increasing in b; alternating signs of the WM zonal
    /// coefficients). `dki`: the parametric response of Morez et al. (2021),
    /// axially symmetric DKI plus an offset per tissue (7 parameters for WM, 4
    /// for GM and CSF), fit to all samples at once and non-increasing in b;
    /// never chosen by `auto`. `auto`: with --graddev, `smooth` if the field
    /// moves b by more than --sigma-b (95th percentile over the mask), else
    /// `kernel`; without --graddev, `smooth` with --b-interp, else `kernel`.
    #[arg(long, value_enum, default_value_t = EstimatorChoice::Auto)]
    response_estimator: EstimatorChoice,
    /// Interpolate the response linearly in b between dense-b grid rows instead
    /// of snapping each measurement to the nearest row (±step/2). Identical on
    /// b-values that sit on grid points; on by default with --graddev.
    #[arg(long)]
    b_interp: bool,
    /// Brain mask NIfTI. If omitted, a mask is computed from the mean b=0
    /// image.
    #[arg(long)]
    mask: Option<PathBuf>,

    /// Output WM fODF NIfTI (SH coefficients, MRtrix basis). Required unless
    /// `--odx` is given.
    #[arg(long)]
    output_wm: Option<PathBuf>,
    /// Output GM compartment NIfTI. Required unless `--odx` is given.
    #[arg(long)]
    output_gm: Option<PathBuf>,
    /// Output CSF compartment NIfTI. Required unless `--odx` is given.
    #[arg(long)]
    output_csf: Option<PathBuf>,

    /// Write a single ODX file containing the WM SH coefficients, the GM and
    /// CSF compartments, the mask and WM peaks, in place of the three NIfTI
    /// outputs. A path with a `.odx` extension is written as a zip archive; any
    /// other path is written as a directory.
    #[arg(long, value_name = "PATH")]
    odx: Option<PathBuf>,

    /// Write the estimated dense-b responses (one row per grid point) to this
    /// directory, as `{wm,gm,csf}_response_denseb.txt`.
    #[arg(long)]
    write_responses_to: Option<PathBuf>,

    /// Read dense-b responses from this directory (the three
    /// `{wm,gm,csf}_response_denseb.txt` files `--write-responses-to` writes,
    /// or a group average from `consh-response mean`) instead of estimating
    /// them. The tensor fit and tissue selection are skipped, and every scan
    /// fit with the same files is deconvolved with the same responses. The
    /// response rows must match this series' dense-b grid (same maximum b and
    /// step).
    #[arg(long, conflicts_with = "write_responses_to")]
    read_responses_from: Option<PathBuf>,

    /// Skip the final multi-tissue intensity normalization (mtnormalise).
    #[arg(long)]
    no_normalize: bool,

    /// Fit without the constant noise-floor compartment (fitted by default).
    ///
    /// The magnitude noise floor is additive in absolute signal units, while
    /// the multi-tissue model is multiplicative with S0-normalized responses.
    /// In voxels whose S0 is far below that of the response-estimation voxels
    /// (cisternal and subarachnoid CSF, non-brain tissue inside the mask) the
    /// three tissue responses cannot reproduce the relative floor, and because
    /// WM is the only compartment that keeps signal at high b, the
    /// non-negative fit assigns the excess to the WM fODF. A constant
    /// compartment absorbs it at the cost of one degree of freedom and needs no
    /// noise estimate. Its per-voxel amplitude is written to the ODX as the
    /// `noise_floor` per-voxel value (on the GM/CSF scale; divide by 0.2821 for
    /// a fraction of S0), and with NIfTI outputs as a `_noisefloor.nii.gz`
    /// file next to the WM output.
    #[arg(long)]
    no_noise_floor: bool,

    // ---- continuous-b knobs ----
    /// Dense b-grid step in s/mm². Must be fine enough to resolve the b-values
    /// of the acquisition.
    #[arg(long, default_value_t = 50.0)]
    step: f64,
    /// Width (standard deviation, s/mm²) of the Gaussian window in b used by
    /// the `kernel` response estimator, and the smoothing length of `smooth`.
    #[arg(long, default_value_t = 150.0)]
    sigma_b: f64,
    /// Maximum SH order of the white matter fODF and response.
    #[arg(long, default_value_t = 8)]
    lmax_wm: usize,
    /// Maximum active-set iterations of the constrained least-squares solver.
    /// The WM non-negativity constraint is imposed at about 700 directions, so
    /// this is higher than the SS3T default (200).
    #[arg(long, default_value_t = 2000)]
    icls_max_iter: usize,
    /// Tikhonov (L2) penalty weight on the WM fODF coefficients with l ≥ 2; the
    /// l=0 term and the GM/CSF amplitudes are not penalized. 0 (default) leaves
    /// only the non-negativity constraint. A positive value damps the
    /// ill-conditioned high-order terms that make peaks in crossing regions
    /// unstable; values of 0.1–5 are typical.
    #[arg(long, default_value_t = 0.0)]
    wm_ridge: f64,
    /// Choose the WM penalty weight per voxel: the largest weight on an internal
    /// path (0 to 8) whose residual stays within this fraction of the
    /// unpenalized fit's. 0 (default) is off. Damps ambiguous crossings while
    /// leaving well-determined single-fiber voxels sharp; values of 0.02–0.10
    /// are typical. Takes precedence over --wm-ridge.
    #[arg(long, default_value_t = 0.0)]
    wm_ridge_anchor: f64,
    /// Scale the WM penalty by each l ≥ 2 coefficient's own diagonal of HᵀH, so
    /// that one `--wm-ridge` or `--wm-ridge-anchor` setting means the same across
    /// acquisitions (shelled or Cartesian, any maximum b). With
    /// `--wm-ridge-anchor` the internal path becomes relative fractions (0 to
    /// 1.6).
    #[arg(long)]
    wm_ridge_relative: bool,
    /// Drop WM peaks whose fODF amplitude is below this value. Removes shallow
    /// spurious lobes, which become more frequent under strong penalties. 0
    /// (default) keeps all peaks; values of 0.05–0.15 on the normalized fODF
    /// scale are typical.
    ///
    /// The threshold is tied to the absolute fODF scale, which changes with
    /// `--quantitative` and between acquisitions; `--peak-min-amplitude-frac`
    /// is preferable for comparisons across subjects.
    #[arg(long, default_value_t = 0.0)]
    peak_min_amplitude: f32,
    /// Scale-invariant form of `--peak-min-amplitude`: the threshold is this
    /// fraction of the volume's median dominant-peak amplitude (the median,
    /// over masked voxels with at least one peak, of each voxel's largest peak).
    ///
    /// The reference and the peaks scale together under any global change of
    /// fODF units, so one value means the same with or without
    /// `--quantitative` and across acquisitions. 0 (default) is off; values of
    /// 0.10–0.25 are typical.
    #[arg(long, default_value_t = 0.0, conflicts_with = "peak_min_amplitude")]
    peak_min_amplitude_frac: f32,

    /// White-matter mask defining the reference b=0 (`DWI_ref`) recorded in the
    /// ODX sidecar. Direct analogue of `dwinormalise individual`, which takes a
    /// mask and normalizes its median b=0.
    ///
    /// If omitted, the reference is taken from the Dhollander single-fiber WM
    /// voxels when responses are estimated, or from voxels where the fitted WM
    /// compartment dominates when they are read from disk.
    #[arg(long, value_name = "NIFTI")]
    reference_mask: Option<PathBuf>,

    /// Write fODFs in absolute signal units, multiplying each voxel's fitted
    /// coefficients by its b=0 signal, instead of the default per-voxel S/S0
    /// scale. The WM fODF amplitude is then proportional to the signal (and so
    /// to intra-cellular volume at high b), which apparent fiber density and
    /// SIFT2 require and per-voxel S/S0 normalization removes (Smith et al.
    /// 2022; Dhollander, Clemente et al. 2021).
    ///
    /// The responses are scaled by one per-scan reference (the median b=0 of
    /// the single-fiber WM voxels), which keeps the relative amplitudes of the
    /// tissues as MRtrix's `amp2response` does. A common scale across subjects
    /// comes from a shared group-average response (`consh-response mean`, read
    /// with --read-responses-from), as in MRtrix; the remaining per-subject
    /// scale is removed by the final normalization, so this option cannot be
    /// combined with --no-normalize.
    ///
    /// Scaling by S0 also suppresses the large spurious fODFs of low-S0 voxels,
    /// which come from dividing the noise floor by S0. It corrects only their
    /// magnitude; the noise-floor compartment corrects their orientation
    /// content.
    #[arg(long, conflicts_with = "no_normalize")]
    quantitative: bool,

    // ---- response-estimation (Dhollander tissue selection) knobs ----
    /// Erosion passes applied to the brain mask before tissue selection.
    #[arg(long, default_value_t = 3)]
    dh_erode: usize,
    /// FA threshold for the crude WM vs GM-CSF split.
    #[arg(long, default_value_t = 0.2)]
    dh_fa: f64,
    /// Final single-fiber WM voxels, as a percentage of refined WM.
    #[arg(long, default_value_t = 0.5)]
    dh_sfwm: f64,
    /// Final GM voxels, as a percentage of refined GM.
    #[arg(long, default_value_t = 2.0)]
    dh_gm: f64,
    /// Final CSF voxels, as a percentage of refined CSF.
    #[arg(long, default_value_t = 10.0)]
    dh_csf: f64,

    /// Use the earlier threshold-based tissue selection (top-N% MD for CSF, FA
    /// and eigenvalue dominance for WM) instead of MRtrix's staged
    /// signal-decay-metric algorithm. For reproducing earlier results only: its
    /// CSF class includes partial-volume voxels, which lowers the CSF response
    /// amplitude.
    #[arg(long)]
    legacy_tissue_selection: bool,
    /// FA above this counts a voxel as a WM single-fiber candidate.
    /// `--legacy-tissue-selection` only.
    #[arg(long, default_value_t = 0.7)]
    fa_wm_threshold: f64,
    /// Eigenvalue ratio gate for "single fiber" (suppresses crossings).
    /// `--legacy-tissue-selection` only.
    #[arg(long, default_value_t = 2.0)]
    fiber_dominance_ratio: f64,
    /// Top-N percent of MD voxels classified as CSF.
    /// `--legacy-tissue-selection` only.
    #[arg(long, default_value_t = 2.5)]
    md_csf_pct: f64,

    // ---- mtnormalise knobs ----
    /// Order of the polynomial bias field model in the normalization step
    /// (order 3 has 20 terms).
    #[arg(long, default_value_t = 3)]
    mtnorm_poly_order: usize,
    /// In the normalization step, use the median of the observed sums of l=0
    /// coefficients as the target instead of 1/√(4π).
    #[arg(long)]
    mtnorm_target_median: bool,
    /// Apply the per-tissue balance factors to the output, like MRtrix3
    /// `mtnormalise -balanced`. With a group-average response the factors
    /// absorb the scale offset between the tissues' responses, which is nearly
    /// constant across subjects, and bring WM tissue sums to the target. Off
    /// by default, as in MRtrix.
    #[arg(long)]
    mtnorm_balanced: bool,

    // ---- DTI / RESTORE ----
    /// Maximum number of RESTORE reweighting iterations in the tensor fit
    /// used for response estimation.
    #[arg(long, default_value_t = 50)]
    restore_max_iter: usize,
    /// RESTORE convergence tolerance on the relative change in tensor
    /// coefficients.
    #[arg(long, default_value_t = 1e-6)]
    restore_tol: f64,

    // ---- standard ----
    /// Diffusion time Δ (big delta), in seconds. Not used by the
    /// reconstruction; accepted for consistency with `cs-fit`.
    #[arg(long)]
    big_delta: Option<f64>,
    /// Gradient pulse duration δ (small delta), in seconds. Not used by the
    /// reconstruction.
    #[arg(long)]
    small_delta: Option<f64>,
    /// Maximum gradient amplitude, in T/m. Used only when Δ and δ are
    /// estimated.
    #[arg(long, default_value_t = TORTOISE_DEFAULT_GMAX)]
    gmax: f64,
    /// Also write per-voxel maps of convergence (`_converged.nii.gz`) and
    /// iteration count (`_iters.nii.gz`) next to `--output-wm`. Not written
    /// with `--odx`.
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
    /// cause an error.
    #[arg(long)]
    overwrite: bool,
    /// Suppress periodic progress and per-step summary messages.
    #[arg(long)]
    quiet: bool,
    /// Interval between progress messages, in seconds.
    #[arg(long, default_value_t = 30)]
    progress_interval_secs: u64,
    /// Provenance mode. Accepted for consistency with the other tools; `consh`
    /// writes no provenance record.
    #[arg(long, value_enum, default_value_t = ProvenanceMode::default())]
    provenance: ProvenanceMode,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum EstimatorChoice {
    Auto,
    Kernel,
    Smooth,
    Dki,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum GraddevIdentity {
    Auto,
    Included,
    Absent,
}

impl From<GraddevIdentity> for odx_rs::IdentityPolicy {
    fn from(v: GraddevIdentity) -> Self {
        match v {
            GraddevIdentity::Auto => Self::Auto,
            GraddevIdentity::Included => Self::Included,
            GraddevIdentity::Absent => Self::Absent,
        }
    }
}

struct StageProgress {
    current: RwLock<Option<Heartbeat>>,
    mask_voxels: usize,
    interval: Duration,
    quiet: bool,
    noise_floor: bool,
}

impl StageProgress {
    fn finish_current(&self) {
        if let Some(hb) = self.current.write().unwrap().take() {
            hb.finish();
        }
    }
}

impl PipelineObserver for StageProgress {
    fn stage(&self, stage: PipelineStage) {
        self.finish_current();
        let (banner, label) = match stage {
            PipelineStage::Dti => (
                Some("stage 1/3: RESTORE tensor fit and dense-b response estimation".to_string()),
                Some("dti"),
            ),
            PipelineStage::Consh => (
                Some(format!(
                    "stage 2/3: continuous-b multi-tissue fit ({} per voxel)",
                    if self.noise_floor { "3 tissues and a noise floor" } else { "3 tissues" }
                )),
                Some("consh"),
            ),
            PipelineStage::Normalize => {
                (Some("stage 3/3: mtnormalise (polynomial bias-field correction)".to_string()), None)
            }
            PipelineStage::ResponseEstimation | PipelineStage::Ss3t => (None, None),
        };
        if let (Some(b), false) = (banner, self.quiet) {
            eprintln!("[consh] {b}");
        }
        if let Some(label) = label {
            *self.current.write().unwrap() =
                Some(Heartbeat::new(label, self.mask_voxels, self.interval, self.quiet));
        }
    }

    fn voxel(&self) {
        if let Some(hb) = self.current.read().unwrap().as_ref() {
            hb.tick();
        }
    }
}

/// Summary of a pipeline run on stderr.
fn report(args: &Cli, dwi: &DwiData, mask_voxels: usize, out: &ConshPipelineResult) {
    if let (Some(path), Some(bmax_eff)) = (&args.graddev, out.grid_bmax) {
        let bmax = dwi.gtab.bvals.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!(
            "[consh] graddev: {}; effective maximum b {bmax_eff:.0} (nominal {bmax:.0}), \
             95th percentile |b_eff - b| {:.0}",
            path.display(),
            out.b_shift_p95.unwrap_or(0.0),
        );
    }
    if let Some(estimator) = out.estimator {
        eprintln!("[consh]   response estimator: {estimator:?}");
    }
    if let Some((rdiag, _)) = &out.estimate {
        eprintln!(
            "[consh]   tissue voxels: WM={} ({:.1}%)  GM={}  CSF={}   dense-b grid rows: {}",
            rdiag.n_wm_voxels,
            100.0 * rdiag.n_wm_voxels as f64 / rdiag.n_brain_voxels.max(1) as f64,
            rdiag.n_gm_voxels,
            rdiag.n_csf_voxels,
            out.responses.wm.n_shells(),
        );
        if let Some(st) = rdiag.stages {
            eprintln!(
                "[consh]   dhollander stages: mask {} → eroded {} → safe {}; \
                 crude WM/GM/CSF {}/{}/{}; refined {}/{}/{}",
                st.mask,
                st.eroded,
                st.safe,
                st.crude_wm,
                st.crude_gm,
                st.crude_csf,
                st.refined_wm,
                st.refined_gm,
                st.refined_csf,
            );
        }
        if let Some(fits) = &rdiag.dki {
            for (name, f) in [("WM", &fits.wm), ("GM", &fits.gm), ("CSF", &fits.csf)] {
                let p = &f.params;
                let shape = if p.isotropic {
                    format!("D={:.4} W={:.5}", p.d_par, p.w[0])
                } else {
                    format!(
                        "D∥={:.4} D⊥={:.4} W=[{:.5}, {:.5}, {:.5}]",
                        p.d_par, p.d_perp, p.w[0], p.w[1], p.w[2]
                    )
                };
                eprintln!(
                    "[consh]   DKI {name:<3} S0={:.4} {shape} C={:.5}  (µm²/ms)  \
                     RMSR={:.5} AIC={:.1} n={} iters={}{}",
                    p.s0,
                    p.offset,
                    f.rmsr,
                    f.aic,
                    f.n_samples,
                    f.iterations,
                    if f.converged { "" } else { " (not converged)" },
                );
            }
        }
    } else {
        let r = &out.responses;
        eprintln!(
            "[consh]   responses read from disk: WM lmax={} ({} rows), GM {} rows, CSF {} rows",
            r.wm.lmax,
            r.wm.n_shells(),
            r.gm.n_shells(),
            r.csf.n_shells()
        );
    }
    if let Some(conv) = &out.fit.converged {
        let n_conv = conv.iter().filter(|&&c| c == 1).count();
        eprintln!("[consh]   solver converged in {n_conv}/{mask_voxels} masked voxels");
    }
    if let Some(diag) = &out.normalization {
        eprintln!(
            "[consh]   balance factors{}: b_WM={:.4}  b_GM={:.4}  b_CSF={:.4}",
            if args.mtnorm_balanced { " (applied to output)" } else { "" },
            diag.tissue_scales[0],
            diag.tissue_scales[1],
            diag.tissue_scales[2]
        );
        eprintln!(
            "[consh]   target_sum={:.4}  lognorm_scale={:.4e}  fit voxels={}  mean |log residual|={:.3e}",
            diag.target_sum_used, diag.lognorm_scale, diag.n_fit_voxels, diag.mean_abs_log_residual
        );
    } else {
        eprintln!("[consh] stage 3/3: skipped (--no-normalize)");
    }
    if let Some((v, source)) = out.reference_b0 {
        let from = match source {
            ReferenceB0Source::UserMask => "the reference mask".to_string(),
            ReferenceB0Source::WhiteMatterSingleFibre => "single-fiber WM voxels".to_string(),
            ReferenceB0Source::WhiteMatterFraction => {
                format!("voxels with WM fraction ≥ {REFERENCE_WM_FRACTION}")
            }
        };
        eprintln!("[consh]   DWI_ref = {v:.1} (median b=0 in {from})");
    }
}

fn main() -> Result<()> {
    let args = Cli::parse();
    if args.odx.is_none()
        && (args.output_wm.is_none() || args.output_gm.is_none() || args.output_csf.is_none())
    {
        anyhow::bail!(
            "must pass either --odx <path> or all three of --output-wm/--output-gm/--output-csf"
        );
    }
    let provenance_builder = ProvenanceBuilder::new("consh", args.provenance);
    let (threads, source) = configure_rayon_threads(args.threads)?;
    if !args.quiet {
        let n = if threads == 0 { effective_thread_count() } else { threads };
        eprintln!("[consh] threads={} source={}", n, source.as_str());
    }

    let bvec_frame = if args.no_bvec_rotation {
        BvecFrame::ImageAxis
    } else {
        BvecFrame::WorldRas
    };
    let dwi = match &args.grad {
        Some(grad) => load_dwi_mrtrix_grad(
            &args.dwi,
            grad,
            args.mask.as_deref(),
            args.big_delta,
            args.small_delta,
            Some(args.gmax),
            bvec_frame,
        ),
        None => load_dwi(
            &args.dwi,
            args.bval.as_ref().expect("clap requires --bval without --grad"),
            args.bvec.as_ref().expect("clap requires --bvec without --grad"),
            args.mask.as_deref(),
            args.big_delta,
            args.small_delta,
            Some(args.gmax),
            bvec_frame,
        ),
    }
    .with_context(|| "failed to load DWI bundle")?;
    let mask_voxels = dwi.mask.iter().filter(|&&v| v).count();
    if !args.quiet {
        eprintln!(
            "[consh] DWI shape {:?}, mask voxels: {}, unique b (tol 50): {}",
            dwi.shape(),
            mask_voxels,
            dwi.gtab.shells(50.0).len()
        );
    }

    let graddev = match &args.graddev {
        Some(path) => Some(
            VoxelGradDev::load(path, args.graddev_identity.into(), &dwi)
                .with_context(|| format!("load gradient deviation field {}", path.display()))?,
        ),
        None => None,
    };
    let reference_mask = match &args.reference_mask {
        Some(path) => {
            let s = dwi.data.shape();
            Some(
                load_mask(path, [s[0], s[1], s[2]])
                    .with_context(|| format!("read reference mask {:?}", path))?,
            )
        }
        None => None,
    };

    let responses = match &args.read_responses_from {
        Some(dir) => {
            let read = |tissue: &str| {
                let path = dir.join(format!("{tissue}_response_denseb.txt"));
                TissueResponse::parse_mrtrix_txt(&path)
                    .with_context(|| format!("read {tissue} response from {}", path.display()))
            };
            ConshResponseSource::Provided(ConshResponses {
                wm: read("wm")?,
                gm: read("gm")?,
                csf: read("csf")?,
            })
        }
        None => ConshResponseSource::Estimate {
            restore: RestoreConfig {
                max_iter: args.restore_max_iter,
                tol: args.restore_tol,
                ..RestoreConfig::default()
            },
            responses: ConshResponseConfig {
                dhollander: DhollanderConfig {
                    fa_wm_threshold: args.fa_wm_threshold,
                    fiber_dominance_ratio: args.fiber_dominance_ratio,
                    md_csf_pct: args.md_csf_pct,
                    lmax_wm: args.lmax_wm,
                    legacy_selection: args.legacy_tissue_selection,
                    stages: DhollanderSelectConfig {
                        erode: args.dh_erode,
                        fa: args.dh_fa,
                        sfwm_pct: args.dh_sfwm,
                        gm_pct: args.dh_gm,
                        csf_pct: args.dh_csf,
                        ..DhollanderSelectConfig::default()
                    },
                },
                step: args.step,
                sigma_b: args.sigma_b,
                smooth: SmoothResponseConfig {
                    smoothing_b: args.sigma_b,
                    ..SmoothResponseConfig::default()
                },
                ..ConshResponseConfig::default()
            },
        },
    };
    let estimator = match args.response_estimator {
        EstimatorChoice::Auto => None,
        EstimatorChoice::Kernel => Some(ResponseEstimator::Kernel),
        EstimatorChoice::Smooth => Some(ResponseEstimator::Smooth),
        EstimatorChoice::Dki => Some(ResponseEstimator::Dki),
    };
    // The per-voxel penalty selection walks an internal ascending path whose
    // first entry (0) is the unpenalized reference.
    let wm_ridge_path = if args.wm_ridge_anchor > 0.0 {
        if args.wm_ridge_relative {
            vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        } else {
            vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        }
    } else {
        Vec::new()
    };
    let cfg = ConshPipelineConfig {
        responses,
        estimator,
        consh: ConshConfig {
            lmax_wm: args.lmax_wm,
            step: args.step,
            icls: IclsConfig { max_iter: args.icls_max_iter, ..IclsConfig::default() },
            wm_l2_ridge: args.wm_ridge,
            wm_ridge_path,
            l2_anchor_slack: args.wm_ridge_anchor,
            wm_ridge_relative: args.wm_ridge_relative,
            fit_noise_floor: !args.no_noise_floor,
            quantitative: args.quantitative,
            b_interp: args.b_interp,
            grid_bmax: None,
        },
        normalize: (!args.no_normalize).then(|| MtnormaliseConfig {
            poly_order: args.mtnorm_poly_order,
            target_sum: if args.mtnorm_target_median {
                None
            } else {
                Some(target_sum_mrtrix_default())
            },
            apply_balance: args.mtnorm_balanced,
            ..MtnormaliseConfig::default()
        }),
        compute_diagnostics: args.diagnostics,
        reference_mask,
    };
    if !args.quiet {
        if args.read_responses_from.is_some() {
            eprintln!("[consh] stage 1/3: reading dense-b responses (no tensor fit or estimation)");
        }
        if args.quantitative {
            eprintln!("[consh]   fODFs in absolute signal units (scaled by each voxel's S0)");
        }
    }
    let progress = StageProgress {
        current: RwLock::new(None),
        mask_voxels,
        interval: Duration::from_secs(args.progress_interval_secs.max(1)),
        quiet: args.quiet,
        noise_floor: !args.no_noise_floor,
    };
    let out = consh_pipeline(&dwi, graddev.as_ref(), &cfg, &progress)
        .with_context(|| "continuous-b multi-tissue fit failed")?;
    progress.finish_current();
    if !args.quiet {
        report(&args, &dwi, mask_voxels, &out);
    }
    for w in &out.warnings {
        eprintln!("[consh] warning: {w}");
    }
    if let (Some(dir), Some(_)) = (&args.write_responses_to, &out.estimate) {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create response directory {}", dir.display()))?;
        write_response_txt(&out.responses.wm, &dir.join("wm_response_denseb.txt"))?;
        write_response_txt(&out.responses.gm, &dir.join("gm_response_denseb.txt"))?;
        write_response_txt(&out.responses.csf, &dir.join("csf_response_denseb.txt"))?;
        if !args.quiet {
            eprintln!("[consh]   wrote dense-b responses to {}", dir.display());
        }
    }
    let result = out.fit;
    let responses = out.responses;
    let b_interp = out.b_interp;
    let reference_b0 = out.reference_b0;
    let mtnorm_target = out.normalization.as_ref().map(|d| d.target_sum_used);
    // Global gain of the applied field: the multiplier a typical (geometric
    // mean) inlier voxel received, the reciprocal of MRtrix3's `lognorm_scale`.
    let mtnorm_scale = out
        .normalization
        .as_ref()
        .and_then(|d| (d.lognorm_scale > 0.0).then(|| 1.0 / d.lognorm_scale));
    let estimator = out.estimator;

    // ---- Write outputs ----
    let quant_meta = ContinuousBQuantMeta {
        generated_by: GeneratedBy {
            name: "consh".into(),
            version: cs_dmri::VERSION.into(),
            description: "Continuous-b multi-tissue CSD for non-shelled q-space.".into(),
        },
        quantitative: args.quantitative,
        reference_b0,
        intensity_normalization_target: mtnorm_target,
        intensity_normalization_scale: mtnorm_scale,
        external_responses: args.read_responses_from.is_some(),
        noise_floor: !args.no_noise_floor,
        sh_degree: args.lmax_wm,
        graddev: args.graddev.clone(),
        b_interp,
        response_estimator: estimator.map(|e| format!("{e:?}")),
    };
    if let Some(odx_path) = &args.odx {
        let directory = odx_path.extension().and_then(|s| s.to_str()) != Some("odx");
        write_continuous_b_odx(
            odx_path,
            &args.dwi,
            &dwi.mask,
            &result.wm,
            &result.gm,
            &result.csf,
            args.lmax_wm,
            [&responses.wm, &responses.gm, &responses.csf],
            "consh_responses",
            args.step,
            result.floor.as_ref(),
            args.overwrite,
            directory,
            args.peak_min_amplitude,
            args.peak_min_amplitude_frac,
            Some(&quant_meta),
        )
        .with_context(|| format!("write ODX {:?}", odx_path))?;
        if !args.quiet {
            eprintln!("[consh] wrote ODX → {}", odx_path.display());
        }
    } else {
        let wm_path = args.output_wm.as_ref().unwrap();
        let gm_path = args.output_gm.as_ref().unwrap();
        let csf_path = args.output_csf.as_ref().unwrap();
        write_4d_f32(wm_path, &args.dwi, &result.wm, args.overwrite)
            .with_context(|| format!("write WM {:?}", wm_path))?;
        write_4d_f32(gm_path, &args.dwi, &result.gm, args.overwrite)
            .with_context(|| format!("write GM {:?}", gm_path))?;
        write_4d_f32(csf_path, &args.dwi, &result.csf, args.overwrite)
            .with_context(|| format!("write CSF {:?}", csf_path))?;
        if let Some(floor) = &result.floor {
            let floor_path = sibling_path(wm_path, "_noisefloor.nii.gz");
            write_4d_f32(&floor_path, &args.dwi, floor, args.overwrite)
                .with_context(|| format!("write noise floor {:?}", floor_path))?;
            if !args.quiet {
                eprintln!("[consh] wrote noise floor → {}", floor_path.display());
            }
        }
        if !args.quiet {
            eprintln!("[consh] wrote WM → {}", wm_path.display());
            eprintln!("[consh] wrote GM → {}", gm_path.display());
            eprintln!("[consh] wrote CSF → {}", csf_path.display());
        }
        if args.diagnostics {
            if let Some(conv) = &result.converged {
                write_3d_u8(&sibling_path(wm_path, "_converged.nii.gz"), &args.dwi, conv, args.overwrite)?;
            }
            if let Some(iters) = &result.iterations {
                write_3d_u32_as_f32(&sibling_path(wm_path, "_iters.nii.gz"), &args.dwi, iters, args.overwrite)?;
            }
        }
    }

    let _provenance = provenance_builder.map(|b| b.finish(effective_thread_count()));
    Ok(())
}
