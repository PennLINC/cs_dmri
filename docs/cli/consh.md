# consh

Continuous-b multi-tissue CSD (ConSH): dense-b response estimation, deconvolution and intensity normalization for non-shelled or multi-shell data. See {doc}`../user/consh`.

```text
Continuous-b multi-tissue CSD (ConSH) for non-shelled diffusion MRI

Usage: consh [OPTIONS] --dwi <DWI>

Options:
      --dwi <DWI>
          4D DWI NIfTI input (non-shelled CS-DSI or any multi-b scheme)

      --bval <BVAL>
          FSL bval file

      --bvec <BVEC>
          FSL bvec file. Components along the image's voxel axes (the dipy/qsiprep convention)

      --grad <GRAD>
          MRtrix gradient table (`x y z b` per row, direction in world RAS), in place of --bval/--bvec. Unambiguous about the frame, which matters for --graddev

      --graddev <NII>
          Gradient-nonlinearity correction at fit time: a 9-volume deviation field on the DWI's exact grid (qsiprep `*_space-ACPC_graddev.nii.gz`, or an HCP/FSL `grad_dev.nii.gz`). Row-major `T` per voxel with `g_eff = Tᵀ g` in the image's voxel axes. Every voxel is deconvolved with its own effective gradient directions and b-values — correcting the direction- dependent b and apparent-shape distortion that rotating a fitted fODF afterwards (`odx graddev`) cannot. Implies --b-interp; the dense-b grid is extended to the largest effective b. Slower: each deviated voxel builds its own forward operator and factorization

      --graddev-identity <GRADDEV_IDENTITY>
          Whether the --graddev file stores `T` (qsiprep/TORTOISE) or `T − I` (HCP/FSL). `auto` decides from the median diagonal
          
          [default: auto]
          [possible values: auto, included, absent]

      --response-estimator <RESPONSE_ESTIMATOR>
          How dense-b response rows are estimated from the pooled single-tissue samples. `kernel`: each row on its own from a Gaussian-in-b window (`--sigma-b`); matches MRtrix's per-shell responses on shelled data, but is step-like there and noisy between the b-values of sparse non-shelled schemes. `smooth`: all rows jointly, linear in b between rows and smooth across rows over about `--sigma-b`, constrained (spherical mean ≥ 0 and non-increasing in b; alternating signs of the WM zonal coefficients). `dki`: the parametric response of Morez et al. (2021), axially symmetric DKI plus an offset per tissue (7 parameters for WM, 4 for GM and CSF), fit to all samples at once and non-increasing in b; never chosen by `auto`. `auto`: with --graddev, `smooth` if the field moves b by more than --sigma-b (95th percentile over the mask), else `kernel`; without --graddev, `smooth` with --b-interp, else `kernel`
          
          [default: auto]
          [possible values: auto, kernel, smooth, dki]

      --b-interp
          Interpolate the response linearly in b between dense-b grid rows instead of snapping each measurement to the nearest row (±step/2). Identical on b-values that sit on grid points; on by default with --graddev

      --mask <MASK>
          Brain mask NIfTI. If omitted, a mask is computed from the mean b=0 image

      --output-wm <OUTPUT_WM>
          Output WM fODF NIfTI (SH coefficients, MRtrix basis). Required unless `--odx` is given

      --output-gm <OUTPUT_GM>
          Output GM compartment NIfTI. Required unless `--odx` is given

      --output-csf <OUTPUT_CSF>
          Output CSF compartment NIfTI. Required unless `--odx` is given

      --odx <PATH>
          Write a single ODX file containing the WM SH coefficients, the GM and CSF compartments, the mask and WM peaks, in place of the three NIfTI outputs. A path with a `.odx` extension is written as a zip archive; any other path is written as a directory

      --write-responses-to <WRITE_RESPONSES_TO>
          Write the estimated dense-b responses (one row per grid point) to this directory, as `{wm,gm,csf}_response_denseb.txt`

      --read-responses-from <READ_RESPONSES_FROM>
          Read dense-b responses from this directory (the three `{wm,gm,csf}_response_denseb.txt` files `--write-responses-to` writes, or a group average from `consh-response mean`) instead of estimating them. The tensor fit and tissue selection are skipped, and every scan fit with the same files is deconvolved with the same responses. The response rows must match this series' dense-b grid (same maximum b and step)

      --no-normalize
          Skip the final multi-tissue intensity normalization (mtnormalise)

      --no-noise-floor
          Fit without the constant noise-floor compartment (fitted by default).
          
          The magnitude noise floor is additive in absolute signal units, while the multi-tissue model is multiplicative with S0-normalized responses. In voxels whose S0 is far below that of the response-estimation voxels (cisternal and subarachnoid CSF, non-brain tissue inside the mask) the three tissue responses cannot reproduce the relative floor, and because WM is the only compartment that keeps signal at high b, the non-negative fit assigns the excess to the WM fODF. A constant compartment absorbs it at the cost of one degree of freedom and needs no noise estimate. Its per-voxel amplitude is written to the ODX as the `noise_floor` per-voxel value (on the GM/CSF scale; divide by 0.2821 for a fraction of S0), and with NIfTI outputs as a `_noisefloor.nii.gz` file next to the WM output.

      --step <STEP>
          Dense b-grid step in s/mm². Must be fine enough to resolve the b-values of the acquisition
          
          [default: 50]

      --sigma-b <SIGMA_B>
          Width (standard deviation, s/mm²) of the Gaussian window in b used by the `kernel` response estimator, and the smoothing length of `smooth`
          
          [default: 150]

      --lmax-wm <LMAX_WM>
          Maximum SH order of the white matter fODF and response
          
          [default: 8]

      --icls-max-iter <ICLS_MAX_ITER>
          Maximum active-set iterations of the constrained least-squares solver. The WM non-negativity constraint is imposed at about 700 directions, so this is higher than the SS3T default (200)
          
          [default: 2000]

      --wm-ridge <WM_RIDGE>
          Tikhonov (L2) penalty weight on the WM fODF coefficients with l ≥ 2; the l=0 term and the GM/CSF amplitudes are not penalized. 0 (default) leaves only the non-negativity constraint. A positive value damps the ill-conditioned high-order terms that make peaks in crossing regions unstable; values of 0.1–5 are typical
          
          [default: 0]

      --wm-ridge-anchor <WM_RIDGE_ANCHOR>
          Choose the WM penalty weight per voxel: the largest weight on an internal path (0 to 8) whose residual stays within this fraction of the unpenalized fit's. 0 (default) is off. Damps ambiguous crossings while leaving well-determined single-fiber voxels sharp; values of 0.02–0.10 are typical. Takes precedence over --wm-ridge
          
          [default: 0]

      --wm-ridge-relative
          Scale the WM penalty by each l ≥ 2 coefficient's own diagonal of HᵀH, so that one `--wm-ridge` or `--wm-ridge-anchor` setting means the same across acquisitions (shelled or Cartesian, any maximum b). With `--wm-ridge-anchor` the internal path becomes relative fractions (0 to 1.6)

      --peak-min-amplitude <PEAK_MIN_AMPLITUDE>
          Drop WM peaks whose fODF amplitude is below this value. Removes shallow spurious lobes, which become more frequent under strong penalties. 0 (default) keeps all peaks; values of 0.05–0.15 on the normalized fODF scale are typical.
          
          The threshold is tied to the absolute fODF scale, which changes with `--quantitative` and between acquisitions; `--peak-min-amplitude-frac` is preferable for comparisons across subjects.
          
          [default: 0]

      --peak-min-amplitude-frac <PEAK_MIN_AMPLITUDE_FRAC>
          Scale-invariant form of `--peak-min-amplitude`: the threshold is this fraction of the volume's median dominant-peak amplitude (the median, over masked voxels with at least one peak, of each voxel's largest peak).
          
          The reference and the peaks scale together under any global change of fODF units, so one value means the same with or without `--quantitative` and across acquisitions. 0 (default) is off; values of 0.10–0.25 are typical.
          
          [default: 0]

      --reference-mask <NIFTI>
          White-matter mask defining the reference b=0 (`DWI_ref`) recorded in the ODX sidecar. Direct analogue of `dwinormalise individual`, which takes a mask and normalizes its median b=0.
          
          If omitted, the reference is taken from the Dhollander single-fiber WM voxels when responses are estimated, or from voxels where the fitted WM compartment dominates when they are read from disk.

      --quantitative
          Write fODFs in absolute signal units, multiplying each voxel's fitted coefficients by its b=0 signal, instead of the default per-voxel S/S0 scale. The WM fODF amplitude is then proportional to the signal (and so to intra-cellular volume at high b), which apparent fiber density and SIFT2 require and per-voxel S/S0 normalization removes (Smith et al. 2022; Dhollander et al. 2021).
          
          The responses are scaled by one per-scan reference (the median b=0 of the single-fiber WM voxels), which keeps the relative amplitudes of the tissues as MRtrix's `amp2response` does. A common scale across subjects comes from a shared group-average response (`consh-response mean`, read with --read-responses-from), as in MRtrix; the remaining per-subject scale is removed by the final normalization, so this option cannot be combined with --no-normalize.
          
          Scaling by S0 also suppresses the large spurious fODFs of low-S0 voxels, which come from dividing the noise floor by S0. It corrects only their magnitude; the noise-floor compartment corrects their orientation content.

      --dh-erode <DH_ERODE>
          Erosion passes applied to the brain mask before tissue selection
          
          [default: 3]

      --dh-fa <DH_FA>
          FA threshold for the crude WM vs GM-CSF split
          
          [default: 0.2]

      --dh-sfwm <DH_SFWM>
          Final single-fiber WM voxels, as a percentage of refined WM
          
          [default: 0.5]

      --dh-gm <DH_GM>
          Final GM voxels, as a percentage of refined GM
          
          [default: 2]

      --dh-csf <DH_CSF>
          Final CSF voxels, as a percentage of refined CSF
          
          [default: 10]

      --legacy-tissue-selection
          Use the earlier threshold-based tissue selection (top-N% MD for CSF, FA and eigenvalue dominance for WM) instead of MRtrix's staged signal-decay-metric algorithm. For reproducing earlier results only: its CSF class includes partial-volume voxels, which lowers the CSF response amplitude

      --fa-wm-threshold <FA_WM_THRESHOLD>
          FA above this counts a voxel as a WM single-fiber candidate. `--legacy-tissue-selection` only
          
          [default: 0.7]

      --fiber-dominance-ratio <FIBER_DOMINANCE_RATIO>
          Eigenvalue ratio gate for "single fiber" (suppresses crossings). `--legacy-tissue-selection` only
          
          [default: 2]

      --md-csf-pct <MD_CSF_PCT>
          Top-N percent of MD voxels classified as CSF. `--legacy-tissue-selection` only
          
          [default: 2.5]

      --mtnorm-poly-order <MTNORM_POLY_ORDER>
          Order of the polynomial bias field model in the normalization step (order 3 has 20 terms)
          
          [default: 3]

      --mtnorm-target-median
          In the normalization step, use the median of the observed sums of l=0 coefficients as the target instead of 1/√(4π)

      --mtnorm-balanced
          Apply the per-tissue balance factors to the output, like MRtrix3 `mtnormalise -balanced`. With a group-average response the factors absorb the scale offset between the tissues' responses, which is nearly constant across subjects, and bring WM tissue sums to the target. Off by default, as in MRtrix

      --restore-max-iter <RESTORE_MAX_ITER>
          Maximum number of RESTORE reweighting iterations in the tensor fit used for response estimation
          
          [default: 50]

      --restore-tol <RESTORE_TOL>
          RESTORE convergence tolerance on the relative change in tensor coefficients
          
          [default: 0.000001]

      --big-delta <BIG_DELTA>
          Diffusion time Δ (big delta), in seconds. Not used by the reconstruction; accepted for consistency with `cs-fit`

      --small-delta <SMALL_DELTA>
          Gradient pulse duration δ (small delta), in seconds. Not used by the reconstruction

      --gmax <GMAX>
          Maximum gradient amplitude, in T/m. Used only when Δ and δ are estimated
          
          [default: 0.08]

      --diagnostics
          Also write per-voxel maps of convergence (`_converged.nii.gz`) and iteration count (`_iters.nii.gz`) next to `--output-wm`. Not written with `--odx`

      --no-bvec-rotation
          Fit with b-vectors in the image-axis (FSL) frame. By default b-vectors are rotated into world (RAS) coordinates before fitting

      --threads <THREADS>
          Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used, then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU

      --overwrite
          Overwrite existing output files. Without this flag, existing outputs cause an error

      --quiet
          Suppress periodic progress and per-step summary messages

      --progress-interval-secs <PROGRESS_INTERVAL_SECS>
          Interval between progress messages, in seconds
          
          [default: 30]

      --provenance <PROVENANCE>
          Provenance mode. Accepted for consistency with the other tools; `consh` writes no provenance record

          Possible values:
          - minimal: Version, build and run-time summary only. Default
          - full:    Additionally records the command line, host name and start time
          - none:    No provenance
          
          [default: minimal]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
