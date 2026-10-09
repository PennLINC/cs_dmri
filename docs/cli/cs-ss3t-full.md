# cs-ss3t-full

Response estimation, SS3T-CSD and intensity normalisation in one step. See {doc}`../user/multitissue`.

```text
Single-shell three-tissue pipeline: response function estimation, SS3T-CSD and multi-tissue intensity normalisation

Usage: cs-ss3t-full [OPTIONS] --dwi <DWI> --bval <BVAL> --bvec <BVEC>

Options:
      --dwi <DWI>
          4D DWI NIfTI input (b=0 volumes and one diffusion-weighted shell)

      --bval <BVAL>
          FSL bval file

      --bvec <BVEC>
          FSL bvec file

      --mask <MASK>
          Brain mask NIfTI. If omitted, a mask is computed from the mean b=0 image

      --output-wm <OUTPUT_WM>
          Output white matter FOD NIfTI, normalised unless `--no-normalize` is given. Required unless `--odx` is given

      --output-gm <OUTPUT_GM>
          Output grey matter compartment NIfTI. Required unless `--odx` is given

      --output-csf <OUTPUT_CSF>
          Output CSF compartment NIfTI. Required unless `--odx` is given

      --odx <PATH>
          Write a single ODX file containing the white matter SH coefficients, the grey matter and CSF compartments (under `sh/`), the brain mask, the response functions (in the header) and white matter peaks. The NIfTI outputs are then not written. A path with a `.odx` extension is written as a zip archive; any other path is written as a directory

      --response-wm <RESPONSE_WM>
          White matter response in MRtrix `.txt` format. If all three of `--response-{wm,gm,csf}` are given, response estimation is skipped

      --response-gm <RESPONSE_GM>
          Grey matter response in MRtrix `.txt` format (see --response-wm)

      --response-csf <RESPONSE_CSF>
          CSF response in MRtrix `.txt` format (see --response-wm)

      --write-responses-to <WRITE_RESPONSES_TO>
          Directory in which to write the estimated responses (`wm_response.txt`, `gm_response.txt`, `csf_response.txt`). Ignored when responses are supplied with `--response-*`

      --no-normalize
          Do not apply multi-tissue intensity normalisation; the SS3T-CSD outputs are written as fitted

      --dh-erode <DH_ERODE>
          Number of erosion passes applied to the brain mask before tissue selection. Not used with --legacy-tissue-selection
          
          [default: 3]

      --dh-fa <DH_FA>
          FA threshold for the initial separation of white matter from grey matter and CSF. Not used with --legacy-tissue-selection
          
          [default: 0.2]

      --dh-sfwm <DH_SFWM>
          Number of single-fibre white matter voxels selected, as a percentage of the refined white matter. Not used with --legacy-tissue-selection
          
          [default: 0.5]

      --dh-gm <DH_GM>
          Number of grey matter voxels selected, as a percentage of the refined grey matter. Not used with --legacy-tissue-selection
          
          [default: 2]

      --dh-csf <DH_CSF>
          Number of CSF voxels selected, as a percentage of the refined CSF. Not used with --legacy-tissue-selection
          
          [default: 10]

      --legacy-tissue-selection
          Use the earlier threshold-based tissue selection instead of the staged selection based on a signal decay metric. CSF voxels are those in the top --md-csf-pct percent of MD; single-fibre white matter voxels have FA above --fa-wm-threshold and eigenvalue ratio above --fiber-dominance-ratio; the remaining voxels are grey matter. The CSF class selected in this way can include partial-volume voxels

      --fa-wm-threshold <FA_WM_THRESHOLD>
          FA above which a voxel is a single-fibre white matter candidate. Used only with --legacy-tissue-selection
          
          [default: 0.7]

      --fiber-dominance-ratio <FIBER_DOMINANCE_RATIO>
          Minimum eigenvalue ratio λ₁ / mean(λ₂, λ₃) for a single-fibre white matter voxel; 0 disables the test. Used only with --legacy-tissue-selection
          
          [default: 2]

      --md-csf-pct <MD_CSF_PCT>
          Percentage of brain voxels with the highest MD that are selected as CSF. Used only with --legacy-tissue-selection
          
          [default: 2.5]

      --niter <NITER>
          Number of SS3T outer iterations. Must be at least 2
          
          [default: 3]

      --bzero-pct <BZERO_PCT>
          Weight of the b=0 volumes in the SS3T fit, as a percentage of the diffusion-weighted volumes. Must be positive
          
          [default: 10]

      --lmax-wm <LMAX_WM>
          Maximum SH order of the white matter FOD and response
          
          [default: 8]

      --mtnorm-poly-order <MTNORM_POLY_ORDER>
          Order of the polynomial bias field model in the normalisation step (order 3 has 20 terms)
          
          [default: 3]

      --mtnorm-target-median
          In the normalisation step, use the median of the observed sums of l=0 coefficients as the target instead of 1/√(4π). The global scale of the input is preserved

      --mtnorm-balanced
          Multiply each output tissue by its balance factor, as in MRtrix3 `mtnormalise -balanced`

      --restore-max-iter <RESTORE_MAX_ITER>
          Maximum number of RESTORE reweighting iterations in the tensor fit used for response estimation
          
          [default: 50]

      --restore-tol <RESTORE_TOL>
          RESTORE convergence tolerance on the relative change in tensor coefficients
          
          [default: 0.000001]

      --big-delta <BIG_DELTA>
          Diffusion time Δ (big delta), in seconds. Not used by the pipeline; accepted for consistency with `cs-fit`

      --small-delta <SMALL_DELTA>
          Gradient pulse duration δ (small delta), in seconds. Not used by the pipeline

      --gmax <GMAX>
          Maximum gradient amplitude, in T/m. Used only when Δ and δ are estimated
          
          [default: 0.08]

      --diagnostics
          Also write per-voxel maps of iteration count (`_iters.nii.gz`), residual (`_residual.nii.gz`) and convergence (`_converged.nii.gz`) next to `--output-wm`. Not written with `--odx`

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
          Provenance mode. Accepted for consistency with the other tools; `cs- ss3t-full` writes no provenance record

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
