# cs-response

Estimates white-matter, gray-matter and cerebrospinal-fluid response functions from single-shell data. See {doc}`../user/multitissue`.

```text
Estimate white matter, gray matter and CSF response functions (Dhollander et al., 2016)

Usage: cs-response [OPTIONS] --dwi <DWI> --bval <BVAL> --bvec <BVEC> --output-wm <OUTPUT_WM> --output-gm <OUTPUT_GM> --output-csf <OUTPUT_CSF>

Options:
      --dwi <DWI>
          4D DWI NIfTI input (b=0 volumes and one diffusion-weighted shell)

      --bval <BVAL>
          FSL bval file

      --bvec <BVEC>
          FSL bvec file

      --mask <MASK>
          Brain mask NIfTI. If omitted, a mask is computed by thresholding the mean b=0 image

      --output-wm <OUTPUT_WM>
          Output single-fiber white matter response (MRtrix `.txt` format)

      --output-gm <OUTPUT_GM>
          Output gray matter response (MRtrix `.txt` format, one column)

      --output-csf <OUTPUT_CSF>
          Output CSF response (MRtrix `.txt` format, one column)

      --dh-erode <DH_ERODE>
          Number of erosion passes applied to the brain mask before tissue selection. Not used with --legacy-tissue-selection
          
          [default: 3]

      --dh-fa <DH_FA>
          FA threshold for the initial separation of white matter from gray matter and CSF. Not used with --legacy-tissue-selection
          
          [default: 0.2]

      --dh-sfwm <DH_SFWM>
          Number of single-fiber white matter voxels selected, as a percentage of the refined white matter. Not used with --legacy-tissue-selection
          
          [default: 0.5]

      --dh-gm <DH_GM>
          Number of gray matter voxels selected, as a percentage of the refined gray matter. Not used with --legacy-tissue-selection
          
          [default: 2]

      --dh-csf <DH_CSF>
          Number of CSF voxels selected, as a percentage of the refined CSF. Not used with --legacy-tissue-selection
          
          [default: 10]

      --legacy-tissue-selection
          Use the earlier threshold-based tissue selection instead of the staged selection based on a signal decay metric. CSF voxels are those in the top --md-csf-pct percent of MD; single-fiber white matter voxels have FA above --fa-wm-threshold and eigenvalue ratio above --fiber-dominance-ratio; the remaining voxels are gray matter. The CSF class selected in this way can include partial-volume voxels

      --fa-wm-threshold <FA_WM_THRESHOLD>
          FA above which a voxel is a single-fiber white matter candidate. Used only with --legacy-tissue-selection
          
          [default: 0.7]

      --fiber-dominance-ratio <FIBER_DOMINANCE_RATIO>
          Minimum eigenvalue ratio λ₁ / mean(λ₂, λ₃) for a single-fiber white matter voxel; 0 disables the test. Used only with --legacy-tissue-selection
          
          [default: 2]

      --md-csf-pct <MD_CSF_PCT>
          Percentage of brain voxels with the highest MD that are selected as CSF. Used only with --legacy-tissue-selection
          
          [default: 2.5]

      --lmax-wm <LMAX_WM>
          Maximum SH order of the white matter response
          
          [default: 8]

      --restore-max-iter <RESTORE_MAX_ITER>
          Maximum number of RESTORE reweighting iterations in the tensor fit
          
          [default: 50]

      --restore-tol <RESTORE_TOL>
          RESTORE convergence tolerance on the relative change in tensor coefficients
          
          [default: 0.000001]

      --restore-outlier-threshold <RESTORE_OUTLIER_THRESHOLD>
          Geman-McClure weight below which a measurement is counted as an outlier in the tensor fit's outlier fraction. Does not affect the fit
          
          [default: 0.04]

      --big-delta <BIG_DELTA>
          Diffusion time Δ (big delta), in seconds. Not used by response estimation; accepted for consistency with `cs-fit`

      --small-delta <SMALL_DELTA>
          Gradient pulse duration δ (small delta), in seconds. Not used by response estimation

      --gmax <GMAX>
          Maximum gradient amplitude, in T/m. Used only when Δ and δ are estimated
          
          [default: 0.08]

      --diagnostics
          Also write the selected voxels of each tissue as mask NIfTIs (`<output_wm_stem>_mask_wm.nii.gz`, `_mask_gm.nii.gz`, `_mask_csf.nii.gz`)

      --no-bvec-rotation
          Use b-vectors in the image-axis (FSL) frame. By default b-vectors are rotated into world (RAS) coordinates

      --threads <THREADS>
          Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used, then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU

      --overwrite
          Overwrite existing `--diagnostics` mask NIfTIs. Without this flag, existing masks cause an error. Response files are always written

      --quiet
          Suppress periodic progress and per-step summary messages

      --progress-interval-secs <PROGRESS_INTERVAL_SECS>
          Interval between progress messages during the voxel-wise tensor fit, in seconds
          
          [default: 30]

      --provenance <PROVENANCE>
          Provenance mode. Accepted for consistency with the other tools; `cs- response` writes no provenance record

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
