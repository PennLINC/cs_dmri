# cs-dti

Fits the diffusion tensor with RESTORE. See {doc}`../user/dti`.

```text
Robust diffusion tensor fit with RESTORE (Chang et al., MRM 2005)

Usage: cs-dti [OPTIONS] --dwi <DWI> --bval <BVAL> --bvec <BVEC> --output-fa <OUTPUT_FA> --output-md <OUTPUT_MD> --output-s0 <OUTPUT_S0> --output-outlier-fraction <OUTPUT_OUTLIER_FRACTION>

Options:
      --dwi <DWI>
          4D DWI NIfTI input

      --bval <BVAL>
          FSL bval file

      --bvec <BVEC>
          FSL bvec file

      --mask <MASK>
          Brain mask NIfTI. If omitted, a mask is computed by thresholding the mean b=0 image

      --output-fa <OUTPUT_FA>
          Output fractional anisotropy (FA) NIfTI (3D, range [0, 1])

      --output-md <OUTPUT_MD>
          Output mean diffusivity (MD) NIfTI (3D), in the inverse of the b-value units

      --output-s0 <OUTPUT_S0>
          Output S₀ NIfTI (3D): fitted signal at b=0, in the units of the input

      --output-outlier-fraction <OUTPUT_OUTLIER_FRACTION>
          Output outlier-fraction NIfTI (3D, range [0, 1]): the fraction of measurements in each voxel classified as outliers (see --outlier-threshold)

      --output-tensor <OUTPUT_TENSOR>
          Output tensor NIfTI (4D, six components in the order Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)

      --output-principal-dir <OUTPUT_PRINCIPAL_DIR>
          Output principal eigenvector NIfTI (4D, three components)

      --max-iter <MAX_ITER>
          Maximum number of RESTORE reweighting iterations
          
          [default: 50]

      --tol <TOL>
          RESTORE convergence tolerance on the relative change in tensor coefficients
          
          [default: 0.000001]

      --outlier-threshold <OUTLIER_THRESHOLD>
          Geman-McClure weight below which a measurement is counted as an outlier in the outlier-fraction map. Does not affect the fit
          
          [default: 0.04]

      --big-delta <BIG_DELTA>
          Diffusion time Δ (big delta), in seconds. Not used by the tensor fit; accepted for consistency with `cs-fit`

      --small-delta <SMALL_DELTA>
          Gradient pulse duration δ (small delta), in seconds. Not used by the tensor fit

      --gmax <GMAX>
          Maximum gradient amplitude, in T/m. Used only when Δ and δ are estimated
          
          [default: 0.08]

      --diagnostics
          Also write per-voxel iteration-count (`_iters`) and convergence (`_converged`) maps next to the FA output

      --no-bvec-rotation
          Fit with b-vectors in the image-axis (FSL) frame. By default b-vectors are rotated into world (RAS) coordinates before fitting

      --threads <THREADS>
          Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used, then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU

      --overwrite
          Overwrite existing output files. Without this flag, existing outputs cause an error

      --quiet
          Suppress periodic progress and per-step summary messages

      --progress-interval-secs <PROGRESS_INTERVAL_SECS>
          Interval between progress messages during the voxel-wise fit, in seconds
          
          [default: 30]

      --provenance <PROVENANCE>
          Provenance mode. Accepted for consistency with the other tools; `cs-dti` writes no provenance record

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
