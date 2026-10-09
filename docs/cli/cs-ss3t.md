# cs-ss3t

Single-shell three-tissue CSD with given response functions. See {doc}`../user/multitissue`.

```text
Single-shell three-tissue constrained spherical deconvolution (SS3T-CSD; Dhollander and Connelly, 2016)

Usage: cs-ss3t [OPTIONS] --dwi <DWI> --bval <BVAL> --bvec <BVEC> --response-wm <RESPONSE_WM> --response-gm <RESPONSE_GM> --response-csf <RESPONSE_CSF> --output-wm <OUTPUT_WM> --output-gm <OUTPUT_GM> --output-csf <OUTPUT_CSF>

Options:
      --dwi <DWI>
          4D DWI NIfTI input (b=0 volumes and one diffusion-weighted shell)

      --bval <BVAL>
          FSL bval file

      --bvec <BVEC>
          FSL bvec file

      --mask <MASK>
          Brain mask NIfTI. If omitted, a mask is computed from the mean b=0 image

      --response-wm <RESPONSE_WM>
          Single-fibre white matter response in MRtrix `.txt` format, with two rows: b=0 (isotropic) and the diffusion-weighted shell (lmax at least --lmax-wm)

      --response-gm <RESPONSE_GM>
          Grey matter response in MRtrix `.txt` format: two rows of one column each (lmax = 0)

      --response-csf <RESPONSE_CSF>
          CSF response in MRtrix `.txt` format: two rows of one column each (lmax = 0)

      --output-wm <OUTPUT_WM>
          Output white matter FOD NIfTI (4D, one volume per SH coefficient)

      --output-gm <OUTPUT_GM>
          Output grey matter compartment NIfTI (4D, one volume)

      --output-csf <OUTPUT_CSF>
          Output CSF compartment NIfTI (4D, one volume)

      --niter <NITER>
          Number of SS3T outer iterations. Must be at least 2
          
          [default: 3]

      --bzero-pct <BZERO_PCT>
          Weight of the b=0 volumes in the fit, as a percentage of the diffusion-weighted volumes. Must be positive
          
          [default: 10]

      --lmax-wm-strategy <LMAX_WM_STRATEGY>
          Selection of the white matter SH order (lmax)

          Possible values:
          - fixed:    The same lmax (--lmax-wm) for every voxel. Default
          - path-bic: For each voxel, the lmax in --lmax-wm-candidates that minimises the Bayesian information criterion
          
          [default: fixed]

      --lmax-wm <LMAX_WM>
          White matter SH order for `--lmax-wm-strategy fixed`. Limited to the lmax of the white matter response
          
          [default: 8]

      --lmax-wm-candidates <LMAX_WM_CANDIDATES>
          Comma-separated candidate even SH orders for `--lmax-wm-strategy path-bic`
          
          [default: 0 2 4 6 8]

      --icls-max-iter <ICLS_MAX_ITER>
          Maximum number of active-set iterations of the inner inequality-constrained least-squares (ICLS) solver
          
          [default: 200]

      --icls-tol <ICLS_TOL>
          ICLS constraint tolerance: constraint i is satisfied if (C x)_i ≥ -tol
          
          [default: 0.0000000001]

      --icls-epsilon <ICLS_EPSILON>
          Tikhonov term ε added to the diagonal of HᵀH in the ICLS solver to ensure strict positive definiteness
          
          [default: 0.0000000001]

      --big-delta <BIG_DELTA>
          Diffusion time Δ (big delta), in seconds. Not used by SS3T-CSD; accepted for consistency with `cs-fit`

      --small-delta <SMALL_DELTA>
          Gradient pulse duration δ (small delta), in seconds. Not used by SS3T-CSD

      --gmax <GMAX>
          Maximum gradient amplitude, in T/m. Used only when Δ and δ are estimated
          
          [default: 0.08]

      --diagnostics
          Also write per-voxel maps of iteration count (`_iters.nii.gz`), residual (`_residual.nii.gz`) and convergence (`_converged.nii.gz`) next to the white matter output

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
          Provenance recorded in the sidecar JSON

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
