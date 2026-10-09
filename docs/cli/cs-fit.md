# cs-fit

Fits a 3D-SHORE basis to every voxel in the mask and writes the coefficients with a JSON sidecar; optionally also an ODX file. See {doc}`../user/shore`.

```text
Compressed-sensing dMRI fit (3D-SHORE basis)

Usage: cs-fit [OPTIONS] --dwi <DWI> --bval <BVAL> --bvec <BVEC> --output <OUTPUT>

Options:
      --dwi <DWI>
          4D DWI NIfTI input

      --bval <BVAL>
          FSL bval file

      --bvec <BVEC>
          FSL bvec file

      --mask <MASK>
          Brain mask NIfTI. If omitted, a mask is computed from the mean b=0 image

      --big-delta <BIG_DELTA>
          Diffusion time Δ (big delta), in seconds. If either Δ or δ is omitted, both are estimated from the maximum b-value and --gmax

      --small-delta <SMALL_DELTA>
          Gradient pulse duration δ (small delta), in seconds

      --gmax <GMAX>
          Maximum gradient amplitude, in T/m. Used only when Δ and δ are estimated
          
          [default: 0.08]

      --output <OUTPUT>
          Output coefficient NIfTI path

      --radial-order <RADIAL_ORDER>
          SHORE radial order. Even values are expected
          
          [default: 6]

      --zeta <ZETA>
          SHORE scale parameter ζ
          
          [default: 700]

      --reg <REG>
          Regularization of the coefficient fit

          Possible values:
          - l1:     L1-regularized (sparse) fit solved with FISTA. Default
          - l2:     L2 (Tikhonov) fit with a closed-form solution
          - amp-nn: Least-squares fit subject to non-negativity of the ODF amplitudes on a dense sphere, solved with the Goldfarb-Idnani inequality-constrained least-squares method
          
          [default: l1]

      --alpha-mode <ALPHA_MODE>
          Selection of the L1 sparsity weight α, made independently for each voxel except in `fixed` mode

          Possible values:
          - fixed:       The same α for every voxel (--alpha)
          - alpha-ratio: α = ratio · α_max for each voxel, where α_max is the smallest α giving an all-zero solution (--alpha-ratio)
          - path-bic:    Along a logarithmic path from α_max to α_max·eps, the α minimizing the Bayesian information criterion
          - l2-anchored: Along the same path, the largest α whose residual sum of squares is within (1 + slack) of a Tikhonov (L2) fit's, using --lambda-n and --lambda-l. If no α satisfies the bound, the α with the smallest residual sum of squares is used. Default
          
          [default: l2-anchored]

      --alpha <ALPHA>
          L1 sparsity weight α. Used only with `--alpha-mode fixed`
          
          [default: 1]

      --alpha-ratio <ALPHA_RATIO>
          Ratio α / α_max, in (0, 1). Used only with `--alpha-mode alpha-ratio`
          
          [default: 0.001]

      --path-n-alphas <PATH_N_ALPHAS>
          Number of α values on the regularization path (path-bic, l2-anchored). Must be at least 2
          
          [default: 20]

      --path-eps <PATH_EPS>
          Ratio α_min / α_max of the regularization path (path-bic, l2-anchored), in (0, 1). Default: 1e-3 for path-bic and 1e-4 for l2-anchored

      --slack <SLACK>
          Residual tolerance for `--alpha-mode l2-anchored`: the selected α satisfies RSS ≤ (1 + slack) · RSS_L2. Smaller values give fits closer to the L2 fit; larger values give sparser fits. Must be ≥ 0
          
          [default: 0.05]

      --max-iter <MAX_ITER>
          Maximum number of FISTA iterations per L1 fit, including each fit on the regularization path
          
          [default: 1000]

      --tol <TOL>
          L1 convergence tolerance on the relative change in coefficients
          
          [default: 0.000001]

      --non-negative
          Enforce non-negative coefficients during the L1 fit

      --lambda-n <LAMBDA_N>
          L2 radial regularization weight λ_N
          
          [default: 0.00000001]

      --lambda-l <LAMBDA_L>
          L2 angular regularization weight λ_L
          
          [default: 0.00000001]

      --amp-nn-max-iter <AMP_NN_MAX_ITER>
          Maximum number of active-set iterations for `--reg amp-nn`
          
          [default: 200]

      --amp-nn-tol <AMP_NN_TOL>
          Constraint tolerance for `--reg amp-nn`: an ODF amplitude is treated as non-negative if it is ≥ -tol
          
          [default: 0.000000001]

      --amp-nn-epsilon <AMP_NN_EPSILON>
          Tikhonov term added to the diagonal of HᵀH for `--reg amp-nn`. Larger values are needed if the Cholesky factorization fails on a rank-deficient design
          
          [default: 0.0000000001]

      --diagnostics
          Also write per-voxel maps of R², residual, iteration count and regularization type, and, for L1 fits with per-voxel α selection, the selected α, next to the coefficient NIfTI

      --no-bvec-rotation
          Fit with b-vectors in the image-axis (FSL) frame. By default b-vectors are rotated into world (RAS) coordinates before fitting

      --threads <THREADS>
          Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used, then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU

      --overwrite
          Overwrite existing output files. Without this flag, existing outputs cause an error. Applies to the coefficient NIfTI, the sidecar JSON and the `--diagnostics` maps

      --quiet
          Suppress periodic progress and per-step summary messages. Errors are still written to stderr

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

      --odx-output <ODX_OUTPUT>
          Also write an ODX file of ODF SH coefficients (MRtrix3/Tournier convention) to this path, using the `cs-odf` defaults: DSI Studio ODF8 peak finding, brain-wide ODF normalization, an anisotropic power map, and lmax equal to the largest even integer ≤ --radial-order. For other settings, run `cs-odf` on the coefficient NIfTI

      --odx-directory
          Write `--odx-output` as a directory instead of a `.odx` zip archive, as with `cs-odf --directory`

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
