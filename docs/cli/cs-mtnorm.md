# cs-mtnorm

Multi-tissue intensity normalization of tissue maps. See {doc}`../user/multitissue`.

```text
Multi-tissue intensity normalization and bias field correction in the log domain (Raffelt et al., 2017; Dhollander et al., 2021)

Usage: cs-mtnorm [OPTIONS] --in-wm <IN_WM> --in-gm <IN_GM> --in-csf <IN_CSF> --mask <MASK> --out-wm <OUT_WM> --out-gm <OUT_GM> --out-csf <OUT_CSF>

Options:
      --in-wm <IN_WM>
          Input white matter FOD NIfTI (4D, one volume per SH coefficient)
      --in-gm <IN_GM>
          Input gray matter NIfTI (3D, or 4D with one volume)
      --in-csf <IN_CSF>
          Input CSF NIfTI (3D, or 4D with one volume)
      --mask <MASK>
          Brain mask NIfTI (3D, binary)
      --out-wm <OUT_WM>
          Output normalized white matter FOD NIfTI (4D, one volume per SH coefficient)
      --out-gm <OUT_GM>
          Output normalized gray matter NIfTI (4D, one volume)
      --out-csf <OUT_CSF>
          Output normalized CSF NIfTI (4D, one volume)
      --poly-order <POLY_ORDER>
          Order of the polynomial bias field model (order 3 has 20 terms) [default: 3]
      --target-sum <TARGET_SUM>
          Target value for the sum over tissues of the normalized l=0 SH coefficients in each voxel. Default: 1/√(4π), as in MRtrix3 `mtnormalise`
      --target-median
          Use the median of the observed sums as the target. The global scale of the input is preserved and only the spatial bias field is removed
      --niter <NITER>
          Number of outer iterations (bias field updates) [default: 15]
      --balance-maxiter <BALANCE_MAXITER>
          Maximum number of iterations of the inner balance-factor and outlier-rejection loop in each outer iteration [default: 7]
      --balanced
          Multiply each output tissue by its balance factor, as in MRtrix3 `mtnormalise -balanced`. Balance factors are used in bias field estimation in either case; this option applies them to the outputs, which changes the relative scale of the tissues
      --diagnostics
          Also write the estimated bias field next to the white matter output (`<wm_stem>_bias.nii.gz`)
      --overwrite
          Overwrite existing output files. Without this flag, existing outputs cause an error
      --quiet
          Suppress per-step summary messages
  -h, --help
          Print help
  -V, --version
          Print version
```
