# Diffusion tensor

The diffusion tensor is estimated with RESTORE (Chang, Jones and Pierpaoli,
2005), an iteratively reweighted least-squares fit of the log-signal in which
measurements with large residuals are down-weighted with the Geman–McClure
M-estimator. This makes the estimate robust to a small number of corrupted
measurements per voxel, such as those caused by motion during diffusion
encoding.

```python
tensor = cs.RestoreModel(dwi.gtab).fit(dwi)
tensor.fa, tensor.md, tensor.s0, tensor.outlier_fraction
tensor.tensor          # (X, Y, Z, 6): Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
tensor.principal_dir   # (X, Y, Z, 3)
tensor.evals           # (X, Y, Z, 3), descending
```

```bash
cs-dti --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
       --output-fa fa.nii.gz --output-md md.nii.gz --output-s0 s0.nii.gz \
       --output-outlier-fraction outliers.nii.gz --output-tensor tensor.nii.gz
```

## Outputs

| Output | Description |
|---|---|
| FA | Fractional anisotropy, between 0 and 1 |
| MD | Mean diffusivity, in units reciprocal to those of the $b$-values |
| $S_0$ | Estimated signal at $b = 0$ |
| Outlier fraction | Fraction of a voxel's measurements whose final weight falls below the outlier threshold |
| Tensor | Six unique elements, in the order Dxx, Dxy, Dxz, Dyy, Dyz, Dzz |
| Principal direction | Eigenvector of the largest eigenvalue |

The outlier fraction is a by-product of the reweighting and does not affect the
fit. Its threshold (default 0.04) is a Geman–McClure weight and corresponds
approximately to a residual of twice the robust noise estimate.

The tensor is fitted in the frame of the gradient directions supplied, so the
principal direction is expressed in the same frame (see {doc}`conventions`).
