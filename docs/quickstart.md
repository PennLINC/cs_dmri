# Quickstart

The examples assume a preprocessed series `dwi.nii.gz` with FSL-format
`dwi.bval` and `dwi.bvec` files and a brain mask `mask.nii.gz` on the same grid.

## Python

```python
import cs_dmri as cs

dwi = cs.DWI.from_files("dwi.nii.gz", "dwi.bval", "dwi.bvec", mask="mask.nii.gz")

# Image quality
qc = dwi.qc()
print(qc)
row = qc.to_dict()                      # one row of a QC table

# Diffusion tensor
tensor = cs.RestoreModel(dwi.gtab).fit(dwi)
fa_img = dwi.to_image(tensor.fa)        # nibabel image on the input grid

# 3D-SHORE
shore = cs.ShoreModel(dwi.gtab, radial_order=6).fit(dwi)
shore.export("shore.odx")

# Single-shell three-tissue CSD (single-shell data)
ss3t = cs.ss3t_pipeline(dwi)
ss3t.export("ss3t.fz")                  # for DSI Studio
```

A {class}`~cs_dmri.dwi.DWI` holds the data, gradient table, affine and mask of
one series, and caches results that several analyses share, such as the tensor
fit.

## Command line

```bash
cs-qc  --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
       --output-tsv qc.tsv

cs-dti --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
       --output-fa fa.nii.gz --output-md md.nii.gz \
       --output-s0 s0.nii.gz --output-outlier-fraction outliers.nii.gz

cs-fit --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
       --output coeffs.nii.gz --odx-output shore.odx

cs-ss3t-full --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
       --odx ss3t.odx
```

Each tool documents its options with `--help`; the same text is reproduced in
the {doc}`command-line reference <cli/index>`.
