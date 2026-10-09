# cs_dmri

Python interface to cs_dmri, a Rust library for the analysis of diffusion MRI
data: image quality control, 3D-SHORE reconstruction, robust diffusion tensor
estimation (RESTORE) and single-shell three-tissue constrained spherical
deconvolution.

```bash
pip install cs_dmri
```

```python
import cs_dmri as cs

dwi = cs.DWI.from_files("dwi.nii.gz", "dwi.bval", "dwi.bvec", mask="mask.nii.gz")

qc = dwi.qc()                                   # image-quality metrics
qc.to_dict()                                    # one row of a QC table
cs.QCReport.column_descriptions()               # its data dictionary

tensor = cs.RestoreModel(dwi.gtab).fit(dwi)     # FA, MD, S0, principal direction
shore = cs.ShoreModel(dwi.gtab).fit(dwi)        # 3D-SHORE coefficients and diagnostics
ss3t = cs.ss3t_pipeline(dwi)                    # single-shell three-tissue CSD
```

Documentation: <https://cs-dmri.readthedocs.io>

Licence: `(MIT OR Apache-2.0) AND MPL-2.0 AND BSD-3-Clause`; see the
documentation for details.
