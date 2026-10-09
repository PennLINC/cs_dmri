# cs_dmri

cs_dmri is a library and set of command-line tools for the analysis of
diffusion MRI data, written in Rust with a Python interface. It provides:

- image quality control: neighboring DWI correlation, a DWI contrast ratio,
  within-volume outlier-slice detection and a fixel-coherence index;
- 3D-SHORE reconstruction with L1-regularized (compressed-sensing), L2 or
  non-negativity-constrained fitting, with orientation distribution functions,
  fiber peaks, propagator-derived scalars and signal synthesis;
- robust diffusion tensor estimation (RESTORE);
- single-shell three-tissue constrained spherical deconvolution, including
  response-function estimation and multi-tissue intensity normalization.

Orientation outputs are written in the [ODX](https://github.com/PennLINC/odx-rs)
format. The command-line tools and the Python interface share one
implementation.

**Documentation:** <https://cs-dmri.readthedocs.io>

## Installation

Python:

```bash
pip install cs_dmri
```

Command-line tools (requires a Rust toolchain, a C compiler and CMake ≥ 3.26):

```bash
git clone https://github.com/PennLINC/cs_dmri
cd cs_dmri
cargo build --release      # tools are placed in target/release/
```

## Example

```python
import cs_dmri as cs

dwi = cs.DWI.from_files("dwi.nii.gz", "dwi.bval", "dwi.bvec", mask="mask.nii.gz")
print(dwi.qc())                                  # image-quality metrics
tensor = cs.RestoreModel(dwi.gtab).fit(dwi)      # FA, MD, ...
shore = cs.ShoreModel(dwi.gtab).fit(dwi)         # 3D-SHORE
shore.export("shore.odx")
```

```bash
cs-qc  --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz --output-tsv qc.tsv
cs-fit --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
       --output coeffs.nii.gz --odx-output shore.odx
```

## Command-line tools

| Tool | Purpose |
|---|---|
| `cs-qc` | Image-quality metrics |
| `cs-fit` | 3D-SHORE fit |
| `cs-odf` | ODFs, peaks and scalars from SHORE coefficients |
| `cs-synth` | Signal synthesis from SHORE coefficients |
| `cs-dti` | RESTORE diffusion tensor fit |
| `cs-response` | Three-tissue response-function estimation |
| `cs-ss3t` | Single-shell three-tissue CSD |
| `cs-mtnorm` | Multi-tissue intensity normalization |
| `cs-ss3t-full` | Response estimation, SS3T-CSD and normalization in one step |

## License

cs_dmri is available under either the MIT license or the Apache License 2.0, at
your option, except for two files derived from MRtrix3 (Mozilla Public License
2.0) and functions adapted from dipy (BSD 3-Clause). See the
[license page](https://cs-dmri.readthedocs.io/en/latest/license.html) and the
`LICENSE-*` files.
