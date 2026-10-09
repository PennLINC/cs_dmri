# Command-line reference

Each page reproduces the tool's `--help` output.

| Tool | Purpose |
|---|---|
| [`cs-qc`](cs-qc.md) | Image-quality metrics for a DWI series, written as JSON and/or a one-row TSV with a JSON data dictionary. |
| [`cs-fit`](cs-fit.md) | Fits a 3D-SHORE basis to every voxel in the mask and writes the coefficients with a JSON sidecar; optionally also an ODX file. |
| [`cs-odf`](cs-odf.md) | Computes ODF spherical-harmonic coefficients, peaks and scalars from `cs-fit` coefficients and writes an ODX file. |
| [`cs-synth`](cs-synth.md) | Synthesizes a DWI series for a target gradient table from `cs-fit` coefficients. |
| [`cs-dti`](cs-dti.md) | Fits the diffusion tensor with RESTORE. |
| [`cs-response`](cs-response.md) | Estimates white-matter, grey-matter and cerebrospinal-fluid response functions from single-shell data. |
| [`cs-ss3t`](cs-ss3t.md) | Single-shell three-tissue CSD with given response functions. |
| [`cs-mtnorm`](cs-mtnorm.md) | Multi-tissue intensity normalisation of tissue maps. |
| [`cs-ss3t-full`](cs-ss3t-full.md) | Response estimation, SS3T-CSD and intensity normalisation in one step. |

```{toctree}
:hidden:

cs-qc
cs-fit
cs-odf
cs-synth
cs-dti
cs-response
cs-ss3t
cs-mtnorm
cs-ss3t-full
```
