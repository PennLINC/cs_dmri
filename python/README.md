# cs-dmri (Python)

Python bindings for [cs-dmri](https://github.com/PennLINC/cs-dmri): diffusion
MRI quality control, 3D-SHORE compressed-sensing fits, RESTORE tensors and
single-shell three-tissue CSD, implemented in Rust. Results are identical to
the `cs-*` command-line tools.

```bash
pip install cs-dmri            # numpy + nibabel
pip install "cs-dmri[dipy]"    # with dipy, for the parity tests and dipy interop examples
```

## Quality control

```python
import cs_dmri as cs

dwi = cs.DWI.from_files("sub-01_dwi.nii.gz", "sub-01_dwi.bval", "sub-01_dwi.bvec",
                        mask="sub-01_brain_mask.nii.gz")
qc = dwi.qc()
print(qc)
# NDC 0.9733 (masked 0.9260)
# DWI contrast ratio 1.1164 (masked 1.4813, good)
# outlier slices 0
# fixel coherence 0.6912

row = qc.to_dict(prefix="t1_")                      # one TSV row
sidecar = cs.QCReport.column_descriptions("t1_")    # BIDS data dictionary for that TSV
```

| Column | Meaning |
|---|---|
| `ndc`, `ndc_masked` | Neighboring DWI correlation over all voxels / inside the mask. Below 0.4 flags a low-quality image. |
| `dwi_contrast_ratio`, `dwi_contrast_ratio_masked` | Neighbour correlation ÷ correlation with the most nearly perpendicular volume. Below 1.1 poor, 1.1–1.3 fair, above 1.3 good. |
| `n_outlier_slices` | Slices that don't lie between their two adjacent slices in the same volume (`qc.flagged_slices` lists them). |
| `fixel_coherence` | FA-weighted fraction of voxels whose principal direction continues coherently into a neighbour. |
| `dimension_*`, `voxel_size_*`, `max_b`, `n_dwi_volumes`, `n_b0_volumes` | Header and gradient-table facts. |

`column_descriptions()` gives the full definitions and, for renamed columns,
which DSI Studio column (as named in qsiprep) each replaces. The metrics are
defined on purpose, not as DSI Studio clones: NDC doesn't depend on volume
order, masks come from the caller, and outlier slices look only within a
volume. See the main README for the reasoning and measurements.

The same metrics on plain arrays: `cs.qc.neighboring_dwi_correlation`,
`cs.qc.dwi_contrast_ratio`, `cs.qc.outlier_slices`, `cs.qc.fixel_coherence`,
`cs.qc.assess`.

## Models

dipy-style: `Model(gtab, **settings).fit(dwi)` returns a fit with array
attributes. A `DWI` caches work several models share, such as its RESTORE
fit (`dwi.tensor`), which QC coherence and response estimation both reuse.

```python
tensor = cs.RestoreModel(dwi.gtab).fit(dwi)            # .fa .md .s0 .outlier_fraction .tensor .principal_dir

shore = cs.ShoreModel(dwi.gtab, radial_order=6).fit(dwi)   # L1 / L2-anchored by default
shore.r2, shore.sparsity, shore.odf_sh(), shore.predict(other_gtab), shore.microstructure()
shore.to_odx("sub-01_shore.odx")
shore.save("sub-01_coeffs.nii.gz")                     # same layout as cs-fit; cs-odf can read it

responses = dwi.estimate_responses()                   # Dhollander, reusing dwi.tensor
ss3t = cs.SS3TModel(dwi.gtab, responses).fit(dwi).mtnormalise()
ss3t.to_odx("sub-01_ss3t.odx")
fit = cs.ss3t_pipeline(dwi)                            # all of the above, = cs-ss3t-full
```

Every heavy call takes `n_threads=` and releases the GIL.

## Spatial conventions

- **Gradients** are taken as given, in the image's voxel-axis frame (FSL and
  dipy's convention). dipy `GradientTable`s are accepted anywhere a gradient
  table is.
- **SH outputs** (SHORE ODFs, SS3T WM FODs) are computed in world RAS when the
  data have an affine (`bvec_frame="auto"`), as the CLI does. ODX files and
  saved coefficients then mean the same thing as `cs-fit` / `cs-ss3t` output.
  `bvec_frame="image"` keeps dipy's convention.
- **nibabel handles all NIfTI I/O.** `DWI` keeps the source header, and every
  NIfTI written for the series (`dwi.to_image(arr)`, `ShoreFit.save`) inherits
  its qform/sform codes and units. Masks must be on the DWI's grid; otherwise
  the error says what differs. qform/sform disagreement, header zooms that
  don't match the affine, and oblique grids raise `cs.SpatialWarning` and are
  listed in `dwi.warnings` and `qc.warnings`.

## Development

```bash
cd python
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest numpy nibabel dipy
maturin develop --release
pytest tests          # test_cli_parity.py also runs when ../qsiprep_testing and the release binaries exist
```

## Licence

`(MIT OR Apache-2.0) AND MPL-2.0 AND BSD-3-Clause`: MIT or Apache-2.0 except
two MRtrix3-derived Rust modules (MPL-2.0) and the q-space QC search adapted
from dipy (BSD-3). See the repository's `LICENSE-*` files.
