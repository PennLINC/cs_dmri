# Changes

## 0.2.0

- 3D-SHORE fitting stops with an error on data with a single non-zero b-value
  shell, whose radial decay it cannot determine. `allow_single_shell` /
  `--allow-single-shell` fits such data for orientation information only;
  propagator-derived scalars are then not computed. Underdetermined fits
  without L1 regularization are flagged.
- New QC columns: `fixel_chain_length`, the FA-weighted mean length of
  fiber chains formed by the principal directions, and `gradient_table_ratio`,
  which checks the 24 axis permutations and flips of the gradient table. A
  permuted or flipped table is reported by name.
- The QC data dictionary uses only BIDS-defined keys; the corresponding
  DSI Studio column is named in each description instead of a `Replaces` key.

## 0.1.0

First release.

- Image-quality metrics: neighboring DWI correlation, DWI contrast ratio,
  outlier slices and fixel coherence (`cs-qc`, `cs_dmri.qc`).
- 3D-SHORE fitting with L1, L2 and non-negative-ODF estimators; ODFs, peaks,
  propagator-derived scalars and signal synthesis.
- RESTORE diffusion tensor fitting.
- Three-tissue response estimation, SS3T-CSD and multi-tissue intensity
  normalization.
- Python interface.
- Export of orientation results to ODX, DSI Studio, dipy PAM5 and MRtrix3
  formats.
