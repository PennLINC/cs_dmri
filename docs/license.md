# License

cs_dmri is distributed under the terms of either the MIT license or the Apache
License, Version 2.0, at the user's option, with the following exceptions.

- Two source files are derived from MRtrix3 and are distributed under the
  Mozilla Public License 2.0: `src/multitissue/dhollander.rs` (the voxel
  selection of `dwi2response dhollander`) and `src/multitissue/mtnormalise.rs`
  (the algorithm of `mtnormalise`). MRtrix3 is copyright the MRtrix3
  contributors. The Mozilla Public License applies to these files only; the
  files that use them are not affected.
- Functions in `src/qc.rs` adapted from dipy (the q-space neighbor and
  contrast-volume searches) are copyright the dipy developers and distributed
  under the BSD 3-Clause license.

The combined license expression is `(MIT OR Apache-2.0) AND MPL-2.0 AND
BSD-3-Clause`. The full texts are in the repository: `LICENSE-MIT`,
`LICENSE-APACHE`, `LICENSE-MRTRIX` and `LICENSE-DIPY`.
