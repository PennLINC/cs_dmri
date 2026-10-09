# Conventions

## Gradient tables

Gradient directions are used as supplied, in the voxel-axis frame of the image
(the FSL and dipy convention). Volumes with a $b$-value at or below the b=0
threshold (default 50 s/mm²) are treated as b=0. A
{class}`~cs_dmri.gradients.GradientTable` may be built from arrays, from FSL
`.bval`/`.bvec` files or from a dipy gradient table; dipy gradient tables are
also accepted directly wherever a gradient table is expected.

## Orientation of spherical-harmonic outputs

Orientation-dependent outputs (SHORE ODFs and SS3T white-matter FODs) are
computed in the world (RAS) frame when the affine of the image is known: the
gradient directions are rotated by the rotational part of the affine before
fitting. ODX files and saved coefficients are therefore in the frame expected by
ODF viewers and by MRtrix3. In Python this is the default
(`bvec_frame="auto"`); `bvec_frame="image"` keeps the voxel-axis frame. On the
command line `--no-bvec-rotation` has the same effect. The frame used is
recorded in the coefficient sidecar.

Tensor outputs (principal direction) are expressed in the frame of the
gradient table supplied to the model.

## NIfTI input and output

In Python, NIfTI files are read and written with nibabel.
{class}`~cs_dmri.dwi.DWI` keeps the header of the source image, and images
derived from it ({meth}`~cs_dmri.dwi.DWI.to_image`,
{meth}`~cs_dmri.shore.ShoreFit.save`) inherit its qform and sform codes and
units.

When a series is loaded, the following are checked:

- the series is four-dimensional;
- a mask given as an image lies on the same voxel grid (shape and affine) as
  the series; otherwise an error describes the difference;
- the qform and sform agree, the voxel sizes in the header agree with the
  affine, and the grid is not oblique.

Conditions in the last group produce a {class}`~cs_dmri.SpatialWarning` and are
listed in `DWI.warnings` and in the QC report.

## Output formats

Orientation results are stored in ODX, in canonical RAS+ voxel order with SH
coefficients in the MRtrix3 (Tournier) real basis. Export to DSI Studio, dipy
and MRtrix3 formats, and the conventions of each, is described in
{doc}`export`.

## Output files on the command line

Outputs are written to a temporary file and moved into place when complete, so
an interrupted run does not leave partial files. Existing outputs are not
replaced unless `--overwrite` is given. Each tool can record a provenance block
(version, build, threads and run time); `--provenance full` additionally records
the command line, host name and start time, and `--provenance none` omits it.
