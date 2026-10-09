# Exporting results

Orientation results, the SHORE ODFs and the SS3T white-matter FODs, are stored
natively in the [ODX](https://github.com/PennLINC/odx-rs) format. From Python
they can also be written for DSI Studio, dipy and MRtrix3, using the format
converters of odx-rs.

## Python

{meth}`ShoreFit.export <cs_dmri.shore.ShoreFit.export>` and
{meth}`SS3TFit.export <cs_dmri.multitissue.SS3TFit.export>` write a fit to one
file or directory. The format is inferred from the extension:

```python
fit = cs.ss3t_pipeline(dwi)

fit.export("ss3t.odx")          # ODX archive
fit.export("ss3t.fz")           # DSI Studio
fit.export("ss3t.fib.gz")       # DSI Studio, older format
fit.export("ss3t.pam5")         # dipy
fit.export("wm_fod.mif")        # MRtrix3 SH image (.mif, .mif.gz, .nii, .nii.gz)
```

Directory outputs have no extension to infer from, so the format must be named:

```python
fit.export("ss3t_odx", format="odx-directory")
fit.export("fixels", format="mrtrix-fixel-dir")                         # NIfTI files
fit.export("fixels_mif", format="mrtrix-fixel-dir", fixel_container="mif")
```

An existing output is replaced only with `overwrite=True`. Outputs are written
to a temporary path and moved into place when complete. `to_odx(path)` is
equivalent to `export(path, format="odx-archive")`.

{meth}`ShoreFit.export <cs_dmri.shore.ShoreFit.export>` also accepts the
options of `cs-odf`: the SH order of the ODF (`lmax`), peak extraction
(`peaks`, `npeaks`, `peak_relative_threshold`, `peak_min_separation_deg`), and
whether to include microstructure scalars and fit diagnostics.

Exports are in world coordinates, so the fit must have been computed in the
world frame (`bvec_frame="world"`, the default when the affine is known; see
{doc}`conventions`).

## Formats

| `format` | Extension | Contents |
|---|---|---|
| `odx-archive` | `.odx` | Everything: SH coefficients, peaks, per-fixel and per-voxel values, metadata |
| `odx-directory` | (directory) | The same, as an uncompressed directory |
| `dsistudio-fz` | `.fz` | ODFs sampled from the SH coefficients, peaks, per-voxel values and the mask |
| `dsistudio-fibgz` | `.fib.gz` | The same, in DSI Studio's older format |
| `dipy-pam5` | `.pam5` | SH coefficients, peaks and per-voxel values |
| `mrtrix-sh-image` | `.mif`, `.mif.gz`, `.nii`, `.nii.gz` | SH coefficients |
| `mrtrix-fixel-dir` | (directory) | Peak directions and per-fixel values |

The list is also available as {data}`cs_dmri.EXPORT_FORMATS`. ODX is the only
format that holds all outputs of a fit; export to ODX as well when the full
result should be kept.

Coordinate conventions:

- ODX files are in canonical RAS+ voxel order, and SH coefficients use the
  MRtrix3 (Tournier) real basis.
- MRtrix3 SH images and fixel directions are in the scanner (world) frame, as
  MRtrix3 expects.
- PAM5 files store SH coefficients in dipy's default basis (`descoteaux07` with
  `legacy=True`) and peak directions in the voxel frame of the file's affine,
  as dipy expects. Results evaluated with dipy's default SH settings therefore
  agree with the MRtrix3 export.

## Command line

The command-line tools write ODX (`cs-odf`, `cs-fit --odx-output`,
`cs-ss3t-full --odx`). Other formats are produced from the ODX file with the
`odx` tool of odx-rs:

```bash
odx convert ss3t.odx ss3t.fz
odx convert ss3t.odx fixels --output-format mrtrix-fixel-dir
```
