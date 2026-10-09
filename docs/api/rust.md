# Rust library

The command-line tools and the Python package are built on the `cs_dmri` Rust
crate, which can also be used directly. Its API documentation is generated with

```bash
cargo doc --no-deps --open
```

The main modules are:

| Module | Contents |
|---|---|
| `qc` | Image-quality metrics and the QC table definition |
| `basis`, `fit`, `solver` | 3D-SHORE basis, whole-volume fitting and the FISTA, Tikhonov and active-set solvers |
| `odf`, `scalars`, `synth` | ODF projection, propagator-derived scalars and signal synthesis |
| `dti` | RESTORE tensor fitting |
| `multitissue` | Response estimation, SS3T-CSD, intensity normalization and the combined pipeline |
| `io` | NIfTI, coefficient-sidecar and ODX input and output |
| `qspace` | Gradient tables and frame conversions |
