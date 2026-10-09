# cs_dmri

cs_dmri is a library and set of command-line tools for the analysis of
diffusion MRI data. It is written in Rust and has a Python interface. It
provides:

- **Image quality control**: neighboring DWI correlation, a DWI contrast ratio,
  within-volume outlier-slice detection and a fixel-coherence index.
- **3D-SHORE reconstruction** with L1-regularized (compressed-sensing), L2 or
  non-negativity-constrained fitting, from which orientation distribution
  functions, fiber peaks, propagator-derived scalars and synthesized signals
  are computed.
- **Diffusion tensor estimation** with the RESTORE algorithm.
- **Single-shell three-tissue constrained spherical deconvolution** (SS3T-CSD),
  including unsupervised response-function estimation and multi-tissue
  intensity normalization.

Orientation outputs are written in the [ODX](https://github.com/PennLINC/odx-rs)
format. The command-line tools and the Python interface share one
implementation and produce identical results.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user/qc
user/shore
user/dti
user/multitissue
user/conventions
```

```{toctree}
:maxdepth: 1
:caption: Reference

cli/index
api/python
api/rust
references
```

```{toctree}
:maxdepth: 1
:caption: Project

development
license
changes
```
