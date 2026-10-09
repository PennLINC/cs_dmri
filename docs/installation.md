# Installation

## Python package

```bash
pip install cs_dmri
```

Wheels are provided for Linux (x86-64 and ARM64), macOS (Intel and Apple
silicon) and Windows (x86-64), for CPython 3.9 and later. The package depends on
NumPy and nibabel. Installing the optional `dipy` extra (`pip install
"cs_dmri[dipy]"`) is only needed to run the parts of the test suite that compare
against dipy.

## Command-line tools

The command-line tools are built from source with a Rust toolchain
(<https://rustup.rs>):

```bash
git clone https://github.com/PennLINC/cs_dmri
cd cs_dmri
cargo build --release
```

The executables are placed in `target/release/`:

| Tool | Purpose |
|---|---|
| `cs-qc` | Image-quality metrics |
| `cs-fit` | 3D-SHORE fit |
| `cs-odf` | ODFs, peaks and scalars from SHORE coefficients (ODX output) |
| `cs-synth` | Signal synthesis from SHORE coefficients |
| `cs-dti` | RESTORE diffusion tensor fit |
| `cs-response` | Three-tissue response-function estimation |
| `cs-ss3t` | SS3T-CSD with given responses |
| `cs-mtnorm` | Multi-tissue intensity normalization |
| `cs-ss3t-full` | Response estimation, SS3T-CSD and normalization in one step |

Building requires a C compiler and CMake 3.26 or later, used by a dependency of
the ODX library.

## Building the Python package from source

```bash
cd python
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release
```

## Parallelism

All tools and Python functions are multithreaded. On the command line,
`--threads N` sets the number of threads; without it, `SLURM_CPUS_PER_TASK`,
then `RAYON_NUM_THREADS`, then the number of logical CPUs is used. In Python,
functions that perform substantial computation accept `n_threads=` and release
the global interpreter lock while they run.
