# cs_dmri

Compressed-sensing reconstruction of diffusion MRI data with the 3D-SHORE basis.

## What it does

`cs_dmri` is a Rust crate plus nine command-line tools that:

1. **Fit** raw 4D DWI volumes to a regularized 3D-SHORE coefficient field (`cs-fit`),
2. **Project** those coefficients to Orientation Distribution Functions and fixels in the [ODX](../odx-rs) format (`cs-odf`),
3. **Synthesize** new DWI volumes from a fitted coefficient field under any target gradient table (`cs-synth`),
4. **Decompose** a single-shell DWI into a white-matter FOD plus isotropic gray-matter and CSF compartments via the SS3T-CSD algorithm (`cs-ss3t`),
5. **Estimate** WM/GM/CSF tissue response functions from a single-shell DWI without external tools (`cs-response`),
6. **Fit** a robust diffusion tensor (RESTORE algorithm) for FA/MD/outlier maps on clinical-quality DWI (`cs-dti`),
7. **Normalise** SS3T tissue maps via a polynomial bias-field correction (`cs-mtnorm`),
8. **Run the entire SS3T pipeline** in one command — DWI to normalised tissues with no MRtrix at runtime (`cs-ss3t-full`),
9. **Score image quality** — NDC, DWI contrast ratio, outlier slices and fixel coherence (`cs-qc`).

The first three form the SHORE-basis compressed-sensing pipeline; `cs-fit`'s output is byte-for-byte compatible with qsirecon's BrainSuite SHORE pipeline (same basis ordering, same default regularization), but adds per-voxel BIC-driven α selection along an L1 regularization path — the "compressed sensing" path that gives the project its name. Peak extraction is delegated to `odx-rs::peak_finder`, which uses MRtrix-style sub-vertex Newton refinement of seeds taken from a discrete sphere search.

`cs-ss3t` is a separate, native-Rust port of the SS3T-CSD algorithm (Dhollander & Connelly, ISMRM 2016) that replaces the MRtrix3Tissue fork. The inner constrained-least-squares solve uses the Goldfarb-Idnani 1983 active-set method; per-voxel agreement against MRtrix3Tissue's reference output is r > 0.99 (see [tests/ss3t_parity.rs](tests/ss3t_parity.rs)).

`cs-response`, `cs-dti`, `cs-mtnorm`, and `cs-ss3t-full` close the loop on the SS3T pipeline: cs_dmri runs the entire single-shell three-tissue workflow end-to-end without any external MRtrix tooling. The whole pipeline is a single command (`cs-ss3t-full`); the individual stages are exposed as separate tools when more control is needed. `cs-dti` is also useful standalone for clinical FA/MD maps with motion-outlier QC channels.

## Build

```bash
cargo build --release
```

Binaries land in `target/release/{cs-fit,cs-odf,cs-synth,cs-ss3t,cs-response,cs-dti,cs-mtnorm,cs-ss3t-full,cs-qc}`.


## Running on HPC

The CLIs are designed to drop unmodified into a SLURM submit script.

- **Threads**: `--threads N` overrides everything. With no flag the resolution
  order is `$SLURM_CPUS_PER_TASK` → `$RAYON_NUM_THREADS` → all logical CPUs.
  Each bin logs the chosen value and source on startup
  (`[cs-fit] threads=16 source=SLURM_CPUS_PER_TASK`), so you can verify in the
  job log that it didn't grab the whole node.
- **Progress heartbeat**: long per-voxel loops emit a one-line status to
  stderr every `--progress-interval-secs` seconds (default 30):
  `[cs-fit] 142000/250000 (56.8%) elapsed=00:02:14 eta=00:01:42`. Plain lines,
  non-TTY safe. Suppress with `--quiet`.
- **Atomic output**: every output file (coefficient NIfTI + sidecar pair, ODX
  archive or directory, sibling diagnostics, microstructure NIfTIs, synth
  output) lands via temp + rename. A SIGKILL'd job leaves either the final
  file fully written or no file at all (modulo a tiny window between paired
  renames). The CLIs `mkdir -p` the output's parent dir.
- **Overwrite guard**: `--overwrite` is required to clobber an existing
  output. Default is to refuse with a clear error — protects against
  accidentally re-running over a finished output.
- **Provenance**: each tool writes a small `provenance` block with version,
  git SHA, build timestamp, threads used, and runtime in seconds. Default
  mode (`--provenance minimal`) carries no PHI surface — no hostname, no
  argv, no wall-clock start time. Use `--provenance full` only when input
  paths and host metadata are safe to retain alongside the output.
  `--provenance none` skips it entirely.
  - `cs-fit` embeds it in the existing JSON sidecar.
  - `cs-synth` writes a new sibling `<output>.json`.
  - `cs-odf` attaches it as an ODX extra value (`cs_dmri_provenance`).

## Recommended configuration

The recommended default for L1 fitting is the **L2-residual-anchored α
selector** (`--alpha-mode l2-anchored`). Per voxel it picks the largest
α whose RSS is within `(1 + slack) · RSS_L2`, where the L2 reference is a
Tikhonov fit using `--lambda-n` / `--lambda-l`. This caps fit looseness
against L2 directly and avoids the two failure modes of plain
`--alpha-mode path-bic`:

- **High-b CS-DSI / 10k+ shells**: BIC's degree-of-freedom penalty
  trades fit for sparsity too aggressively; median RSS lands ~45 % above
  L2 in half the brain. The slack constraint pulls those voxels back to
  L2-quality fit while keeping L1 sparsity.
- **Infant / low-SNR / low-FA**: BIC under-sparsifies — picks an α so
  small the L1 prior buys nothing (median 26 of 29 nonzero coefs on real
  infant data). The slack rule lets the selector pick a much larger
  meaningful α (median 5–18 of 29 nonzero) at the cost of only a few
  percent of fit looseness.

```
--reg l1 --alpha-mode l2-anchored --slack 0.05 --radial-order 4
```

The CLI defaults pick `--alpha-mode l2-anchored`, `--slack 0.05`, and
`--path-eps 1e-4` automatically, so a minimal invocation is just
`--reg l1 --radial-order 4`. (The default `--radial-order` remains 6 for
qsirecon parity, so set it to 4 explicitly for the focused config.)

`--alpha-mode path-bic` remains available for byte-for-byte parity with
the historical qsirecon BrainSuite-SHORE pipeline, and `--alpha-mode
fixed` / `alpha-ratio` are the cheap escape hatches. See the empirical
sweeps in [scripts/bench_fit_options.py](scripts/bench_fit_options.py)
and the synthetic-harness comparison in
[../specs/baby_like.yaml](../specs/baby_like.yaml) for the trade-offs.

---

## Python

`pip install cs_dmri` (or `maturin develop --release` inside `python/`) gives a
dipy-style Python API over everything below, with results identical to the
command-line tools:

```python
import cs_dmri as cs

dwi = cs.DWI.from_files("dwi.nii.gz", "dwi.bval", "dwi.bvec", mask="mask.nii.gz")
print(dwi.qc())                                      # NDC, contrast ratio, outlier slices, coherence
fit = cs.ShoreModel(dwi.gtab).fit(dwi)               # cs-fit
ss3t = cs.ss3t_pipeline(dwi)                         # cs-ss3t-full
```

See [python/README.md](python/README.md).

## Quality control (`cs-qc`)

`cs-qc` scores a DWI series and writes the results as JSON and/or a one-row TSV.
The TSV gets a BIDS-style JSON data dictionary describing every column.

```bash
cs-qc --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask brain_mask.nii.gz \
      --output-tsv sub-01_desc-image_qc.tsv --prefix t1_ --output-json sub-01_qc.json
```

| Column | Definition | Replaces (DSI Studio, as named in qsiprep) |
|---|---|---|
| `dimension_{x,y,z}`, `voxel_size_{x,y,z}`, `max_b` | Header and gradient-table facts | same names |
| `n_dwi_volumes`, `n_b0_volumes` | Volumes above / at or below the b=0 threshold | `num_directions` (counted volumes, not directions) |
| `ndc`, `ndc_masked` | Neighboring DWI correlation over all voxels / inside the mask | `neighbor_corr`, `masked_neighbor_corr` |
| `dwi_contrast_ratio`, `dwi_contrast_ratio_masked` | Neighbour correlation ÷ perpendicular-volume correlation | `dwi_contrast` |
| `n_outlier_slices` | Slices that don't lie between their two adjacent slices in the same volume | `num_bad_slices` |
| `fixel_coherence` | FA-weighted share of voxels whose principal direction continues coherently (0–1) | `coherence_index` |

The definitions differ from DSI Studio's on purpose, which is why the renamed
columns have new names:

- **NDC averages over every b>0 volume.** DSI Studio counts each neighbour pair
  once, keyed on volume index, which makes its NDC depend on acquisition order:
  0.970–0.977 under random reorderings of one HASC92 series, where this
  definition gives 0.9733 every time. Otherwise it is dipy's definition:
  repeated acquisitions may pair with each other. In merged AP+PA series each
  volume pairs with its twin from the other run, which measured slightly lower
  NDC than excluding twins (0.786 vs 0.803), so pairing them adds a cross-run
  consistency check rather than inflating the score. The b=0 threshold is a
  parameter (`--b0-threshold`, default 50).
- **Masks come from the caller.** DSI Studio uses its own internal mask, and its
  construction changed between versions. On one series the contrast ratio is
  1.12 unmasked, 1.48 inside a SynthStrip mask, 1.74 with DSI Studio 2024's mask
  and 1.86 with DSI Studio 2026's. Without `--mask`, `cs-qc` falls back to
  "mean b=0 above 1% of its maximum" and records `mask_source: auto-b0`.
- **Outlier slices look only within a volume.** After in-plane smoothing
  (σ = 2 voxels), a slice is flagged when its mean absolute deviation from the
  average of its two adjacent slices exceeds 2.5 × half their mean absolute
  difference. The maximum over about 64,000 slices of eight raw and
  preprocessed qsiprep test series is 2.39. Injected dropout to 50% signal is
  caught about 90% of the time, and to 70% about two times in three.
- **Fixel coherence** uses odx-rs's primary coherence on the RESTORE principal
  direction, weighted and thresholded by FA (lowest 10% dropped, 15°). It is a
  0–1 fraction, unlike DSI Studio's unbounded index from a GQI fib.

Every fitting tool (`cs-fit`, `cs-dti`, `cs-response`, `cs-ss3t`,
`cs-ss3t-full`) also prints the masked NDC and contrast ratio of its input, with
a `WARNING:` line when NDC < 0.4 (Yeh et al. 2019) or contrast < 1.1. The NDC
neighbour search and the contrast-volume search are adapted from dipy (BSD-3;
the contrast search from dipy PR #4224, a port of DSI Studio's definition made
with Fang-Cheng Yeh's permission). The library API is `cs_dmri::qc`.

## `cs-fit` — fit DWI to SHORE coefficients

Loads a 4D DWI plus its FSL gradient files (`.bval` / `.bvec`) and an optional brain mask, fits a regularized 3D-SHORE coefficient field per in-mask voxel, and writes a 4D NIfTI of coefficients plus a JSON sidecar capturing the basis parameters, deltas, solver settings, and the bvec frame used.

**Inputs**
- 4D DWI NIfTI (`--dwi`)
- FSL `.bval` and `.bvec` (`--bval`, `--bvec`)
- Optional brain mask (`--mask`); auto-thresholded from the b0 mean if omitted

**Outputs**
- 4D coefficient NIfTI (`--output`); the 4th axis indexes SHORE basis functions
- JSON sidecar at the same path with `.json` extension
- With `--diagnostics`: sibling NIfTIs `<stem>_r2.nii.gz`, `_rmse.nii.gz`, `_alpha.nii.gz`, `_bic.nii.gz`, `_sparsity.nii.gz`

**Key flags**

| Flag | Default | Purpose |
| --- | --- | --- |
| `--reg {l1,l2}` | `l1` | L1: FISTA sparse fit (CS path). L2: closed-form Tikhonov, fast smoke test. |
| `--alpha-mode {fixed,alpha-ratio,path-bic,l2-anchored}` | `l2-anchored` | L1 α-selection. `l2-anchored` is recommended (caps fit looseness against a Tikhonov reference; works on high-b CS-DSI + infant scans where `path-bic` mis-fires). `path-bic` is the historical qsirecon-parity choice. |
| `--slack` | `0.05` | L2-residual slack for `--alpha-mode=l2-anchored`. Per voxel the chosen α has RSS ≤ (1+slack)·RSS_L2. Tighter (0.02) matches L2's ODF cleanliness; looser (0.10) preserves more sparsity. |
| `--radial-order` | `6` | SHORE radial order. **Set this to `4` for the recommended config**; the default of 6 is for qsirecon parity. |
| `--zeta` | `700.0` | SHORE radial scale parameter. |
| `--alpha` | `1.0` | Fixed α (only when `--alpha-mode=fixed`). |
| `--alpha-ratio` | `1e-3` | α / α_max ratio (only when `--alpha-mode=alpha-ratio`). Typical 1e-3 .. 1e-2. |
| `--path-n-alphas` | `20` | Grid size along the L1 regularization path (`path-bic` / `l2-anchored`). |
| `--path-eps` | `1e-3` (path-bic) / `1e-4` (l2-anchored) | α_min / α_max for the path. The wider default for `l2-anchored` lets the slack constraint bind on every voxel; without it ~17% of high-b voxels fall back to argmin RSS. Override explicitly for parity sweeps. |
| `--max-iter` | `1000` | FISTA max iterations per fit. |
| `--tol` | `1e-6` | FISTA convergence tolerance. |
| `--non-negative` | off | Enforce non-negative coefficients. |
| `--lambda-n`, `--lambda-l` | `1e-8` | L2-only: radial / angular regularization weights. |
| `--big-delta`, `--small-delta` | estimated | Diffusion timing in seconds. Estimated via TORTOISE's max-bval heuristic if either is missing. |
| `--gmax` | `0.08` | Maximum gradient amplitude (T/m); only used when deltas are estimated. |
| `--diagnostics` | off | Also write per-voxel R², RMSE, α, BIC, and sparsity NIfTIs next to the coefficients. |
| `--odx-output PATH` | off | Also project the fit to a Tournier ODX (one-step alternative to `cs-fit … && cs-odf …`). Uses `cs-odf`'s defaults — DSI-Studio ODF8 peaks, brain-wide ODF normalization, AP DPV, lmax = largest even ≤ radial_order, and the full microstructure suite (RTOP/RTAP/RTPP/MSD/QIV/NG with default `K=10` fit-failure rejection and `[0, p99]` colormap hints in the ODX header). The coefficient NIfTI is still written; tune peak settings, alternative lmax, or microstructure rejection threshold by re-running `cs-odf` against it. |
| `--odx-directory` | off | Emit `--odx-output` as a directory tree instead of a `.odx` archive. |
| `--no-bvec-rotation` | off | Keep bvecs in image-axis frame. **Set this only for byte-for-byte qsirecon parity** — the rotated default is what TRXViz and most ODF viewers expect. |
| `--threads` | auto | Cap rayon's worker pool. Default: `$SLURM_CPUS_PER_TASK` → `$RAYON_NUM_THREADS` → all logical CPUs. |
| `--overwrite` | off | Allow clobbering existing outputs. Default: refuse. |
| `--quiet` | off | Suppress progress heartbeat and per-step summary lines. |
| `--progress-interval-secs` | `30` | Seconds between heartbeat lines during the per-voxel fit. |
| `--provenance {minimal,full,none}` | `minimal` | Provenance written into the sidecar JSON. `minimal` carries no PHI surface; `full` adds argv, hostname, and wall-clock start. |

`cs-fit --help` lists the full set.

**Example: fit on the HASC55 sub-sampled scheme for sub-01**

```bash
cs-fit \
    --dwi  /data/qsiprep/sub-01/ses-1/dwi/sub-01_ses-1_acq-HASC55_run-01_space-T1w_desc-preproc_dwi.nii.gz \
    --bval /data/qsiprep/sub-01/ses-1/dwi/sub-01_ses-1_acq-HASC55_run-01_space-T1w_desc-preproc_dwi.bval \
    --bvec /data/qsiprep/sub-01/ses-1/dwi/sub-01_ses-1_acq-HASC55_run-01_space-T1w_desc-preproc_dwi.bvec \
    --mask /data/qsiprep/sub-01/ses-1/dwi/sub-01_ses-1_acq-HASC55_run-01_space-T1w_desc-brain_mask.nii.gz \
    --reg l1 --alpha-mode l2-anchored --radial-order 4 \
    --diagnostics \
    --output ~/cs-bench-csdsi/focused/sub-01_HASC55_coeffs.nii.gz
```

This produces `sub-01_HASC55_coeffs.nii.gz`, `sub-01_HASC55_coeffs.json`, plus `_r2`, `_rmse`, `_alpha`, `_bic`, `_sparsity` siblings.

To skip the separate `cs-odf` step when you only need the default-settings ODX, append `--odx-output ~/cs-bench-csdsi/focused/sub-01_HASC55.odx` to the same call. The coefficient NIfTI is still written, so you can re-run `cs-odf` later for custom peak settings, alternative lmax, or microstructure scalars without redoing the fit.

---

## `cs-odf` — project coefficients to ODF SH and fixels (ODX output)

Reads a coefficient NIfTI (and its sidecar) from `cs-fit`, applies the analytical SHORE → Tournier-ordered SH transform, samples the ODF on the seed sphere, refines each accepted local maximum to a sub-vertex peak direction in continuous SH, and writes everything into a single `.odx` archive.

**Inputs**
- Coefficient NIfTI from `cs-fit` (`--coeffs`); the matching `.json` sidecar is read automatically
- Optional brain mask (`--mask`); defaults to "any voxel with a nonzero coefficient"

**Outputs**
- A `.odx` file (or directory, with `--directory`) containing:
  - `sh/coefficients` — Tournier-ordered SH coefficients per masked voxel
  - Per-voxel fixels: direction + amplitude + QA + dispersion per peak (dispersion = FMLS lobe integral ÷ peak amplitude, MRtrix `fod2fixel -disp`)
  - GFA per voxel
  - Anisotropic-power DPV (default; Dell'Acqua 2014) for slice backgrounds
  - Any sibling diagnostic NIfTIs from `cs-fit --diagnostics`, copied in as DPVs
  - Per-voxel **RTOP**, **RTAP**, **RTPP**, **MSD**, **QIV**, **NG** as DPVs (default on; pass `--no-microstructure` to skip). Voxels with implausibly large values (default `> 10× p99`) are NaN'd as fit failures rather than embedded as data — RTOP/RTAP can be 100s–1000s× the brain median in degenerate fits where E(0) collapses, and those numbers shouldn't be trusted. The threshold and per-scalar reject counts go in the header extra `cs_dmri_microstructure_outliers` for traceability; per-scalar median / p95 / p99 / max + a suggested `[0, p99]` colormap range go in `cs_dmri_microstructure_display` so a viewer can auto-range against the cleaned distribution.
  - With `--microstructure-nifti` (alias: `--microstructure`): the same six scalars as sibling NIfTIs (`<output_stem>_rtop.nii.gz`, …) for FSL-style downstream pipelines.

**Key flags**

| Flag | Default | Purpose |
| --- | --- | --- |
| `--lmax` | largest even ≤ radial_order | Maximum even SH order for the ODF. |
| `--name` | `coefficients` | Field name under `sh/`. The default is what trxviz and `odx-rs`'s mrtrix loader expect. |
| `--directory` | off | Emit a directory tree instead of a zipped archive. |
| `--no-anisotropic-power` | off | Skip the AP DPV. Disable only if you supply your own scalar background. |
| `--ap-norm-factor` | `1e-5` | Log-shift for AP; matches dipy's default. |
| `--no-global-normalize` | off | Skip DSI-Studio-style global QA normalization. The default normalizes every voxel's SH by the brain-wide max of `max(ODF) − min(ODF)` so glyphs render at consistent scale. |
| `--no-diagnostic-dpvs` | off | Don't auto-load sibling `_r2`, `_rmse`, `_alpha`, `_bic`, `_sparsity` NIfTIs. |
| `--no-peaks` | off | Skip fixel extraction; emit SH-only ODX. |
| `--peak-npeaks` | `5` | Maximum peaks per voxel. |
| `--peak-relative-threshold` | `0.5` | Drop peaks below this fraction of the voxel's strongest peak. |
| `--peak-min-separation-deg` | `25.0` | Minimum angular separation between accepted peaks. |
| `--no-microstructure` | off | Skip computing RTOP/RTAP/RTPP/MSD/QIV/NG. By default these are computed per voxel (cheap — sub-second on a typical brain), fit-failure outliers are rejected to NaN, and the cleaned scalars are embedded as DPVs alongside reject metadata (`cs_dmri_microstructure_outliers`) and display hints (`cs_dmri_microstructure_display`). RTAP/RTPP use the first peak direction per voxel; voxels without a detected peak come back as NaN. See [scripts/microstructure_math.md](scripts/microstructure_math.md) for derivations. |
| `--microstructure-nifti` | off | Also write each scalar to a sibling NIfTI of the ODX (`<output_stem>_rtop.nii.gz`, …). Aliased to the legacy `--microstructure`. |
| `--microstructure-outlier-factor <K>` | `10.0` | Voxels whose scalar exceeds `K × p99` (per scalar, brain-wide) are NaN'd as fit failures. With `p99/median ≈ 8`, K=10 corresponds to "more than ~80× the median is implausible." Tune higher to keep more, lower to reject more. K=5 trims the marginal upper tail; K=20+ catches only the most catastrophic failures. |
| `--no-microstructure-outlier-rejection` | off | Disable the rejection — every finite scalar value lands in the ODX, including the implausible ones. Use only for debugging fit quality or comparing against legacy outputs. |
| `--scalar-units {um,mm}` | `um` | Length unit for emitted microstructure scalars. `um` matches TORTOISE's `EstimateMAPMRI` output (q in 1/μm → RTOP in /μm³, etc.) and lands values in the familiar [0, ~few] range. `mm` keeps the dipy / cs_dmri internal convention (q in 1/mm → RTOP in /mm³ ~ 10⁵ for brain). Physically equivalent — only the displayed magnitudes differ. |
| `--threads` | auto | Cap rayon's worker pool. Default: `$SLURM_CPUS_PER_TASK` → `$RAYON_NUM_THREADS` → all logical CPUs. |
| `--overwrite` | off | Allow clobbering an existing ODX, output directory, or sibling microstructure NIfTI. |
| `--quiet` | off | Suppress progress heartbeat and per-step summary lines. |
| `--progress-interval-secs` | `30` | Seconds between heartbeat lines during long parallel loops. |
| `--provenance {minimal,full,none}` | `minimal` | Embedded in the ODX as `cs_dmri_provenance`. See "Running on HPC" above for the PHI tradeoffs. |

A note on peaks: the seed sphere is just a starting grid — every accepted seed is then Newton-refined in continuous SH (`SpherePeakFinder::find_peaks_with_sh` in `odx-rs`), so peak directions are sub-vertex and not snapped to a discrete sphere.

**Example: project the fit from above to ODX**

```bash
cs-odf \
    --coeffs ~/cs-bench-csdsi/focused/sub-01_HASC55_coeffs.nii.gz \
    --output ~/cs-bench-csdsi/focused/sub-01_HASC55.odx
```

Open the result in TRXViz to see SH glyphs, fixel arrows, and the anisotropic-power background, with all `cs-fit --diagnostics` maps available as switchable DPV overlays.

---

## `cs-synth` — synthesize a DWI from coefficients

Loads coefficients (and the basis from the sidecar) and an arbitrary target FSL gradient table, then evaluates the SHORE expansion at the target q-space points to produce a 4D DWI. Output is clamped to non-negative.

**Inputs**
- Coefficient NIfTI from `cs-fit` (`--coeffs`)
- Target FSL `.bval` and `.bvec` for the scheme you want to predict onto

**Outputs**
- 4D DWI NIfTI (`--output`) with the spatial grid of the input coefficients and the directional sampling of the target scheme

**Key flags**

| Flag | Default | Purpose |
| --- | --- | --- |
| `--big-delta`, `--small-delta` | from sidecar | Optional overrides. The sidecar value is used unless you set this. |
| `--gmax` | `0.08` | Used only if deltas need to be re-estimated and the sidecar lacks them. |
| `--threads` | auto | Cap rayon's worker pool. Default: `$SLURM_CPUS_PER_TASK` → `$RAYON_NUM_THREADS` → all logical CPUs. |
| `--overwrite` | off | Allow clobbering an existing output NIfTI and sidecar JSON. |
| `--quiet` | off | Suppress progress heartbeat and per-step summary lines. |
| `--progress-interval-secs` | `30` | Seconds between heartbeat lines during synthesis. |
| `--provenance {minimal,full,none}` | `minimal` | Provenance written to a sibling `<output>.json`. `none` skips the sidecar entirely. |

**Example: predict the ABCD scheme from the HASC55 fit**

```bash
cs-synth \
    --coeffs ~/cs-bench-csdsi/focused/sub-01_HASC55_coeffs.nii.gz \
    --bval   /data/qsiprep/sub-01/ses-1/dwi/sub-01_ses-1_acq-ABCD_space-T1w_desc-preproc_dwi.bval \
    --bvec   /data/qsiprep/sub-01/ses-1/dwi/sub-01_ses-1_acq-ABCD_space-T1w_desc-preproc_dwi.bvec \
    --output ~/cs-bench-csdsi/focused/sub-01_HASC55-to-ABCD_predicted.nii.gz
```

The synthesized volume can be compared voxel-by-voxel against the observed `acq-ABCD` DWI to score how well the HASC55 sub-sampled scheme captured the full-DSI signal.

---

## `cs-dti` — Robust diffusion-tensor fit (RESTORE)

A native-Rust implementation of RESTORE (Chang, Jones & Pierpaoli, MRM 53(5), 2005) — the de-facto standard for clinical DTI when motion-induced volume dropouts contaminate a subset of measurements per voxel. Vanilla weighted-least-squares treats those outliers as if they fit log-Gaussian noise; RESTORE detects them via Studentized residuals and downweights them with the Geman-McClure M-estimator, iterating until the weights stabilize.

Outputs FA, MD, S₀, and an outlier-fraction map per voxel; optionally the full tensor (BIDS-style 6 components) and principal eigenvector.

License posture: implemented from the 2005 MRM paper; dipy's BSD-3 `RestoreModel` was consulted for sanity-checking sign/scale conventions; no MRtrix code (`dwi2tensor -method restore`, MPL-2.0) was read.

**Inputs**
- 4D DWI NIfTI (`--dwi`)
- FSL `.bval` / `.bvec` (`--bval`, `--bvec`)
- Optional brain mask (`--mask`); auto-thresholded from b0 mean if omitted

**Outputs**
- `--output-fa` — fractional anisotropy ∈ [0, 1] (3D NIfTI)
- `--output-md` — mean diffusivity (3D, units track the bvals' convention)
- `--output-s0` — signal at b=0 (3D, input intensity units)
- `--output-outlier-fraction` — RESTORE's QC channel, fraction of measurements per voxel rejected as outliers (3D, ∈ [0, 1])
- `--output-tensor PATH` (optional) — 4D NIfTI, 6 components in BIDS lower-triangular order: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
- `--output-principal-dir PATH` (optional) — 4D NIfTI, principal eigenvector (3 channels)
- With `--diagnostics`: sibling NIfTIs `<fa_stem>_iters.nii.gz`, `_converged.nii.gz`

**Key flags**

| Flag | Default | Purpose |
| --- | --- | --- |
| `--max-iter` | `50` | RESTORE reweighting iterations cap. |
| `--tol` | `1e-6` | Convergence threshold (relative change in tensor coefs). |
| `--outlier-threshold` | `0.04` | Geman-McClure weight cutoff for the outlier-fraction map (0.04 ≈ 2σ). Doesn't affect the fit, only the QC channel. |
| `--no-bvec-rotation`, `--threads`, `--overwrite`, `--quiet`, `--progress-interval-secs`, `--provenance` | as elsewhere | (See "Running on HPC" above.) |

**Example**

```bash
cs-dti \
    --dwi  /path/to/dwi.nii.gz \
    --bval /path/to/dwi.bval --bvec /path/to/dwi.bvec \
    --mask /path/to/brain_mask.nii.gz \
    --output-fa  out/fa.nii.gz \
    --output-md  out/md.nii.gz \
    --output-s0  out/s0.nii.gz \
    --output-outlier-fraction out/outliers.nii.gz \
    --output-tensor out/tensor.nii.gz
```

On a 250K-voxel clinical brain volume this finishes in ~7 seconds on 14 cores.

**What it means in practice**

The outlier-fraction map is an underused but powerful QC channel. Healthy clinical data typically lands at <10% outlier fraction in most voxels; high-fraction regions flag motion-corrupted slices, eddy artifacts, or grossly noisy voxels. Use it as a first-pass mask refiner: voxels with >30% outlier fraction are unreliable for downstream tractography or microstructure work.

---

## `cs-response` — WM/GM/CSF response estimation (Dhollander 2016)

Implements the Dhollander 2016 unsupervised three-tissue response estimation algorithm (ISMRM Workshop abstract) using cs_dmri's RESTORE DTI fit for the underlying FA/MD/eigenvalue maps. Outputs three MRtrix-format `.txt` files that drop directly into `cs-ss3t --response-{wm,gm,csf}`.

This is the missing piece for a fully MRtrix-free SS3T pipeline. With `cs-response` you can run **raw DWI → responses → SS3T → mtnormalise** entirely within cs_dmri (the only external tool is MRtrix's `mtnormalise` for the optional final normalization step; we may port this too in the future).

License posture: the per-tissue signal averaging is cs_dmri's own, written from the Dhollander 2016 abstract and informed by dipy BSD-3 idioms. The **voxel selection** is a port of MRtrix's `dwi2response dhollander`, which is MPL-2.0 — see [Third-party licences](#third-party-licences).

Per-voxel WM Pearson r vs MRtrix's reference SS3T pipeline is **0.97** (vs 0.996 when both sides use MRtrix-derived responses), with GM spatial r 0.97 and CSF spatial r 0.998 — within practical tolerance for tractography and microstructure use.

**Inputs**
- 4D DWI NIfTI + bval/bvec/mask (same format as `cs-fit`).

**Outputs**
- `--output-wm` — single-fibre WM response, MRtrix `.txt` format (n_shells × (lmax/2 + 1) zonal SH coefs).
- `--output-gm` — GM response (1 column, lmax = 0).
- `--output-csf` — CSF response (1 column, lmax = 0).
- With `--diagnostics`: per-tissue selection masks (`<wm_stem>_mask_wm.nii.gz`, `_mask_gm.nii.gz`, `_mask_csf.nii.gz`) for visual QC.

**Algorithm** (one paragraph)

Run RESTORE per voxel for FA and the tensor eigenvectors. Then select training voxels by the staged algorithm MRtrix runs, whose discriminator is the **signal decay metric** `SDM = mean over shells of log(S̄₀/S̄_b)`, volume-weighted — a tensor-free stand-in for `b·ADC`, so it is not confounded by the tensor model breaking down in free water:

1. *Preparation* — erode the brain mask `--dh-erode` passes (default 3), compute the SDM, drop voxels where any shell mean or the SDM itself is non-finite or non-positive.
2. *Crude* — `FA > --dh-fa` (default 0.2) splits WM off; a parameter-free optimal threshold (Ridgway et al. 2009) on the SDM splits CSF from GM in the remainder.
3. *Refined* — WM sheds its high-SDM tail (median + 2 × 1.4826 MAD), and those outliers are re-offered to CSF; GM and CSF each shed their partial-volume tails by a further optimal threshold.
4. *Final* — CSF is the top `--dh-csf`% of refined CSF by SDM (default 10), GM the `--dh-gm`% of refined GM closest to the refined-GM median SDM (default 2), single-fibre WM the highest-FA `--dh-sfwm`% of refined WM (default 0.5).

Per tissue, average the b=0 and DWI-shell signals; for WM, fit zonal SH per voxel relative to its principal eigenvector before averaging across voxels.

Deviations from MRtrix, all documented at their site in `src/multitissue/dhollander.rs`: FA comes from the existing RESTORE fit rather than a re-run `dwi2tensor`; SDM shells are b-value clusters, so the metric is also defined for non-shelled CS-DSI schemes; and the single-fibre WM stage is FA-ranked (MRtrix's `-wm_algo fa`) rather than the built-in Dhollander 2019 two-tissue-CSD metric.

`--legacy-tissue-selection` restores the pre-2026 threshold triple (CSF = top `--md-csf-pct` of MD, WM = `FA > --fa-wm-threshold` with eigenvalue dominance above `--fiber-dominance-ratio`, GM = the rest). It is kept only to reproduce older runs: selecting CSF by MD admits partial-volume voxels, which on QST synthetic data put the CSF response b=0 amplitude at 1.88× WM against MRtrix's 3.16× on identical data. The staged selection scores 3.20×.

**Key flags**

| Flag | Default | Purpose |
| --- | --- | --- |
| `--fa-wm-threshold` | `0.7` | WM single-fibre FA gate. |
| `--fiber-dominance-ratio` | `2.0` | Eigenvalue ratio gate for "single fibre" (suppresses crossings). 0 to disable. |
| `--md-csf-pct` | `2.5` | Top-N% MD voxels classified as CSF. |
| `--lmax-wm` | `8` | Max SH order for the WM response. |
| `--restore-max-iter`, `--restore-tol`, `--restore-outlier-threshold` | RESTORE defaults | Knobs for the underlying DTI fit. |

**Example**

```bash
cs-response \
    --dwi  /path/to/dwi.nii.gz \
    --bval /path/to/dwi.bval --bvec /path/to/dwi.bvec \
    --mask /path/to/brain_mask.nii.gz \
    --output-wm  out/wm_response.txt \
    --output-gm  out/gm_response.txt \
    --output-csf out/csf_response.txt \
    --diagnostics

cs-ss3t \
    --dwi  /path/to/dwi.nii.gz \
    --bval /path/to/dwi.bval --bvec /path/to/dwi.bvec \
    --mask /path/to/brain_mask.nii.gz \
    --response-wm  out/wm_response.txt \
    --response-gm  out/gm_response.txt \
    --response-csf out/csf_response.txt \
    --output-wm  out/wm_fod.nii.gz \
    --output-gm  out/gm.nii.gz \
    --output-csf out/csf.nii.gz
```

End-to-end runtime on a 250K-voxel clinical brain: ~5 s for `cs-response` + ~3 min for `cs-ss3t`.

---

## `cs-mtnorm` — Multi-tissue intensity normalisation

Native-Rust port of the log-domain algorithm behind MRtrix3's `mtnormalise`
(Raffelt et al. ISMRM 2017; Dhollander et al. ISMRM 2021). Closes the SS3T loop
without depending on MRtrix's `mtnormalise` at runtime.

Algorithm: estimate a smooth multiplicative field `f(v) = exp(polynomial(v))`
(default order 3) and per-tissue balance factors `b_t` (geometric mean 1) such
that `Σ_t b_t x_t(v) ≈ T · f(v)` over inlier voxels. Each of 15 main iterations
runs an inner loop (≤7 iters) that solves the balance factors by least squares
and rejects outlier voxels via an IQR rule on `log(Σ_t b_t x_t / f)`, then
refits the polynomial log-field by weighted least squares. Output:
`y_t(v) = x_t(v) / f(v)` for every SH coefficient — the balance factors steer
the fit but are only baked into the output with `--balanced`, matching MRtrix.

License posture: a port of MRtrix3's `cpp/cmd/mtnormalise.cpp`, which is
MPL-2.0 — see [Third-party licences](#third-party-licences).

**Inputs**
- `--in-{wm,gm,csf}` — three tissue NIfTIs (WM 4D with n_sh channels; GM/CSF 3D or 4D-with-singleton).
- `--mask` — 3D brain mask.

**Outputs**
- `--out-{wm,gm,csf}` — bias-corrected tissue NIfTIs (4D f32).
- With `--diagnostics`: `<wm_stem>_bias.nii.gz` — the recovered bias field for inspection.

**Key flags**

| Flag | Default | Purpose |
| --- | --- | --- |
| `--poly-order` | `3` | Polynomial order for the bias-field fit (20 monomials at order 3). |
| `--target-median` | off | Use the median observed sum as `T` instead of `1/sqrt(4π)`. Preserves global scale. |
| `--target-sum N` | — | Explicit target `T` (overrides both default and median). |
| `--niter` | `15` | Main iterations (field updates), matching MRtrix3. |
| `--balance-maxiter` | `7` | Max inner balance-factor / outlier-rejection iterations per main iteration, matching MRtrix3. |
| `--balanced` | off | Multiply the *output* tissues by their balance factors, like MRtrix3 `mtnormalise -balanced`. Has critical consequences for AFD normalisation — leave off unless you know why you need it. |

Empirical parity vs MRtrix mtnormalise (measured 2026-08-12, identical pre-norm FOD inputs from the QST pilot, same brain mask): balance factors agree to 4–5 significant digits, field ratio cs/MRtrix = 1.00000 across the mask (r = 1.000000), and the WM-dominant-voxel tissue-sum median is 0.2798 for both (target 0.28209). The implementations are numerically interchangeable.

**Example**

```bash
cs-mtnorm \
    --in-wm  out/${SUB}_wm.nii.gz \
    --in-gm  out/${SUB}_gm.nii.gz \
    --in-csf out/${SUB}_csf.nii.gz \
    --mask   ${SUB}_brain_mask.nii.gz \
    --out-wm  out/${SUB}_wm_norm.nii.gz \
    --out-gm  out/${SUB}_gm_norm.nii.gz \
    --out-csf out/${SUB}_csf_norm.nii.gz \
    --diagnostics
```

---

## `cs-ss3t-full` — End-to-end SS3T pipeline in one command

Chains [`cs-response`](#cs-response--wmgmcsf-response-estimation-dhollander-2016) →
[`cs-ss3t`](#cs-ss3t--single-shell-3-tissue-csd) →
[`cs-mtnorm`](#cs-mtnorm--multi-tissue-intensity-normalisation) so a single
invocation goes from raw single-shell DWI to normalised WM/GM/CSF tissue maps
without any external MRtrix tooling. All intermediates stay in memory; nothing
is written to disk until the final outputs.

Stages can be skipped:
- `--response-{wm,gm,csf} PATH` — skip estimation, use supplied responses.
- `--no-normalize` — skip the mtnormalise stage.

**Example: full pipeline from scratch**

```bash
cs-ss3t-full \
    --dwi  /path/to/dwi.nii.gz \
    --bval /path/to/dwi.bval --bvec /path/to/dwi.bvec \
    --mask /path/to/brain_mask.nii.gz \
    --output-wm  out/${SUB}_wm.nii.gz \
    --output-gm  out/${SUB}_gm.nii.gz \
    --output-csf out/${SUB}_csf.nii.gz \
    --diagnostics
```

**Example: reuse pre-computed responses**

```bash
cs-ss3t-full \
    --dwi  /path/to/dwi.nii.gz \
    --bval /path/to/dwi.bval --bvec /path/to/dwi.bvec \
    --mask /path/to/brain_mask.nii.gz \
    --response-wm  responses/wm.txt \
    --response-gm  responses/gm.txt \
    --response-csf responses/csf.txt \
    --output-wm  out/${SUB}_wm.nii.gz \
    --output-gm  out/${SUB}_gm.nii.gz \
    --output-csf out/${SUB}_csf.nii.gz
```

Runtime on a 250K-voxel clinical brain: ~5 s (DTI) + ~3 min (SS3T) + ~1 s (mtnormalise) ≈ **4 minutes total**. Most flags from the underlying `cs-response`, `cs-ss3t`, and `cs-mtnorm` binaries are exposed as `--<knob>` pass-throughs (run `cs-ss3t-full --help` for the full list).

---

## `cs-ss3t` — Single-Shell 3-Tissue CSD

A native-Rust port of the SS3T-CSD algorithm (Dhollander & Connelly, ISMRM 2016, abstract 3010), independent of the MRtrix3Tissue fork. Decomposes a single-shell DWI into a white-matter fiber-orientation distribution plus isotropic gray-matter and CSF compartments. The inner constrained-least-squares solve uses Goldfarb-Idnani's 1983 active-set algorithm, so cs-ss3t produces output that matches MRtrix3Tissue's `ss3t_csd_beta1` (followed by `mtnormalise`) to high precision: per-voxel WM Pearson r ~0.996, GM/CSF spatial r ~0.999. See [tests/ss3t_parity.rs](tests/ss3t_parity.rs) for the full comparison harness.

**Inputs**
- 4D DWI NIfTI (`--dwi`) with **exactly one b=0 shell and one DWI shell**. cs-ss3t will refuse a multi-shell DWI; for that, MRtrix3's MSMT-CSD is the right tool.
- FSL `.bval` and `.bvec` (`--bval`, `--bvec`).
- Optional brain mask (`--mask`); auto-thresholded from the b0 mean if omitted.
- Three MRtrix-format tissue response `.txt` files (`--response-wm`, `--response-gm`, `--response-csf`). See "Estimating the responses" below.

**Outputs**
- Three 4D NIfTIs:
  - `--output-wm` — WM FOD, **45 SH coefficients** per voxel at `lmax = 8`, in MRtrix even-real-SH order (the same convention TRXViz, mrview, and `cs-odf` use).
  - `--output-gm` — GM compartment, **1 coefficient** per voxel.
  - `--output-csf` — CSF compartment, **1 coefficient** per voxel.
- One JSON sidecar at `<wm_stem>.json` recording the inputs, algorithm parameters, sphere identity, and provenance.
- With `--diagnostics`: sibling NIfTIs `<wm_stem>_iters.nii.gz` (sum of inner ICLS iterations across the 7 inner solves per voxel), `_residual.nii.gz` (final ‖augmented_signal − full_prediction‖₂), `_converged.nii.gz` (1 iff every inner ICLS call converged).

**Estimating the responses**

cs-ss3t doesn't estimate responses itself — you generate them once per dataset using MRtrix3's `dwi2response`:

```bash
# Convert FSL bval/bvec to MRtrix .b in a .mif container
mrconvert dwi.nii.gz dwi.mif -fslgrad dwi.bvec dwi.bval

# Estimate WM/GM/CSF responses (Dhollander 2016 algorithm)
dwi2response dhollander dwi.mif \
    response_wm.txt response_gm.txt response_csf.txt \
    -mask brain_mask.nii.gz
```

The resulting `.txt` files are plain text with one row per shell (b=0 row first by convention, then the DWI shell), each row holding the even-order zonal SH coefficients of the response on that shell. cs-ss3t accepts files exactly as `dwi2response` produces them — no conversion needed. If the WM file declares a higher lmax than `--lmax-wm`, cs-ss3t silently clamps and emits a stderr line.

**Key flags**

| Flag | Default | Purpose |
| --- | --- | --- |
| `--niter` | `3` | SS3T outer alternating iterations. Must be ≥ 2; the original abstract uses 3. |
| `--bzero-pct` | `10` | Weight on b=0 volumes as a percentage of the DWI volume count. Controls how heavily the b=0 measurements contribute to the inverse. |
| `--lmax-wm` | `8` | Maximum SH order for the WM FOD. Clamped to the WM response file's lmax with a stderr warning if the file is smaller. Matches qsirecon's hardcoded value. |
| `--icls-max-iter` | `200` | Inner ICLS active-set iteration cap per CSD inner solve. Rarely needs tuning — typical voxels finish in 20–80. |
| `--icls-tol` | `1e-10` | Constraint-satisfaction tolerance: `(C x)_i ≥ -tol` is treated as "satisfied". |
| `--icls-epsilon` | `1e-10` | Tikhonov stabilizer added to the ICLS Hessian diagonal to guarantee strict positive-definiteness. Increase if you see Cholesky failures on rank-deficient inputs. |
| `--diagnostics` | off | Also write `_iters`, `_residual`, `_converged` sibling NIfTIs next to the WM output. |
| `--big-delta`, `--small-delta` | estimated | Diffusion timing in seconds; SS3T itself ignores these but cs-ss3t records them in the sidecar for downstream tools. Estimated via TORTOISE's heuristic if either is missing. |
| `--gmax` | `0.08` | Used only when deltas need to be estimated. |
| `--no-bvec-rotation` | off | Keep bvecs in image-axis frame. The default rotation matches MRtrix's world-RAS convention; pass this only if you've explicitly built responses against image-axis bvecs. |
| `--threads` | auto | Cap rayon's worker pool. (See "Running on HPC" above.) |
| `--overwrite`, `--quiet`, `--progress-interval-secs`, `--provenance` | as elsewhere | (See "Running on HPC" above.) |

`cs-ss3t --help` lists the full set.

**Example: SS3T on a preprocessed PNC subject**

```bash
ROOT=ss3t_test_data
STEM=sub-01_ses-1_space-ACPC

cs-ss3t \
    --dwi  ${ROOT}/${STEM}_desc-preproc_dwi.nii.gz \
    --bval ${ROOT}/${STEM}_desc-preproc_dwi.bval \
    --bvec ${ROOT}/${STEM}_desc-preproc_dwi.bvec \
    --mask ${ROOT}/${STEM}_desc-brain_mask.nii.gz \
    --response-wm  ${ROOT}/${STEM}_model-ss3t_param-fod_label-WM_dwimap.txt \
    --response-gm  ${ROOT}/${STEM}_model-ss3t_param-fod_label-GM_dwimap.txt \
    --response-csf ${ROOT}/${STEM}_model-ss3t_param-fod_label-CSF_dwimap.txt \
    --output-wm  out/${STEM}_wm.nii.gz \
    --output-gm  out/${STEM}_gm.nii.gz \
    --output-csf out/${STEM}_csf.nii.gz \
    --diagnostics
```

**Optional: post-process with `mtnormalise`**

The raw cs-ss3t outputs are *not* intensity-normalized — coefficient magnitudes follow the response-function units. Most downstream tools (MRtrix3 tractography, qsirecon SS3T pipelines) consume mtnormalised outputs where the three tissues sum to a constant globally. To normalize, follow up with MRtrix3's `mtnormalise` (a separate dependency, not bundled with cs_dmri):

```bash
mtnormalise out/${STEM}_wm.nii.gz  out/${STEM}_wm_norm.nii.gz \
            out/${STEM}_gm.nii.gz  out/${STEM}_gm_norm.nii.gz \
            out/${STEM}_csf.nii.gz out/${STEM}_csf_norm.nii.gz \
            -mask ${ROOT}/${STEM}_desc-brain_mask.nii.gz
```

**Visualizing the WM FOD**

The WM output is a 4D NIfTI with 45 SH coefficients per voxel in MRtrix even-real-SH order, so any tool that reads MRtrix-basis SH images (mrview's `-odf.load_sh`, `cs-odf`, TRXViz with `--name coefficients`) opens it directly:

```bash
mrview out/${STEM}_wm_norm.nii.gz -odf.load_sh out/${STEM}_wm_norm.nii.gz
```

`cs-odf` will project the WM FOD to fixels and an ODX archive in the same step it does for SHORE-fitted coefficients — see the example below.

**What it means in practice**

SS3T-CSD answers a different question from `cs-fit`: instead of giving you a basis-coefficient field that re-synthesizes the DWI, it gives you a tissue partition. Each voxel comes out as three numbers (WM mass, GM mass, CSF mass) plus a directional WM FOD. That's the input format for fixel-based analysis, MRtrix anatomically-constrained tractography, and partial-volume-aware diffusion metrics.

The single-shell setting is fundamentally underdetermined — three isotropic compartments can't be fully separated from b=0 + one-shell signal alone. SS3T breaks the degeneracy with the SDM ordering assumption SDM(WM) < SDM(GM) < SDM(CSF) and the iterative alternation that gives the algorithm its name: at each outer iteration it subtracts the current CSF prediction from the data, fits WM+GM on the residual, then subtracts WM and refits GM+CSF. The `niter=3` default is the abstract's recommendation. Each inner CSD solve is a strictly-convex QP with linear inequality constraints (FOD amplitudes ≥ 0 on a dense sphere; isotropic coefficients ≥ 0); cs-ss3t solves it via Goldfarb-Idnani 1983, the same dual active-set method MRtrix3Tissue uses. License-wise the implementation is paper-derived rather than ported from MRtrix's MPL-2.0 source.

---

## What the operations mean in practice

### Fitting (`cs-fit`)

Replace the noisy, scheme-specific 4D DWI with a per-voxel coefficient vector in a smooth, scheme-agnostic basis. The 3D-SHORE basis factors as a Laguerre-polynomial radial part (in q-space radius) times a real spherical harmonic angular part (even ℓ only). Once you have coefficients, every downstream operation — ODF projection, peak extraction, DWI synthesis — is a fixed linear transformation of those numbers.

The `--alpha-mode path-bic` strategy sweeps a logarithmically spaced grid of L1 sparsity weights from `α_max` (everything zero) down to `α_max · path-eps` and picks, per voxel, the α that minimizes the Bayesian Information Criterion. That trades model size against residual fit on a per-voxel basis: a CSF voxel ends up with fewer nonzero coefficients than a crossing-fiber voxel, automatically. L2 (Tikhonov) is the closed-form alternative: faster, denser, and useful as a parity check or as a smoke test when a fit looks suspect.

### ODF projection (`cs-odf`)

The Tournier-SH transform of a SHORE coefficient vector is the same matrix multiply for every voxel, so this step is cheap. The interesting part is fixel extraction: a discrete sphere search (the DSI Studio ODF8 hemisphere, 321 vertices) finds local maxima as seeds, then each accepted seed is refined to a sub-vertex direction by Newton iteration on the continuous SH expansion (`odx-rs::peak_finder::SpherePeakFinder::find_peaks_with_sh`, which mirrors MRtrix's `Math::SH::get_peak`). The `--peak-relative-threshold` and `--peak-min-separation-deg` flags prune the refined peaks. QA per fixel is `peak − min(ODF)` — a directionality measure that's small in CSF and large in white matter.

Each fixel also carries a **dispersion** value: the same FMLS watershed `fod2fixel` uses (`odx-rs::fmls`) segments the ODF into lobes, and dispersion is the lobe's quadrature integral divided by its peak amplitude — MRtrix `fod2fixel -disp`. The peak finder still decides which fixels exist (FMLS runs with permissive thresholds purely to *measure* each accepted peak's lobe); each refined peak is matched to its lobe by nearest sphere vertex. Because both integral and peak scale linearly with the fODF, dispersion is invariant to QA normalization and `--quantitative` rescaling, so it means the same thing on every output scale — small for a tight single-fibre lobe, large for a fanning one.

The output ODX bundles SH, fixels, GFA, anisotropic-power, and any `cs-fit --diagnostics` scalars into one archive that TRXViz reads directly.

### Synthesis (`cs-synth`)

Re-evaluate the SHORE expansion at any q-space points you like. Three real uses:

1. **Self-consistency check** — predict the original scheme back and compare to observed DWI; large residuals flag suspect fits.
2. **Cross-scheme prediction / harmonization** — fit on a sparse acquisition (HASC55) and predict a denser one (ABCD) to validate how much information the sparse scheme retained, or to harmonize across protocols.
3. **Compression** — coefficients are roughly an order of magnitude smaller than 4D DWI, so the coefficient NIfTI plus its sidecar is a compact, lossy archival format.

---

## End-to-end example: fit → ODF → cross-scheme prediction

```bash
SUB=sub-01
DWI_ROOT=/data/qsiprep/${SUB}/ses-1/dwi
OUT=~/cs-bench-csdsi/focused
mkdir -p "${OUT}"

# 1. Fit the HASC55 (sub-sampled) acquisition
cs-fit \
    --dwi  "${DWI_ROOT}/${SUB}_ses-1_acq-HASC55_run-01_space-T1w_desc-preproc_dwi.nii.gz" \
    --bval "${DWI_ROOT}/${SUB}_ses-1_acq-HASC55_run-01_space-T1w_desc-preproc_dwi.bval" \
    --bvec "${DWI_ROOT}/${SUB}_ses-1_acq-HASC55_run-01_space-T1w_desc-preproc_dwi.bvec" \
    --mask "${DWI_ROOT}/${SUB}_ses-1_acq-HASC55_run-01_space-T1w_desc-brain_mask.nii.gz" \
    --reg l1 --alpha-mode l2-anchored --radial-order 4 \
    --diagnostics \
    --output "${OUT}/${SUB}_HASC55_coeffs.nii.gz"

# 2. Project to ODFs and extract fixels
cs-odf \
    --coeffs "${OUT}/${SUB}_HASC55_coeffs.nii.gz" \
    --output "${OUT}/${SUB}_HASC55.odx"

# 3. Predict the ABCD (full-DSI) scheme from the HASC55 fit
cs-synth \
    --coeffs "${OUT}/${SUB}_HASC55_coeffs.nii.gz" \
    --bval   "${DWI_ROOT}/${SUB}_ses-1_acq-ABCD_space-T1w_desc-preproc_dwi.bval" \
    --bvec   "${DWI_ROOT}/${SUB}_ses-1_acq-ABCD_space-T1w_desc-preproc_dwi.bvec" \
    --output "${OUT}/${SUB}_HASC55-to-ABCD_predicted.nii.gz"
```

[scripts/gold_standard.sh](scripts/gold_standard.sh) wraps this pattern across every subject in the bundles root.

---

## Library use

The crate is also a library. Top-level re-exports from `src/lib.rs`:

| Item | Use |
| --- | --- |
| `Basis`, `BasisMetadata`, `ShoreBasis` | Build / inspect a SHORE basis. |
| `FitConfig`, `fit_volume`, `fit_volume_with_alpha_strategy` | Run the per-voxel fit driver. |
| `fit_volume_reporting`, `fit_volume_with_alpha_strategy_reporting` | Same drivers with an `on_voxel: impl Fn() + Sync` callback for progress. |
| `FistaSolver`, `TikhonovSolver` | Solver backends. |
| `AlphaPath`, `AlphaStrategy`, `alpha_max` | Per-voxel α-path machinery. |
| `FitDiagnostics`, `VoxelAlphaResult` | Diagnostic outputs from a fit. |
| `synthesize_volume`, `synthesize_volume_reporting` | Reverse pass: coefficients → DWI; the `_reporting` variant takes an on-plane callback. |
| `multitissue::{Ss3tConfig, Ss3tResponses, Ss3tPlan, fit_volume_ss3t, fit_volume_ss3t_reporting}` | SS3T pipeline: build the per-volume plan, run the per-voxel alternating loop in parallel. |
| `multitissue::{TissueResponse, Ss3tSidecar}` | MRtrix `.txt` response loader; sidecar struct describing an SS3T fit. |
| `multitissue::{LmaxWmStrategy, Ss3tVolumePlan, fit_voxel_path_bic_into}` | Per-voxel BIC-driven WM lmax selection for SS3T. |
| `multitissue::{DhollanderConfig, estimate_responses, write_response_txt}` | Three-tissue response estimation (Dhollander 2016) — used by `cs-response`. |
| `multitissue::{MtnormaliseConfig, mtnormalise, target_sum_mrtrix_default}` | Polynomial bias-field correction + optional per-tissue scaling (Raffelt 2017) — used by `cs-mtnorm` and `cs-ss3t-full`. |
| `dti::{RestoreConfig, fit_voxel_restore, fit_volume_restore_reporting}` | RESTORE robust DTI fit (Chang et al. MRM 2005) — used by `cs-dti` and `cs-response`. |
| `solver::{IclsSolver, IclsConfig, IclsDiagnostics, IclsWorkspace}` | Goldfarb-Idnani 1983 ICLS — exposed publicly so other reconstructions can reuse the active-set QP solver. |
| `solver::{ShoreIclsSolver}` | SHORE-fit with hard amplitude non-negativity via ICLS (`cs-fit --reg amp-nn`). |
| `solver::{CsdSolver, CsdConfig}` | Tournier 2007 iterative-reweighting CSD; faster than ICLS but produces softer constraint enforcement. Kept for users who want the single-tissue Tournier behavior. |
| `voxel_loop::{run, run_init}` | Generic per-masked-voxel parallel driver shared by every reconstruction in the crate. |
| `io::aux::{write_3d_f32, write_3d_u8, write_3d_u32_as_f32, write_4d_f32, sibling_path}` | Atomic NIfTI writers used by every CLI binary for diagnostic / output volumes. |
| `GradientTable` | bvals, bvecs, big/small delta, frame conventions. |
| `CoefficientsFile`, `SidecarMetadata` | Read/write the coefficient NIfTI + JSON sidecar pair. The pair is written atomically (temp + rename). |
| `DwiData`, `load_dwi` | Read DWI + bval + bvec + mask with affine canonicalization. |
| `configure_rayon_threads`, `effective_thread_count`, `ThreadSource` | Resolve the rayon thread cap (CLI → SLURM → RAYON env → default), report which source won. |
| `Heartbeat` | Time-based progress reporter for long parallel loops; non-TTY safe. |
| `Provenance`, `ProvenanceBuilder`, `ProvenanceMode` | Reproducibility block written into sidecars / ODX extras. PHI-defensive defaults. |
| `atomic_write`, `atomic_write_pair` | Helpers used by the bins for crash-safe output. |
| `VERSION`, `GIT_SHA`, `BUILD_TIMESTAMP` | Compile-time provenance constants. |

Module map (under `src/`): `basis` (SHORE basis + design matrix), `fit` (per-voxel parallel driver), `solver` (FISTA + Tikhonov + α-path + Tournier-CSD + Goldfarb-Idnani ICLS + SHORE-ICLS amp-NN), `multitissue` (SS3T pipeline: response I/O, response *estimation* (Dhollander 2016), forward operators, per-voxel routine, volume driver, sidecar), `dti` (RESTORE robust tensor fit), `sh` (zonal-convolution helpers + re-exports of `odx-rs`'s MRtrix-basis SH primitives), `odf` (analytical SHORE → Tournier SH), `qspace` (gradient table + frame rotation + shell partition), `synth` (reverse pass), `io` (NIfTI + sidecar I/O + shared aux writers), `voxel_loop` (generic parallel-over-masked-voxels driver), `math` (Laguerre, Legendre, Gamma, hypergeometric).

---

## Tests and benchmark scripts

```bash
cargo test
```

runs:

- [tests/alpha_selection.rs](tests/alpha_selection.rs) — per-voxel BIC-driven α-path unit tests
- [tests/roundtrip.rs](tests/roundtrip.rs) — synthetic DWI → fit → synthesize → compare
- [tests/scalars.rs](tests/scalars.rs) — microstructure scalar recovery on synthetic tensors
- **[tests/optimization_kkt.rs](tests/optimization_kkt.rs)** — KKT residuals for plain & non-negative FISTA and the Tikhonov normal equations. The gold-standard correctness witness: passing means the solver hit the *true* convex minimum within tolerance.
- **[tests/optimization_limits.rs](tests/optimization_limits.rs)** — boundary behavior: α → 0 matches OLS, α ≥ α_max returns 0, square invertible M with α → 0 recovers M⁻¹y, λ → 0 / λ → ∞ Tikhonov limits, log-α-path endpoints exact.
- **[tests/optimization_recovery.rs](tests/optimization_recovery.rs)** — planted-support sparse recovery: fixed-α support agreement, path-BIC support-size sanity, recovery-vs-noise monotonicity, BIC-formula pin.
- **[tests/optimization_sklearn.rs](tests/optimization_sklearn.rs)** — cross-check against scikit-learn's `Lasso` on 8 fixtures (well/ill-conditioned, over/underdetermined, non-negative, near α_max, near OLS). Asserts coefficients, objective value, and KKT residual all agree.
- **[tests/ss3t_parity.rs](tests/ss3t_parity.rs)** — `#[ignore]`-marked, local-only end-to-end comparison of `cs-ss3t` against MRtrix3Tissue / qsirecon reference outputs on a real subject. Looks for `<repo>/../ss3t_test_data/` and skips silently if absent. If `mtnormalise` is on PATH it normalizes both sides before comparing. Writes `cs_dmri_*.nii.gz`, `cs_dmri_*_norm.nii.gz`, and `reference_*.nii.gz` to `target/ss3t_parity/` for off-line inspection (mrview, fsleyes), prints a per-tissue parity report, and asserts only that the pipeline runs end-to-end. Run explicitly with `cargo test --release --test ss3t_parity -- --ignored --nocapture`.

The sklearn fixtures live in [tests/data/lasso_fixtures/](tests/data/lasso_fixtures/) as one `.bin` per case (~365 KB total, custom self-describing format — see [scripts/generate_lasso_fixtures.py](scripts/generate_lasso_fixtures.py) for the layout) plus a `manifest.json` listing them. The Rust loader uses `f64::from_le_bytes` and pulls in **no extra crates** (no `npyz`, no zip, no zstd-sys), so the test compiles cleanly on HPCs that don't have a working C toolchain for the bundled-zstd build. Regenerate when you change the case list:

```bash
mamba activate trx     # env that has sklearn / numpy
python scripts/generate_lasso_fixtures.py
cargo test --test optimization_sklearn
```

Seeds are derived deterministically from the case name (sha256), so re-running the script produces byte-identical fixtures across machines.

[scripts/](scripts/):
- `gold_standard.sh` — multi-subject fit / ODF / synth pipeline across the bundles root.
- `bench_fit_options.py` — sweep solver / α-strategy / radial-order combinations and emit per-config outputs under `~/cs-bench-csdsi/focused/`.
- `bench_plots.R`, `gold_compare.Rmd` — R-side analysis and a parity report against qsirecon.
- `validate_microstructure.py` — synthesize multi-tensor voxels, fit cs_dmri *and* dipy iso-MAPMRI, plot every microstructure scalar against analytical truth and against dipy, and report Pearson r + median relative error. Run inside the `qsirecon` conda env.
- `microstructure_math.md` — closed-form derivations for the new scalars: BrainSuiteSHORE ↔ dipy iso-MAPMRI basis equivalence, ζ ↔ µ change of variables, per-mode α conversion factor, and the kernel for each scalar.

---

## License

cs_dmri is dual-licensed under either of

- the [Apache License, Version 2.0](LICENSE-APACHE), or
- the [MIT license](LICENSE-MIT),

at your option, except for the third-party-derived code listed below: two
MRtrix3-derived files under the Mozilla Public License 2.0, and dipy-derived
functions under the BSD 3-Clause licence. The crate's SPDX expression is
`(MIT OR Apache-2.0) AND MPL-2.0 AND BSD-3-Clause`.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work by you, as defined in the Apache-2.0 license, shall
be dual-licensed as above, without any additional terms or conditions.

### Third-party licences

Two files are ports of MRtrix3 code:

- [`src/multitissue/dhollander.rs`](src/multitissue/dhollander.rs) — the
  voxel-selection stages of `dwi2response dhollander` and the `mrthreshold` /
  `OptimalThreshold` / `Erode` / `median` primitives it calls;
- [`src/multitissue/mtnormalise.rs`](src/multitissue/mtnormalise.rs) — the
  log-domain field fit, balance factors and outlier rejection of `mtnormalise`.

MRtrix3 is © 2008-2026 the MRtrix3 contributors under the **Mozilla Public
License 2.0**, so those files are licensed MPL-2.0 rather than MIT/Apache-2.0,
and each carries its own notice and an `SPDX-License-Identifier: MPL-2.0` line.
See [LICENSE-MRTRIX](LICENSE-MRTRIX).

MPL-2.0 is a *file-scoped* copyleft: it governs the file that contains covered
code and requires that file's source to stay available under MPL-2.0, but it
does not reach the files that merely call into it, so the two licences coexist.
Keeping that true is a maintenance constraint, not a formality — **do not copy
MRtrix-derived logic out of these files into other modules**, and anyone
redistributing cs_dmri must keep those files' source available under MPL-2.0.

[`src/qc.rs`](src/qc.rs) contains functions ported from
[dipy](https://dipy.org/) (`find_qspace_neighbors`, `neighboring_dwi_correlation`,
and `find_qspace_contrast` / `dwi_contrast` from dipy PR #4224). Those are
© 2008-2026 the dipy developers under the BSD 3-Clause licence; see
[LICENSE-DIPY](LICENSE-DIPY).
