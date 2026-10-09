# cs-qc

Image-quality metrics for a DWI series, written as JSON and/or a one-row TSV with a JSON data dictionary. See {doc}`../user/qc`.

```text
DWI image-quality metrics: neighboring DWI correlation (NDC), DWI contrast ratio, outlier slices and fixel coherence

Usage: cs-qc [OPTIONS] --dwi <DWI> --bval <BVAL> --bvec <BVEC>

Options:
      --dwi <DWI>                    4D DWI NIfTI input
      --bval <BVAL>                  FSL bval file
      --bvec <BVEC>                  FSL bvec file
      --mask <MASK>                  Brain mask NIfTI on the DWI grid. If omitted, the masked metrics use voxels where the mean b=0 image exceeds 1% of its maximum, and the JSON report records `mask_source: auto-b0`
      --b0-threshold <B0_THRESHOLD>  b-value at or below which a volume is treated as b=0, in s/mm² [default: 50]
      --slice-axis <SLICE_AXIS>      Spatial axis (0, 1 or 2) along which slices are assessed for outliers [default: 2]
      --no-coherence                 Do not fit the diffusion tensor or compute fixel coherence
      --output-json <OUTPUT_JSON>    Write the full report (metrics, outlier slices and settings) as JSON
      --output-tsv <OUTPUT_TSV>      Write the metrics as a one-row TSV, with a `<stem>.json` data dictionary describing each column
      --prefix <PREFIX>              Prefix added to TSV column names (e.g. `raw_`) [default: ""]
      --threads <THREADS>            Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used, then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU
      --overwrite                    Overwrite existing output files. Without this flag, existing outputs cause an error
      --quiet                        Suppress progress and summary messages on stderr
  -h, --help                         Print help
  -V, --version                      Print version
```
