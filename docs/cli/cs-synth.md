# cs-synth

Synthesizes a DWI series for a target gradient table from `cs-fit` coefficients. See {doc}`../user/shore`.

```text
Synthesize diffusion-weighted images from SHORE coefficients for a given gradient table

Usage: cs-synth [OPTIONS] --coeffs <COEFFS> --bval <BVAL> --bvec <BVEC> --output <OUTPUT>

Options:
      --coeffs <COEFFS>
          Coefficient NIfTI written by `cs-fit`. The JSON sidecar is read from the matching `.json` file next to it

      --bval <BVAL>
          FSL bval file of the gradient table to synthesize

      --bvec <BVEC>
          FSL bvec file of the gradient table to synthesize

      --big-delta <BIG_DELTA>
          Diffusion time Δ (big delta), in seconds. If omitted, the value in the coefficient sidecar is used

      --small-delta <SMALL_DELTA>
          Gradient pulse duration δ (small delta), in seconds. If omitted, the value in the coefficient sidecar is used

      --gmax <GMAX>
          Maximum gradient amplitude, in T/m. Used only when Δ and δ are absent from the sidecar and must be estimated
          
          [default: 0.08]

      --output <OUTPUT>
          Output 4D DWI NIfTI

      --threads <THREADS>
          Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used, then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU

      --overwrite
          Overwrite an existing output NIfTI and sidecar JSON. Without this flag, existing outputs cause an error

      --quiet
          Suppress periodic progress and per-step summary messages

      --progress-interval-secs <PROGRESS_INTERVAL_SECS>
          Interval between progress messages during synthesis, in seconds
          
          [default: 30]

      --provenance <PROVENANCE>
          Provenance recorded in a JSON sidecar next to the output; with `none`, no sidecar is written

          Possible values:
          - minimal: Version, build and run-time summary only. Default
          - full:    Additionally records the command line, host name and start time
          - none:    No provenance
          
          [default: minimal]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
