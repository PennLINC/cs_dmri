# cs-odf

Computes ODF spherical-harmonic coefficients, peaks and scalars from `cs-fit` coefficients and writes an ODX file. See {doc}`../user/shore`.

```text
Project SHORE coefficients onto ODF spherical harmonics (MRtrix3/Tournier convention) and write an ODX file

Usage: cs-odf [OPTIONS] --coeffs <COEFFS> --output <OUTPUT>

Options:
      --coeffs <COEFFS>
          Coefficient NIfTI written by cs-fit. The JSON sidecar is read from the same location

      --output <OUTPUT>
          Output .odx file, or output directory with --directory

      --mask <MASK>
          Brain mask NIfTI. If omitted, all voxels with at least one non-zero coefficient are used

      --lmax <LMAX>
          Maximum (even) SH order. Default: the largest even integer ≤ the SHORE radial order

      --name <NAME>
          Name of the SH field under sh/ in the ODX
          
          [default: coefficients]

      --directory
          Write the ODX as a directory instead of a zip archive

      --no-anisotropic-power
          Do not compute the per-voxel anisotropic power map (Dell'Acqua et al., 2014)

      --ap-norm-factor <AP_NORM_FACTOR>
          Normalisation factor in the logarithm of the anisotropic power map
          
          [default: 0.00001]

      --no-global-normalize
          Do not apply global ODF normalisation. By default the quantity QA = max(ODF) − min(ODF) is computed in each voxel, and all SH coefficients are divided by its maximum over the mask, as in DSI Studio. Relative amplitudes between voxels are preserved

      --no-diagnostic-dpvs
          Do not copy the diagnostic maps written by `cs-fit --diagnostics` (`<stem>_r2.nii.gz`, `<stem>_rmse.nii.gz`, `<stem>_alpha.nii.gz`, `<stem>_bic.nii.gz`, `<stem>_sparsity.nii.gz`) into the ODX. By default, each of these found next to the coefficient NIfTI is stored as a per-voxel field

      --no-peaks
          Do not extract ODF peaks (fixels). By default, local maxima of each ODF are located on the DSI Studio ODF8 hemisphere (321 vertices), filtered by relative amplitude and angular separation, and refined by Newton iteration on the continuous SH representation

      --peak-npeaks <PEAK_NPEAKS>
          Maximum number of peaks per voxel
          
          [default: 5]

      --peak-relative-threshold <PEAK_RELATIVE_THRESHOLD>
          Discard peaks with amplitude below this fraction of the largest peak in the voxel
          
          [default: 0.5]

      --peak-min-separation-deg <PEAK_MIN_SEPARATION_DEG>
          Minimum angular separation between peaks, in degrees
          
          [default: 25]

      --threads <THREADS>
          Number of worker threads. If omitted, `$SLURM_CPUS_PER_TASK` is used, then `$RAYON_NUM_THREADS`, otherwise one thread per logical CPU

      --overwrite
          Overwrite an existing output ODX, output directory or microstructure NIfTI. Without this flag, existing outputs cause an error

      --quiet
          Suppress periodic progress and per-step summary messages

      --progress-interval-secs <PROGRESS_INTERVAL_SECS>
          Interval between progress messages, in seconds
          
          [default: 30]

      --provenance <PROVENANCE>
          Provenance recorded in the ODX as the extra value `cs_dmri_provenance`

          Possible values:
          - minimal: Version, build and run-time summary only. Default
          - full:    Additionally records the command line, host name and start time
          - none:    No provenance
          
          [default: minimal]

      --no-microstructure
          Do not compute propagator scalars from the SHORE coefficients: return to the origin, axis and plane probabilities (RTOP, RTAP, RTPP), mean squared displacement (MSD), q-space inverse variance (QIV) and non-Gaussianity (NG). By default these are computed in each voxel, outliers are set to NaN (see `--microstructure-outlier-factor`), and the maps are stored as per-voxel fields in the ODX together with per-scalar display ranges (`cs_dmri_microstructure_display`) and outlier counts (`cs_dmri_microstructure_outliers`). RTAP and RTPP are defined relative to the first peak direction and are NaN in voxels without a peak

      --microstructure-nifti
          Also write each propagator scalar to a NIfTI next to the ODX (`<output_stem>_rtop.nii.gz`, …) with the RAS+ affine

      --microstructure-outlier-factor <MICROSTRUCTURE_OUTLIER_FACTOR>
          Outlier threshold factor K: values of a propagator scalar greater than K times the 99th percentile of its finite values within the mask are set to NaN
          
          [default: 10]

      --no-microstructure-outlier-rejection
          Disable outlier rejection for propagator scalars. All finite values are retained

      --scalar-units <SCALAR_UNITS>
          Length unit of the propagator scalars

          Possible values:
          - um: q in 1/μm: RTOP in μm⁻³, RTAP in μm⁻², RTPP in μm⁻¹, MSD in μm², QIV in μm⁵. Default
          - mm: q in 1/mm: RTOP in mm⁻³, RTAP in mm⁻², RTPP in mm⁻¹, MSD in mm², QIV in mm⁵
          
          [default: um]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
