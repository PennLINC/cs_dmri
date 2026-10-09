# Quality control

cs_dmri computes a small set of image-quality metrics from a DWI series, its
gradient table and a brain mask. They are intended for screening and for
comparison across scans acquired with the same protocol.

```python
qc = dwi.qc()
qc.to_dict(prefix="raw_")                    # flat row
cs.QCReport.column_descriptions("raw_")      # data dictionary for that row
```

```bash
cs-qc --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
      --output-tsv qc.tsv --prefix raw_ --output-json qc.json
```

The tabular output is accompanied by a JSON data dictionary in the BIDS
convention (`LongName`, `Description`, `Units`) describing each column.

## Neighboring DWI correlation

Let $q_i = \sqrt{b_i}\,\mathbf{g}_i$ be the approximate q-space position of
volume $i$, with $b$-value $b_i$ and unit gradient direction $\mathbf{g}_i$.
For every diffusion-weighted volume $i$ (those with $b_i$ above the b=0
threshold), its neighbour $n(i)$ is the other diffusion-weighted volume that
minimises

$$
\min\left(\lVert q_i - q_j \rVert,\ \lVert q_i + q_j \rVert\right),
$$

which treats antipodal directions as equivalent. The neighboring DWI
correlation (NDC; Yeh et al., 2019) is the mean, over all diffusion-weighted
volumes, of the Pearson correlation between volume $i$ and volume $n(i)$
across voxels:

$$
\mathrm{NDC} = \frac{1}{|D|}\sum_{i \in D} \rho\left(S_i, S_{n(i)}\right).
$$

Because every diffusion-weighted volume contributes one term, the value does
not depend on the order in which volumes are stored. Head motion, eddy-current
distortion and signal dropout reduce the correlation between neighbouring
volumes and therefore lower NDC. Yeh et al. (2019) suggest that values below 0.4
indicate a low-quality image.

`ndc` is computed over all voxels in the field of view and `ndc_masked` over the
brain mask. The masked value is the one to compare between scans.

## DWI contrast ratio

For every diffusion-weighted volume $i$, a contrast volume $c(i)$ is chosen
whose q-space direction is closest to perpendicular to $q_i$: for each
candidate $q_j$, the component of $q_j$ perpendicular to $q_i$ is rescaled to
the length of $q_i$, and the candidate nearest to this vector is selected.
Candidates parallel to $q_i$ are excluded. The DWI contrast ratio is

$$
\frac{\frac{1}{|D|}\sum_{i} \rho\left(S_i, S_{n(i)}\right)}
     {\frac{1}{|D|}\sum_{i} \rho\left(S_i, S_{c(i)}\right)}.
$$

A ratio close to one indicates that volumes acquired along perpendicular
directions are about as similar as neighbouring ones, that is, the series
carries little angular contrast. Values below 1.1 are conventionally regarded as
poor, 1.1–1.3 as fair and above 1.3 as good. The ratio depends strongly on the
mask, so `dwi_contrast_ratio_masked` should be preferred.

## Outlier slices

Outlier slices are detected separately within each volume, without reference to
other volumes. Each slice along the slice axis is first smoothed in-plane with a
Gaussian kernel ($\sigma$ = 2 voxels). For an interior slice $k$, with the
in-mask voxels of that slice indexed by $v$, the inconsistency ratio is

$$
R_k = \frac{\operatorname{mean}_v \left| I_k(v) - \tfrac{1}{2}\left(I_{k-1}(v) + I_{k+1}(v)\right) \right|}
           {\tfrac{1}{2}\operatorname{mean}_v \left| I_{k+1}(v) - I_{k-1}(v) \right|}.
$$

A slice that varies smoothly between its neighbours has a small ratio. A slice
affected by signal dropout or corruption departs from both neighbours, and is
flagged when $R_k$ exceeds 2.5. Slices with fewer than 100 in-mask voxels and
the first and last slice are not scored. `n_outlier_slices` counts the flagged
(volume, slice) pairs; the Python report also lists them
({attr}`~cs_dmri.qc.QCReport.flagged_slices`).

## Fixel coherence

The fixel-coherence index measures the spatial continuity of the principal
diffusion direction. A RESTORE tensor fit gives one direction per voxel,
weighted by its fractional anisotropy (FA); voxels in the lowest 10% of FA are
not evaluated. Each evaluated voxel's direction is expressed in voxel-index
coordinates and rounded to the nearest lattice step. The voxel is counted as
connected when the direction in the voxel one step forward or one step backward
along that lattice step lies within 15° of its own. The index is the FA-weighted
fraction of evaluated voxels that are connected, between 0 and 1. It is computed
with the primary-coherence method of the ODX library.

## Header and gradient-table summaries

The table also records the image dimensions, voxel size, maximum $b$-value and
the numbers of diffusion-weighted and b=0 volumes. Loading data through
{class}`~cs_dmri.dwi.DWI` additionally reports header conditions that may affect
interpretation; see {doc}`conventions`.

## Masks

All masked metrics use the mask supplied by the caller. When no mask is given,
a fallback mask is formed from voxels whose mean b=0 signal exceeds 1% of its
maximum, and the report records `mask_source: auto-b0`. A brain mask from a
dedicated skull-stripping method is preferable.

## Input checks in the fitting tools

The fitting tools (`cs-fit`, `cs-dti`, `cs-response`, `cs-ss3t`,
`cs-ss3t-full`) report the masked NDC and contrast ratio of their input before
fitting, and print a warning when NDC is below 0.4 or the contrast ratio is
below 1.1.

## Relation to the columns previously reported by qsiprep

qsiprep's image-quality table was previously produced with DSI Studio. Metrics
whose definition differs here carry new names:

| Column | Previous column |
|---|---|
| `dimension_{x,y,z}`, `voxel_size_{x,y,z}`, `max_b` | unchanged |
| `n_dwi_volumes`, `n_b0_volumes` | `num_directions` |
| `ndc`, `ndc_masked` | `neighbor_corr`, `masked_neighbor_corr` |
| `dwi_contrast_ratio`, `dwi_contrast_ratio_masked` | `dwi_contrast` |
| `n_outlier_slices` | `num_bad_slices` |
| `fixel_coherence` | `coherence_index` |

The data dictionary records the previous name of each renamed column in a
`Replaces` field.
