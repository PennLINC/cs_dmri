# Continuous-b multi-tissue CSD (ConSH)

Multi-shell multi-tissue constrained spherical deconvolution (MSMT-CSD;
Jeurissen et al., 2014) estimates a white-matter fiber orientation distribution
(fODF) together with isotropic gray-matter and CSF compartments, using one
response function per tissue and per b-value shell. Acquisitions that do not
sample q-space on shells, such as Cartesian (DSI-like) and compressed-sensing
DSI schemes, have many distinct b-values with few directions each, and rounding
them into shells discards information.

ConSH (continuous-b spherical harmonic deconvolution) removes the need for
shells. The tissue responses are defined on a dense grid of b-values
(`0, step, 2·step, …`), each measurement is modeled with the response at its
own b-value, and all tissues are fit jointly in every voxel. On a shelled
acquisition it reduces exactly to MSMT-CSD. Deconvolution of non-shelled data
against responses defined continuously in b was also described independently
by Morez et al. (2021), whose parametric response model is available as one of
the response estimators below.

```python
fit = cs.consh_pipeline(dwi)                 # responses, deconvolution, normalization
fit.export("consh.odx")                      # or .fz, .pam5, .mif, a fixel directory, ...

responses = cs.estimate_dense_b_responses(dwi)
fit = cs.ConshModel(dwi.gtab, responses).fit(dwi).mtnormalise()
```

```bash
consh --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
      --odx consh.odx --write-responses-to responses/
```

## Acquisition requirements

Separating three tissues requires diffusion data at three or more distinct
b-values, b=0 included, as MRtrix3's `msmt_csd` requires at least as many shells
as tissues. b-values within 50 s/mm² of each other count as one. Data with b=0
and a single b-value cannot separate three tissues, and ConSH stops with an
error; such data are analyzed with {doc}`single-shell three-tissue CSD
<multitissue>`.

## Dense-b response functions

Single-fiber white-matter, gray-matter and CSF voxels are selected as for SS3T
(Dhollander et al., 2016; see {doc}`multitissue`), from a RESTORE tensor fit.
Their signals are divided by one per-scan reference, the median b=0 signal of
the single-fiber white-matter voxels, which removes receiver gain while keeping
the relative amplitudes of the tissues (the CSF b=0 signal is about 2.6 times
that of white matter), as MRtrix3's `amp2response` does. The rows of the
dense-b responses are then estimated from the pooled samples with one of three
estimators:

`kernel`
: Each row is fit on its own from the samples within a Gaussian window in b
  (width `sigma_b`, default 150 s/mm²). On shelled data the rows match
  per-shell responses; on sparse non-shelled schemes the rows between sampled
  b-values have little support.

`smooth`
: All rows are fit jointly, with the response interpolated linearly in b
  between rows, a smoothness penalty across rows, and constraints that hold at
  any SH order: the spherical mean is non-negative and does not increase with b,
  and the white-matter zonal coefficients alternate in sign. Suited to sparse
  non-shelled sampling and to gradient-deviated b-values.

`dki`
: The parametric response of Morez et al. (2021): an axially symmetric
  diffusion-kurtosis signal plus a constant offset (seven parameters for white
  matter, four for gray matter and CSF), fit to all samples at once and
  constrained not to increase with b. It needs three or more distinct non-zero
  b-values.

`auto` (the default) chooses between `kernel` and `smooth`: with a
gradient-deviation field, `smooth` when the field shifts b-values by more than
`sigma_b` (95th percentile over the mask); without one, `smooth` when the fit
interpolates in b and `kernel` otherwise.

## Deconvolution

Each voxel's signal, divided by its own mean b=0 signal, is fit by
inequality-constrained least squares with the white-matter fODF amplitude
constrained non-negative on a dense set of directions and the gray-matter and
CSF amplitudes constrained non-negative. Each measurement is modeled with the
response row nearest its b-value, or with linear interpolation between the two
bracketing rows (`b_interp`, `--b-interp`).

A fourth, constant compartment absorbs the magnitude noise floor (on by
default; `noise_floor=False`, `--no-noise-floor`). The noise floor is additive
in absolute signal units while the tissue model is multiplicative, so in voxels
with a much lower b=0 signal than the response voxels (cisternal and
subarachnoid CSF, non-brain tissue inside the mask) the tissue responses cannot
reproduce it, and without this compartment the excess is assigned to the
white-matter fODF.

An optional L2 penalty on the white-matter coefficients with l ≥ 2 stabilizes
peaks in crossing regions: with one weight for all voxels (`wm_ridge`), or with
the weight chosen per voxel as the largest that keeps the residual within a
given fraction of the unpenalized fit (`wm_ridge_anchor`).

## Gradient nonlinearity

A gradient-deviation field (a 9-component image on the DWI grid, such as
qsiprep's `*_graddev.nii.gz`, which stores `T`, or an HCP/FSL `grad_dev.nii.gz`,
which stores `T − I`) gives every voxel its own effective gradient directions
and b-values. With a field, the tensor fit, the response estimation and the
deconvolution all use the per-voxel tables, the response is interpolated in b,
and the dense-b grid is extended to the largest effective b-value. This corrects
the direction-dependent b-value error that rotating a fitted fODF afterwards
cannot. Responses read from disk must cover the extended grid.

## Normalization and outputs

The final step is the multi-tissue intensity normalization of MRtrix3's
`mtnormalise` (see {doc}`multitissue`). The noise-floor compartment is not
part of the tissue sum but is divided by the same bias field, so it stays on
the scale of the gray-matter and CSF amplitudes.

The `consh` tool writes the white-matter fODF (SH coefficients in the MRtrix3
basis) and the gray-matter and CSF maps as NIfTI images, with the noise floor
as a `_noisefloor.nii.gz` image next to the white-matter output, or one ODX
file. The ODX header records the dense-b responses, the grid step and a
BIDS-style description of how the fit was made, including whether the output is
in absolute signal units and the reference b=0 signal (`ReferenceB0Signal`): the
median b=0 in a white-matter mask, the single-fiber voxels, or voxels where the
fitted white-matter fraction exceeds 0.7, in that order of preference. From
Python, fits can also be exported to DSI Studio, dipy and MRtrix3 formats (see
{doc}`export`).

## Quantitative use

Fiber density (the apparent fiber density of Raffelt et al., 2012) is
comparable across subjects only when the fODF amplitude is proportional to the
signal and the same response functions are used for every subject (Dhollander,
Clemente et al., 2021; Smith et al., 2022). By default ConSH fits each voxel's
signal divided by its own b=0 signal, which makes fiber density a fraction of
the voxel's signal. With `quantitative=True` (`--quantitative`) the fitted
coefficients are multiplied back by the voxel's b=0 signal. This is equivalent
to deconvolving the absolute signal, while keeping the ratio of penalty to data
constant across voxels. It requires the final normalization, which then removes
the smooth bias field and the remaining global scale.

For a study, a group-average response is estimated in three passes:

```bash
# 1. responses for every scan
for s in sub-*; do
  consh --dwi $s/dwi.nii.gz --bval $s/dwi.bval --bvec $s/dwi.bvec --mask $s/mask.nii.gz \
        --write-responses-to responses/$s --odx $s/consh_individual.odx
done
# 2. pooled response
consh-response mean responses/sub-* --output responses/group
# 3. every scan with the group response
for s in sub-*; do
  consh --dwi $s/dwi.nii.gz --bval $s/dwi.bval --bvec $s/dwi.bvec --mask $s/mask.nii.gz \
        --read-responses-from responses/group --quantitative \
        --output-wm $s/wm_fod.nii.gz --output-gm $s/gm.nii.gz --output-csf $s/csf.nii.gz
done
```

```python
group = cs.DenseBResponses.pool([cs.estimate_dense_b_responses(d) for d in dwis])
fits = [cs.consh_pipeline(d, responses=group, quantitative=True) for d in dwis]
```

Pooling follows MRtrix3's `responsemean`: each scan's response is first rescaled
by the geometric mean over rows of the ratio of the group's mean l=0 term to its
own, which compensates differences in b=0 contrast between scans, and the
rescaled responses are then averaged (or their median taken). The deviation of
each scan from the pool is reported; a scan far from the others usually has a
poor single-fiber selection. All scans must share the grid step, the maximum
b-value and the white-matter SH order. Because ConSH uses every b-value, fiber
densities from acquisitions with different maximum b-values are not on the same
scale even with the same response, so a study should use one protocol.

Further points for quantitative work:

- Keep the normalization target fixed (the default 1/√(4π)) rather than the
  median, so that the scale is the same for every subject.
- The noise-floor compartment removes signal from the white-matter fODF by
  design. Where conventional three-tissue fractions are needed, renormalize the
  three tissues to sum to the target or fit without the compartment.
- MRtrix3's fixel-based analysis and SIFT2 tools read the NIfTI outputs, or the
  white-matter fODF exported as an MRtrix3 SH image.
- Peak thresholds given as a fraction of the median dominant-peak amplitude
  (`--peak-min-amplitude-frac`) mean the same with and without absolute-scale
  output, unlike absolute thresholds.
