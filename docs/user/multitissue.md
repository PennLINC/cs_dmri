# Single-shell three-tissue CSD

Single-shell three-tissue constrained spherical deconvolution (SS3T-CSD;
Dhollander and Connelly, 2016) decomposes data with one b=0 shell and one
diffusion-weighted shell into a white-matter fiber orientation distribution
(FOD) and isotropic gray-matter and cerebrospinal-fluid compartments. The
workflow has three steps: estimation of tissue response functions,
deconvolution, and multi-tissue intensity normalization.

```python
responses = dwi.estimate_responses()
fit = cs.SS3TModel(dwi.gtab, responses, lmax_wm=8).fit(dwi)
fit = fit.mtnormalise()
fit.to_odx("ss3t.odx")

fit = cs.ss3t_pipeline(dwi)        # the three steps in one call
```

```bash
cs-ss3t-full --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
             --odx ss3t.odx --write-responses-to responses/
```

## Response functions

Response functions are estimated with the unsupervised three-tissue method of
Dhollander et al. (2016). Voxel selection follows the staged procedure of
MRtrix3's `dwi2response dhollander`:

1. the brain mask is eroded and a signal decay metric (the mean over shells of
   the log ratio of the b=0 signal to the shell signal) is computed;
2. white matter is separated from the remainder by an FA threshold, and gray
   matter from cerebrospinal fluid by an automatic threshold on the decay metric
   (Ridgway et al., 2009);
3. each class is refined to remove partial-volume voxels;
4. final samples are drawn: cerebrospinal fluid from the highest decay-metric
   voxels, gray matter around the median decay metric, and single-fiber white
   matter from the highest-FA voxels.

FA and the principal direction come from a RESTORE tensor fit. The gray-matter
and cerebrospinal-fluid responses are isotropic; the white-matter response is
obtained by fitting zonal spherical harmonics about each voxel's principal
direction and averaging across voxels. Responses are read and written in the
MRtrix3 text format ({class}`~cs_dmri.multitissue.TissueResponse`, `cs-response`).

## Deconvolution

SS3T-CSD alternates between fitting the white-matter FOD with one isotropic
compartment and fitting both isotropic compartments with the white-matter
prediction removed, for a fixed number of iterations (default 3). Each step is a
constrained least-squares problem, with non-negative FOD amplitudes on a dense
sphere and non-negative isotropic compartments, solved with the Goldfarb–Idnani
active-set method. The white-matter FOD order (default 8) is limited by the
order of the white-matter response; it may alternatively be selected per voxel
by the Bayesian information criterion among a set of candidate orders.

## Intensity normalization

Multi-tissue normalization follows MRtrix3's `mtnormalise` (Raffelt et al.,
2017; Dhollander et al., 2021). A smooth multiplicative field, the exponential
of a polynomial (default order 3), and per-tissue balance factors are estimated
so that the balanced sum of the tissue compartments approximates a constant
($1/\sqrt{4\pi}$ by default) in inlier voxels, with outliers rejected by an
interquartile-range rule. The tissue maps are divided by the field. As in
MRtrix3, the balance factors are applied to the output only when requested
(`balanced=True`, `--balanced`).

## Outputs

The white-matter FOD is stored as real spherical-harmonic coefficients in the
MRtrix3 convention; gray matter and cerebrospinal fluid are scalar maps. The ODX
output contains the FOD coefficients, both isotropic compartments, the mask, the
response functions and FOD peaks.
