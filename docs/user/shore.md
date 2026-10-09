# 3D-SHORE reconstruction

The simple harmonic oscillator based reconstruction and estimation (3D-SHORE)
represents the diffusion signal in each voxel as a linear combination of basis
functions that factor into a radial part, built from Laguerre polynomials in the
q-space radius, and an angular part, built from even-order real spherical
harmonics (Özarslan et al., 2013; Merlet and Deriche, 2013). The basis is
determined by its radial order and a scale parameter $\zeta$. Once the
coefficients are known, the orientation distribution function (ODF), the signal
at arbitrary q-space positions and several propagator-derived scalars follow by
fixed linear or closed-form operations.

The basis definition and ordering follow the BrainSuite convention used by
qsirecon. The default radial order is 6 (72 coefficients) and the default
$\zeta$ is 700.

## Fitting

With design matrix $M$ (rows: measurements; columns: basis functions), signal
$s$ and coefficients $c$, three estimators are available.

L1 (compressed sensing)
: $\min_c \tfrac{1}{2n}\lVert Mc - s\rVert_2^2 + \alpha\lVert c\rVert_1$, solved
  with FISTA (Beck and Teboulle, 2009). The sparsity weight $\alpha$ is chosen
  per voxel by one of four strategies:

  - `l2-anchored` (default): along a logarithmic path of $\alpha$ values, the
    largest $\alpha$ whose residual sum of squares is within a factor
    $(1+\text{slack})$ of that of the L2 fit;
  - `path-bic`: the $\alpha$ on the same path that minimizes the Bayesian
    information criterion;
  - `alpha-ratio`: a fixed fraction of $\alpha_{\max}$, the smallest $\alpha$
    for which all coefficients are zero;
  - `fixed`: one $\alpha$ for all voxels.

L2
: Tikhonov regularization with separate radial and angular penalties
  ($\lambda_N$, $\lambda_L$), solved in closed form.

Non-negative ODF
: Least squares subject to non-negative ODF amplitudes on a dense sphere, solved
  with the Goldfarb–Idnani active-set method (Goldfarb and Idnani, 1983).

```python
fit = cs.ShoreModel(dwi.gtab, radial_order=6, regularization="l1",
                    alpha_mode="l2-anchored", slack=0.05).fit(dwi)
fit.r2, fit.rmse, fit.sparsity, fit.alpha
```

```bash
cs-fit --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz \
       --output coeffs.nii.gz --diagnostics
```

The diffusion times $\Delta$ and $\delta$ enter the basis through the q-values.
When they are not supplied, both are estimated from the maximum $b$-value and
the maximum gradient amplitude.

## Outputs of a fit

Coefficients
: A 4D image of coefficients with a JSON sidecar recording the basis, diffusion
  times, solver settings and the frame of the gradient directions. Python fits
  are saved in the same layout with {meth}`~cs_dmri.shore.ShoreFit.save`.

Diagnostics
: Per-voxel coefficient of determination ($R^2$), root-mean-square error,
  fraction of nonzero coefficients and, where applicable, the selected
  $\alpha$ and the BIC.

## Derived quantities

ODF
: The ODF is projected analytically onto real spherical harmonics in the
  MRtrix3 (Tournier) convention, up to an even order no greater than the radial
  order ({meth}`~cs_dmri.shore.ShoreFit.odf_sh`).

Peaks
: Local maxima of the ODF are found on a 321-direction hemisphere and refined to
  sub-vertex precision on the continuous spherical-harmonic expansion. For each
  peak the amplitude, the quantitative anisotropy (peak value minus ODF minimum)
  and a dispersion measure (the integral of the peak's lobe divided by its
  amplitude) are recorded.

Scalars
: Generalized fractional anisotropy and anisotropic power (Dell'Acqua et al.,
  2014) per voxel; return-to-origin, return-to-axis and return-to-plane
  probabilities (RTOP, RTAP, RTPP), mean squared displacement (MSD), q-space
  inverse variance (QIV) and non-Gaussianity (NG) from closed-form expressions
  in the SHORE coefficients ({meth}`~cs_dmri.shore.ShoreFit.microstructure`).
  RTAP and RTPP require a fiber direction per voxel. Voxels whose value exceeds
  a multiple of the 99th percentile are treated as fit failures and set to NaN.

Synthesis
: The signal can be evaluated for any gradient table
  ({meth}`~cs_dmri.shore.ShoreFit.predict`, `cs-synth`), for example to compare
  a fit with held-out measurements or to predict another acquisition scheme.

ODX output
: {meth}`~cs_dmri.shore.ShoreFit.to_odx` and `cs-odf` (or `cs-fit
  --odx-output`) write the ODF coefficients, peaks, scalars and diagnostics to
  an ODX file in RAS+ orientation.
