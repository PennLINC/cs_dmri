# BrainSuiteSHORE microstructure scalars — derivation notes

These are the closed-form expressions used by `cs_dmri::scalars::microstructure` and
the matching unit tests. The goal is to reuse the published dipy formulas (which
were validated against analytical multi-tensor truth) without re-deriving them
from scratch.

## 1. Basis equivalence

BrainSuiteSHORE (cs-dmri / qsirecon) and dipy isotropic MAPMRI use the *same*
spherical-SHORE family — generalized Laguerre × Gaussian × real spherical
harmonic — with two cosmetic differences: the parameterization of the radial
scale and the truncation of the radial index set.

### BrainSuiteSHORE radial signal basis

$$
R^{\mathrm{BS}}_{n\ell}(q) \;=\; \kappa^{\mathrm{BS}}_{n\ell}\,
   \Bigl(\tfrac{q^2}{\zeta}\Bigr)^{\ell/2}\,
   e^{-q^2/(2\zeta)}\,
   L_{n-\ell}^{\ell+1/2}\!\bigl(\tfrac{q^2}{\zeta}\bigr)
\qquad\text{with}\qquad
\kappa^{\mathrm{BS}}_{n\ell} \;=\; \sqrt{\dfrac{2(n-\ell)!}{\zeta^{3/2}\,\Gamma(n+3/2)}}.
$$

Iteration: $n = 0,1,\ldots,N$, $\ell$ even with $0 \le \ell \le n$,
$m = -\ell,\ldots,\ell$. Real-SH ordering is BrainSuite's
("cosine block, m=0, sine block").

### dipy isotropic MAPMRI radial signal basis (Özarslan 2013 eq 61)

$$
\phi^{\mathrm{iso}}_{j\ell}(q) \;=\; (-1)^{\ell/2}\,\sqrt{4\pi}\,
   \bigl(2\pi^2\mu^2 q^2\bigr)^{\ell/2}\,
   e^{-2\pi^2\mu^2 q^2}\,
   L_{j-1}^{\ell+1/2}\!\bigl(4\pi^2\mu^2 q^2\bigr).
$$

Setting

$$\boxed{\;\mu \;=\; \dfrac{1}{2\pi\sqrt{\zeta}}\;\;\Longleftrightarrow\;\;\zeta \;=\; \dfrac{1}{4\pi^2\mu^2}\;}$$

makes the Laguerre argument $4\pi^2\mu^2 q^2 = q^2/\zeta$ and the Gaussian
envelope $e^{-2\pi^2\mu^2 q^2} = e^{-q^2/(2\zeta)}$ identical between the two
bases.

### Per-mode conversion factor

Identifying the Laguerre orders ($j-1 = n-\ell$, i.e. $j = n-\ell+1$),

$$
\phi^{\mathrm{iso}}_{j\ell}(q) \;=\; \alpha_{n\ell}\, R^{\mathrm{BS}}_{n\ell}(q),
\qquad
\alpha_{n\ell} \;=\;
   (-1)^{\ell/2}\,
   \sqrt{\dfrac{2\pi\,\zeta^{3/2}\,\Gamma(n+3/2)}{2^{\ell}\,(n-\ell)!}}.
$$

If the same diffusion signal is expanded in both bases,

$$
c^{\mathrm{iso}}_{j\ell m_{\text{Desc}}}
   \;=\;
   \dfrac{1}{\alpha_{n\ell}}\,
   \sum_{m_{\text{BS}}} P^{(\ell)}_{m_{\text{BS}}, m_{\text{Desc}}}\,
   c^{\mathrm{BS}}_{n\ell m_{\text{BS}}},
$$

where $P^{(\ell)}$ is the orthogonal $(2\ell{+}1)\times(2\ell{+}1)$ permutation
that takes BrainSuite's real-SH ordering to Descoteaux's. Because $P^{(\ell)}$
is orthogonal and we always evaluate the angular part *in the same basis as
the coefficients*,

$$
\sum_{m_{\text{Desc}}} c^{\mathrm{iso}}_{j\ell m_{\text{Desc}}}\,
   Y^{\text{Desc}}_{\ell m_{\text{Desc}}}(\hat u)
\;=\;
\dfrac{1}{\alpha_{n\ell}}
\sum_{m_{\text{BS}}} c^{\mathrm{BS}}_{n\ell m_{\text{BS}}}\,
   Y^{\text{BS}}_{\ell m_{\text{BS}}}(\hat u).
$$

So we never need the explicit BS ↔ Desc permutation: we just dot the
BrainSuite SH block (`brainsuite_sh_block(ell, theta, phi)`) against the
BrainSuite coefficient block.

### Truncation difference

For radial order $N$:

| Basis | per-$\ell$ Laguerre orders | #coeffs (N=6) |
|-------|----------------------------|---------------|
| dipy SHORE | $0, 1, \ldots, (N-\ell)/2$ | 50 |
| dipy iso-MAPMRI | $0, 1, \ldots, (N-\ell)/2$ | 50 |
| BrainSuiteSHORE | $0, 1, \ldots, N-\ell$ | 72 |

BrainSuiteSHORE is a strict superset — every dipy mode is a BrainSuite mode,
plus extra higher-Laguerre-order modes per $\ell$. Each extra mode contributes
to a scalar via the *same* closed-form kernel (the dipy formula was derived as
an integral identity that's valid for arbitrary non-negative Laguerre order $j-1$,
independent of any artificial truncation). So we evaluate the dipy kernel at
$j = n-\ell+1$ for *every* BrainSuite mode and sum.

## 2. Signal normalization

The dipy `_mapmri_coef` is divided by $\sum_i c_i B_i$ (line 346, 453 of
`dipy/reconst/mapmri.py`) so that the predicted propagator integrates to 1
(equivalently, predicted $E(0)=1$). cs-dmri does *not* normalize the DWI
signal before fitting, so before any scalar evaluation we must divide the
coefficient block by the predicted $E(0)$:

$$
\hat E(0)
\;=\;
\sum_{n} c^{\mathrm{BS}}_{n,0,0}\,
   R^{\mathrm{BS}}_{n,0}(0)\,
   Y^{\mathrm{BS}}_{0,0}(\theta,\phi).
$$

Since $Y^{\mathrm{BS}}_{0,0} = 1/\sqrt{4\pi}$ and only $\ell=0$ modes survive
at $q=0$,

$$
\hat E(0)
\;=\;
\dfrac{1}{\sqrt{4\pi}}
   \sum_{n=0}^{N}
   c^{\mathrm{BS}}_{n,0,0}\,
   \kappa^{\mathrm{BS}}_{n,0}\,
   L_n^{1/2}(0),
\qquad
L_n^{1/2}(0) \;=\; \dfrac{\Gamma(n+3/2)}{n!\,\Gamma(3/2)}.
$$

`cs_dmri::scalars::microstructure::predict_e0` implements this; every other
scalar receives `coefs / e0`.

## 3. Scalar formulas

Let $\tilde c$ denote the normalized coefficient vector (`coefs / e0`). All
indices below run over BrainSuite ordering with $j(n,\ell) = n-\ell+1$.

### RTOP

From `dipy/reconst/shore.py:382-403`, with the radial sum widened to the
full BrainSuite range:

$$
\text{RTOP}
\;=\;
\sum_{n=0}^{N} \tilde c_{n,0,0}\,(-1)^n\,
\sqrt{\dfrac{16\pi\,\zeta^{3/2}\,\Gamma(n+3/2)}{n!}}.
$$

This is identical to evaluating the dipy iso-MAPMRI isotropic-RTOP kernel at
every BrainSuite $(n,0,0)$ mode (verified algebraically — the two derivations
agree exactly).

### MSD

From `dipy/reconst/shore.py:428-463`:

$$
\text{MSD}
\;=\;
\sum_{n=0}^{N} \tilde c_{n,0,0}\,(-1)^n\,
\sqrt{\dfrac{9\,\Gamma(n+3/2)}{8\pi^6\,\zeta^{7/2}\,n!}}\,
{}_2F_1(-n,\tfrac{5}{2};\tfrac{3}{2};2).
$$

The hypergeometric series terminates after $n+1$ terms (first argument is a
non-positive integer). `cs_dmri::math::hyp2f1` already implements the series.

### RTAP (returns 0 if no principal direction)

From `dipy/reconst/mapmri.py:627-679`, with $j = n-\ell+1$ (any non-negative
Laguerre order). Per-mode kernel:

$$
\kappa^{\text{RTAP}}_{j\ell}
\;=\;
\dfrac{2\,(-1)^{j-1}\,2^{-(\ell+3)/2}}{\pi}
\sum_{k=0}^{j-1}
   \dfrac{(-1)^k\,\binom{j+\ell-1/2}{j-k-1}\,\Gamma(\tfrac{\ell+1}{2}+k)}
         {k!\,\bigl(\tfrac{1}{2}\bigr)^{(\ell+1)/2 + k}}.
$$

(The `binomialfloat(a, k) = Γ(a+1) / (Γ(k+1) Γ(a-k+1))` of dipy is
real-valued.) Then

$$
\text{RTAP}
\;=\;
\dfrac{1}{\mu^2}
\sum_{n,\ell}
   \dfrac{\kappa^{\text{RTAP}}_{n-\ell+1,\,\ell}}{\alpha_{n,\ell}}
\sum_{m_{\text{BS}}} \tilde c_{n\ell m_{\text{BS}}}\,
   Y^{\text{BS}}_{\ell m_{\text{BS}}}(\hat u_1),
\qquad
\dfrac{1}{\mu^2} = 4\pi^2\zeta.
$$

### RTPP

From `dipy/reconst/mapmri.py:571-625`. Per-mode kernel:

$$
\kappa^{\text{RTPP}}_{j\ell}
\;=\;
\dfrac{(-1/2)^{\ell/2}}{\sqrt{\pi}}
\sum_{k=0}^{j-1}
   \dfrac{(-1)^k\,\binom{j+\ell-1/2}{j-k-1}\,\Gamma(\tfrac{\ell}{2}+k+\tfrac{1}{2})}
         {k!\,\bigl(\tfrac{1}{2}\bigr)^{\ell/2 + 1/2 + k}}.
$$

$$
\text{RTPP}
\;=\;
\dfrac{1}{\mu}
\sum_{n,\ell}
   \dfrac{\kappa^{\text{RTPP}}_{n-\ell+1,\,\ell}}{\alpha_{n,\ell}}
\sum_{m_{\text{BS}}} \tilde c_{n\ell m_{\text{BS}}}\,
   Y^{\text{BS}}_{\ell m_{\text{BS}}}(\hat u_1),
\qquad
\dfrac{1}{\mu} = 2\pi\sqrt{\zeta}.
$$

### QIV

QIV (Hosseinbor 2013) is defined as the inverse of the q-space second moment:

$$
\text{QIV} \;=\; \dfrac{1}{\displaystyle\int E(q)\,|q|^2\,d^3q}.
$$

The dipy iso-MAPMRI form (`dipy/reconst/mapmri.py:754-797`) bakes a clever
linear-in-coefficient expression that is correct for the dipy iso basis subset
but anti-correlates with truth when extended to BrainSuite's larger mode set
(the alternating signs interact destructively with extra odd-Laguerre modes —
verified empirically in the validation harness, which saw $r=-0.92$).

We instead implement the direct integral. Only $\ell=0$ modes contribute
(angular orthogonality kills $\ell>0$). For each `(n, 0, 0)` mode the radial
integral against $|q|^2 \cdot q^2\,dq$ is a closed-form Laguerre integral
(Gradshteyn 7.414.4). The result:

$$
\int E(q)|q|^2\,d^3q
\;=\;
6\sqrt{2\pi}\,\zeta^{5/2}
\sum_{n=0}^{N}
   \tilde c_{n,0,0}\,
   \kappa^{\mathrm{BS}}_{n,0}\,
   \dfrac{\Gamma(n+3/2)}{n!}\,
   {}_2F_1\!\bigl(-n,\tfrac{5}{2};\tfrac{3}{2};2\bigr).
$$

Then $\text{QIV} = 1/\bigl(\text{this integral}\bigr)$. The validation harness
recovers $r=+0.99$, median relative error $\approx 9\%$ against analytical
truth on the single-fiber sweep — comparable to dipy iso-MAPMRI's QIV
($r=+1.00$) and far better than the failed linear-extension approach.

### NG

From `dipy/reconst/mapmri.py:799-824`:

$$
\text{NG} \;=\; \sqrt{1 \;-\; \dfrac{\tilde c_{0,0,0}^{\,2}}{\sum_i \tilde c_i^{\,2}}}.
$$

This is well-defined regardless of basis ordering. The $(0,0,0)$ mode is the
isotropic Gaussian; $\tilde c_{0,0,0}$ is the energy in that mode.

### NG∥ / NG⊥

Skipped in v1 — these require an anisotropic-Hermite indexing
$(n_1,n_2,n_3)$ that has no direct counterpart in the spherical SHORE basis.
A future version could project $\tilde c$ onto a tensor frame first.

### PA

PA was originally formulated as a change of basis between an *anisotropic*
MAPMRI fit and an isotropic SHORE fit. cs-dmri only carries the isotropic
SHORE fit, so the PA closed form is degenerate (the change-of-basis is the
identity and PA collapses to NG-style energy ratios). Skipped in v1 — adding
PA would require fitting the anisotropic MAPMRI variant alongside, which is
out of scope for this change.

## 4. Unit conventions: cs-dmri vs TORTOISE

cs-dmri inherits dipy's q-space convention: b-values in s/mm², deltas in
seconds, so $q = \sqrt{b/(4\pi^2\tau)}$ comes out in **1/mm**. Every length-
density scalar therefore carries an mm-based unit:

| scalar | dim | cs-dmri / dipy units (default `--scalar-units mm`) | TORTOISE units (default `--scalar-units um`) | conversion |
|--------|-----|------------------------------------------|----------------------------------|--------------|
| RTOP | $1/L^3$ | mm⁻³ | μm⁻³ | × $10^{-9}$ |
| RTAP | $1/L^2$ | mm⁻² | μm⁻² | × $10^{-6}$ |
| RTPP | $1/L^1$ | mm⁻¹ | μm⁻¹ | × $10^{-3}$ |
| MSD  | $L^2$   | mm²  | μm²  | × $10^{+6}$ |
| QIV  | $L^5$   | mm⁵  | μm⁵  | × $10^{+15}$ |
| NG   | dimless | —    | —    | × 1 |

TORTOISE divides its B-matrix by 1000 (`MAPMRIModel.cxx:890`) before
computing q, which converts s/mm² to ms/μm² and lands its q in 1/μm. This is
purely an internal unit choice — the underlying physics is identical.

For typical brain WM (e.g., $\lambda_1 = 1.5\times10^{-3}$ mm²/s, $\tau \approx
0.04$ s) the analytical RTOP is ~$2 \times 10^5$ mm⁻³, which equals
$2 \times 10^{-4}$ μm⁻³ — the latter is what TORTOISE viewers display as
"in the [0, ~few-percent] range".

`cs-odf --microstructure --scalar-units um` (the default) emits TORTOISE-
compatible magnitudes; `--scalar-units mm` keeps the dipy/cs-dmri-internal
convention for cross-checking against dipy's `MapmriModel.fit().rtop()` etc.

## 5. Sanity checks

* For a pure Gaussian propagator with isotropic diffusivity $D$ and effective
  diffusion time $\tau$, the analytical RTOP is
  $1/\bigl(4\pi D\tau\bigr)^{3/2}$. The Rust unit tests fit
  BrainSuiteSHORE to a synthetic single-tensor signal with this $D, \tau$ and
  assert the computed RTOP matches within 5%.
* For a single-fiber tensor with eigenvalues $\lambda_1, \lambda_2, \lambda_3$,
  ground truth: $\text{RTAP} = 1/(4\pi\sqrt{\lambda_2\lambda_3}\,\tau)$,
  $\text{RTPP} = 1/(2\sqrt{\pi\lambda_1\tau})$,
  $\text{MSD} = 2(\lambda_1+\lambda_2+\lambda_3)\tau$. Validation script
  `scripts/validate_microstructure.py` runs the full sweep and reports
  Pearson r vs analytical truth and vs dipy iso-MAPMRI.
