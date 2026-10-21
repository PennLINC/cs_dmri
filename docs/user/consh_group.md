# Walkthrough: a group ConSH analysis

This walkthrough runs a complete group analysis with ConSH: a response for every
scan, one group response, every scan refit with it in absolute signal units, and
a comparison of fiber density between two groups. It uses a simulated
compressed-sensing DSI study with a known group difference, so each step can be
checked against the truth. The section {ref}`consh-group-real-data` lists what
changes for acquired data.

The steps take about 10 minutes on a 24-core workstation. The three scripts used
here can be downloaded: {download}`simulate.py <../examples/consh_group/simulate.py>`,
{download}`run_consh.sh <../examples/consh_group/run_consh.sh>` and
{download}`analyze.py <../examples/consh_group/analyze.py>`.

## 1. Simulate the study

The data are simulated with [TRXScan](https://github.com/PennLINC/TRXScan) from a
phantom built from one adult brain: its tissue maps and a tractogram of one
million streamlines. Eight subjects are drawn from it, each with its own random
subset of 400,000 streamlines and its own noise, and scanned with the same
compressed-sensing DSI protocol:

- a random half of the q-space points of a DSI lattice (|q| ≤ 5 lattice
  units), with b-values up to 4000 s/mm², plus six b=0 volumes: 134 volumes at
  23 distinct b-values, without shells;
- 2.5 mm isotropic voxels, and a signal-to-noise ratio of 30 for the b=0 signal
  of white matter.

In the four "patients", fiber density is reduced by 30% in the streamlines that
cross the midline through the body of the corpus callosum. Their weight is
reduced by 30%, and the white-matter fraction they account for in each voxel is
reduced by the same proportion and replaced by CSF, as in tissue loss. Because
most voxels the bundle passes through also contain other fibers, the reduction
in a voxel's total fiber density is smaller than 30% everywhere but the
midline.

```bash
pip install cs_dmri "trxscan[trx]"
trxscan-fetch sub-60501
python simulate.py study/bids --per-group 4
```

The result is a BIDS dataset with a `group` column in `participants.tsv`, a
brain mask for every scan under `derivatives/masks`, and the true white-matter
fraction under `derivatives/truth`. The simulation has no motion, distortion or
eddy currents, so the data need no preprocessing.

## 2. Check the data

A quick quality check of each scan (see {doc}`qc`) confirms that the gradient
table and the image agree before anything is fitted:

```bash
cs-qc --dwi study/bids/sub-01/dwi/sub-01_dwi.nii.gz \
      --bval study/bids/sub-01/dwi/sub-01_dwi.bval \
      --bvec study/bids/sub-01/dwi/sub-01_dwi.bvec \
      --mask study/bids/derivatives/masks/sub-01/dwi/sub-01_desc-brain_mask.nii.gz \
      --output-tsv sub-01_qc.tsv
```

```text
[cs-qc] NDC 0.8920 (masked 0.7438), DWI contrast ratio 1.2919 (masked 1.9772, good),
outlier slices 0, fixel coherence 0.6887, chain length 6.8674 mm, gradient-table ratio 1.0000
```

A gradient-table ratio of 1 means that no permutation or flip of the b-vectors
gives a more coherent fiber field than the table as given.

## 3. Estimate a response for every scan

`consh` estimates dense-b responses from each scan's single-fiber white-matter,
gray-matter and CSF voxels and writes them with `--write-responses-to`. The fit it
makes with them is on the default per-voxel S/S0 scale; it is kept here only for
comparison in step 6.

```bash
for sub in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08; do
  dwi=study/bids/$sub/dwi/${sub}_dwi
  consh --dwi $dwi.nii.gz --bval $dwi.bval --bvec $dwi.bvec \
        --mask study/bids/derivatives/masks/$sub/dwi/${sub}_desc-brain_mask.nii.gz \
        --write-responses-to study/consh/responses/$sub \
        --output-wm study/consh/$sub/${sub}_desc-individual_wm_fod.nii.gz \
        --output-gm study/consh/$sub/${sub}_desc-individual_gm.nii.gz \
        --output-csf study/consh/$sub/${sub}_desc-individual_csf.nii.gz
done
```

Each response is a table with one row per 50 s/mm² of b, from 0 to the top of
the scan's b range (83 rows here).

## 4. Pool the responses

```bash
consh-response mean study/consh/responses/sub-* --output study/consh/responses/group
```

```text
[consh-response] wm: 83 rows, lmax 8, magnitude multipliers 0.994–1.007, r_0 deviation from pool (rescaled): mean 0.16%  max 0.20%
[consh-response] gm: 83 rows, lmax 0, magnitude multipliers 0.994–1.005, r_0 deviation from pool (rescaled): mean 0.20%  max 0.27%
[consh-response] csf: 83 rows, lmax 0, magnitude multipliers 0.995–1.005, r_0 deviation from pool (rescaled): mean 0.43%  max 0.56%
```

Each scan's response is rescaled by a magnitude multiplier before averaging, as
in MRtrix3's `responsemean`. The report gives the range of multipliers and how
far each rescaled response lies from the pool. Here every scan is within 0.6% of
it. A scan far from the others usually has a poor single-fiber selection and
should be inspected before it contributes to the group response.

All scans must share the grid step (`--step`), the maximum b-value and the
white-matter SH order; `consh-response` refuses responses that do not.

## 5. Refit every scan with the group response

```bash
for sub in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08; do
  dwi=study/bids/$sub/dwi/${sub}_dwi
  consh --dwi $dwi.nii.gz --bval $dwi.bval --bvec $dwi.bvec \
        --mask study/bids/derivatives/masks/$sub/dwi/${sub}_desc-brain_mask.nii.gz \
        --read-responses-from study/consh/responses/group --quantitative \
        --output-wm study/consh/$sub/${sub}_desc-wm_fod.nii.gz \
        --output-gm study/consh/$sub/${sub}_desc-gm.nii.gz \
        --output-csf study/consh/$sub/${sub}_desc-csf.nii.gz
done
```

`--read-responses-from` skips the tensor fit and tissue selection and uses the
same responses for every scan. `--quantitative` writes the fODFs in absolute
signal units, so that fiber density is proportional to the signal rather than to
the voxel's b=0 signal, and the final normalization removes the bias field and
the remaining difference in global scale between scans (see {doc}`consh`).

Steps 3 to 5 are also the script {download}`run_consh.sh
<../examples/consh_group/run_consh.sh>`. In Python they are:

```python
import cs_dmri as cs

dwis = [cs.DWI.from_files(f"{b}.nii.gz", f"{b}.bval", f"{b}.bvec", mask=m) for b, m in scans]
group = cs.DenseBResponses.pool([cs.estimate_dense_b_responses(d) for d in dwis])
fits = [cs.consh_pipeline(d, responses=group, quantitative=True) for d in dwis]
```

## 6. Compare the groups

Fiber density in a voxel is taken here as the l=0 coefficient of the white-matter
fODF, which is proportional to its integral. Because the simulated subjects share
one anatomy and do not move, voxels correspond across subjects without
registration, and the group difference can be read voxel by voxel. The script
compares it with the true change in white-matter fraction:

```bash
python analyze.py study/bids study/consh figures/
```

```{figure} consh_group/consh_group_maps.png
:alt: Three axial maps of fiber density change between patients and controls.

Change in fiber density between the groups in an axial slice through the body
of the corpus callosum, shown in white matter: the planted change (left), the
change measured with the group response in absolute units (middle), and with
each scan's own response on the per-voxel S/S0 scale (right).
```

```{figure} consh_group/consh_group_scatter.png
:alt: Scatter plot of estimated against true fiber density change.
:width: 70%

Estimated against true change in the 4304 white-matter voxels where the planted
reduction is at least 1%.
```

With the group response in absolute units, the measured change follows the
planted one: in the affected voxels the mean change is −6.3% for a true −5.8%,
with a slope of 1.00 and a correlation of 0.89 between estimated and true
change. In unaffected white matter the median change is −0.04%, and 95% of
voxels lie within about ±6%, the noise of a four-against-four comparison.

On the per-voxel S/S0 scale the measured loss is about twice the true one
(mean −14.7%, slope 1.98). The tissue lost in the patients is replaced by CSF,
which raises the voxel's b=0 signal, so a fiber density expressed as a fraction
of that signal falls further than the fibers themselves. This is why
quantitative comparisons use absolute signal units and a common response.

(consh-group-real-data)=
## Acquired data

For an acquired study, the steps above apply after these changes:

- **Preprocessing.** Run the scans through a preprocessing pipeline such as
  qsiprep, which provides the corrected series, a brain mask and, when a
  gradient-nonlinearity correction is applied, a gradient-deviation field. Pass
  that field to every `consh` run with `--graddev`; the responses estimated with
  it cover the extended b range.
- **One protocol.** Fiber density depends on which b-values contribute, so all
  scans should use the same acquisition, and they must share the maximum
  b-value and grid step for their responses to be pooled.
- **Registration and fixels.** Subjects differ in anatomy, so the comparison is
  made after registration to a study template and, for fixel-based analysis,
  after segmenting the fODFs into fixels. The MRtrix3 fixel-based analysis
  steps (`population_template`, `mrregister`, `fod2fixel`, `fixelcorrespondence`,
  `fixelcfestats`) read the white-matter fODF NIfTI images from step 5
  (Dhollander, Clemente et al., 2021).
- **Inspection.** Check the quality report of every scan, and the spread of the
  per-scan responses in step 4, before pooling.
