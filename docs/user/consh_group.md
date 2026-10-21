# Group analysis with ConSH

Fiber densities are comparable across subjects when every scan is fitted with
the same response functions in absolute signal units (see {doc}`consh`). This
takes three passes over the preprocessed scans: estimate a response for every
scan, pool them into one group response, and refit every scan with the group
response.

The commands below assume one preprocessed series and brain mask per subject,
for example from qsiprep:

```bash
subjects="sub-01 sub-02 sub-03"
prep=derivatives/qsiprep
out=derivatives/consh
```

## 1. A response for every scan

```bash
for sub in $subjects; do
  dwi=$prep/$sub/dwi/${sub}_space-ACPC_desc-preproc_dwi
  consh --dwi $dwi.nii.gz --bval $dwi.bval --bvec $dwi.bvec \
        --mask $prep/$sub/dwi/${sub}_space-ACPC_desc-brain_mask.nii.gz \
        --write-responses-to $out/responses/$sub \
        --odx $out/$sub/${sub}_desc-individual.odx
done
```

`--write-responses-to` writes the scan's dense-b responses
(`{wm,gm,csf}_response_denseb.txt`). The fit made with them, on the default
per-voxel S/S0 scale, is not used further. When a gradient-deviation field is
available, pass it with `--graddev` here and in step 3.

## 2. Pool the responses

```bash
consh-response mean $out/responses/sub-* --output $out/responses/group
```

The report gives, for each tissue, the range of per-scan magnitude multipliers
and how far each rescaled response lies from the pool. A scan far from the
others usually has a poor single-fiber selection and should be inspected before
it contributes; `--statistic median` resists a single outlying scan. All scans
must share the grid step, the maximum b-value and the white-matter SH order.

## 3. Refit every scan with the group response

```bash
for sub in $subjects; do
  dwi=$prep/$sub/dwi/${sub}_space-ACPC_desc-preproc_dwi
  consh --dwi $dwi.nii.gz --bval $dwi.bval --bvec $dwi.bvec \
        --mask $prep/$sub/dwi/${sub}_space-ACPC_desc-brain_mask.nii.gz \
        --read-responses-from $out/responses/group --quantitative \
        --output-wm $out/$sub/${sub}_desc-wm_fod.nii.gz \
        --output-gm $out/$sub/${sub}_desc-gm.nii.gz \
        --output-csf $out/$sub/${sub}_desc-csf.nii.gz
done
```

`--quantitative` writes the fODFs in absolute signal units, and the final
normalization removes the bias field and the remaining global scale. The
white-matter fODF images use the MRtrix3 SH basis and can be passed to the
MRtrix3 fixel-based analysis and SIFT2 tools. `--odx` writes one ODX file per
scan in place of the three NIfTI images.

## In Python

```python
import cs_dmri as cs

dwis = [cs.DWI.from_files(f"{b}.nii.gz", f"{b}.bval", f"{b}.bvec", mask=m) for b, m in scans]
group = cs.DenseBResponses.pool([cs.estimate_dense_b_responses(d) for d in dwis])
group.save("derivatives/consh/responses/group")
for sub, dwi in zip(subjects, dwis):
    fit = cs.consh_pipeline(dwi, responses=group, quantitative=True)
    fit.to_odx(f"derivatives/consh/{sub}/{sub}_desc-group.odx")
```

`DenseBResponses.save` and `DenseBResponses.load` use the same directory layout
as the command-line tools, so the passes can be split between them.
