#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Group ConSH on the simulated study: per-scan responses, one pooled group
# response, and every scan refit with the group response in absolute units.
#
#   bash run_consh.sh /data/consh_group/bids /data/consh_group/consh
set -euo pipefail
bids=$1
out=$2
subjects=$(cd "$bids" && ls -d sub-*)

# 1. A response for every scan. The fit made with it, on the default per-voxel
#    S/S0 scale, is kept for comparison.
for sub in $subjects; do
  dwi=$bids/$sub/dwi/${sub}_dwi
  consh --dwi "$dwi.nii.gz" --bval "$dwi.bval" --bvec "$dwi.bvec" \
        --mask "$bids/derivatives/masks/$sub/dwi/${sub}_desc-brain_mask.nii.gz" \
        --write-responses-to "$out/responses/$sub" \
        --output-wm "$out/$sub/${sub}_desc-individual_wm_fod.nii.gz" \
        --output-gm "$out/$sub/${sub}_desc-individual_gm.nii.gz" \
        --output-csf "$out/$sub/${sub}_desc-individual_csf.nii.gz" --quiet
done

# 2. One group response, pooled over all scans.
consh-response mean "$out"/responses/sub-* --output "$out/responses/group"

# 3. Every scan with the group response, in absolute signal units.
for sub in $subjects; do
  dwi=$bids/$sub/dwi/${sub}_dwi
  consh --dwi "$dwi.nii.gz" --bval "$dwi.bval" --bvec "$dwi.bvec" \
        --mask "$bids/derivatives/masks/$sub/dwi/${sub}_desc-brain_mask.nii.gz" \
        --read-responses-from "$out/responses/group" --quantitative \
        --output-wm "$out/$sub/${sub}_desc-wm_fod.nii.gz" \
        --output-gm "$out/$sub/${sub}_desc-gm.nii.gz" \
        --output-csf "$out/$sub/${sub}_desc-csf.nii.gz" --quiet
done
