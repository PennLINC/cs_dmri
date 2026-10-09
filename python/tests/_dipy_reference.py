# Reference implementations for parity tests, adapted from dipy.
#
# find_qspace_contrast / dwi_contrast are from dipy pull request #4224
# (https://github.com/dipy/dipy/pull/4224, head dc32011), which was not yet in
# a dipy release. Copyright (c) 2008-2026, dipy developers; BSD 3-Clause
# licence, reproduced in LICENSE-DIPY at the repository root.
import numpy as np


def find_qspace_contrast(bvals, bvecs, b0_threshold=50):
    b0s_mask = bvals <= b0_threshold
    dwi_indices = np.flatnonzero(~b0s_mask)
    qvecs = np.sqrt(bvals)[:, np.newaxis] * bvecs
    out = []
    for dwi_index in dwi_indices:
        qvec = qvecs[dwi_index]
        qvec_norm_sq = np.dot(qvec, qvec)
        qvec_norm = np.sqrt(qvec_norm_sq)
        min_distance, contrast_index = np.inf, None
        for candidate_index in dwi_indices:
            if candidate_index == dwi_index:
                continue
            candidate = qvecs[candidate_index]
            parallel = qvec * (np.dot(qvec, candidate) / qvec_norm_sq)
            perpendicular = candidate - parallel
            if np.allclose(perpendicular, 0.0):
                continue
            perpendicular *= qvec_norm / np.linalg.norm(perpendicular)
            distance = np.linalg.norm(candidate - perpendicular)
            if distance < min_distance:
                min_distance, contrast_index = distance, candidate_index
        out.append((dwi_index, contrast_index))
    return out


def dwi_contrast(data, neighbor_indices, contrast_indices, mask=None):
    n_corr, c_corr = [], []
    for (i, n), (_, c) in zip(neighbor_indices, contrast_indices):
        if mask is not None:
            a, b, d = data[..., i][mask], data[..., n][mask], data[..., c][mask]
        else:
            a, b, d = data[..., i].ravel(), data[..., n].ravel(), data[..., c].ravel()
        n_corr.append(np.corrcoef(a, b)[0, 1])
        c_corr.append(np.corrcoef(a, d)[0, 1])
    return np.mean(n_corr) / np.mean(c_corr)
