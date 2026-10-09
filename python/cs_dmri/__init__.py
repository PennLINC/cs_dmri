# SPDX-License-Identifier: MIT OR Apache-2.0
"""cs_dmri: diffusion MRI quality control, SHORE compressed-sensing fits,
RESTORE tensors and single-shell three-tissue CSD, implemented in Rust.

>>> import cs_dmri as cs
>>> dwi = cs.DWI.from_files("dwi.nii.gz", "dwi.bval", "dwi.bvec", mask="mask.nii.gz")
>>> print(dwi.qc())                      # doctest: +SKIP
>>> dwi.qc().to_dict(prefix="raw_")      # doctest: +SKIP
"""

from . import dti, multitissue, qc, shore
from ._cs_dmri import version as _version
from .dti import RestoreFit, RestoreModel
from ._spatial import SpatialWarning
from .dwi import DWI
from .gradients import GradientTable, read_bvals_bvecs
from .multitissue import (ResponseSet, SS3TFit, SS3TModel, TissueResponse, estimate_responses, mtnormalise,
                          ss3t_pipeline)
from .qc import QCReport
from .shore import ShoreFit, ShoreModel

__version__ = _version()

__all__ = [
    "DWI",
    "GradientTable",
    "QCReport",
    "ResponseSet",
    "RestoreFit",
    "RestoreModel",
    "SS3TFit",
    "SS3TModel",
    "SpatialWarning",
    "ShoreFit",
    "ShoreModel",
    "TissueResponse",
    "__version__",
    "dti",
    "estimate_responses",
    "mtnormalise",
    "multitissue",
    "qc",
    "read_bvals_bvecs",
    "shore",
    "ss3t_pipeline",
]
