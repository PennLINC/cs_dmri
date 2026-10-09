# SPDX-License-Identifier: MIT OR Apache-2.0
"""cs-dmri: diffusion MRI quality control, SHORE compressed-sensing fits,
RESTORE tensors and single-shell three-tissue CSD, implemented in Rust.

>>> import cs_dmri as cs
>>> dwi = cs.DWI.from_files("dwi.nii.gz", "dwi.bval", "dwi.bvec", mask="mask.nii.gz")
>>> print(dwi.qc())                      # doctest: +SKIP
>>> dwi.qc().to_dict(prefix="raw_")      # doctest: +SKIP
"""

from . import dti, qc
from ._cs_dmri import version as _version
from .dti import RestoreFit, RestoreModel
from .dwi import DWI
from .gradients import GradientTable, read_bvals_bvecs
from .qc import QCReport

__version__ = _version()

__all__ = [
    "DWI",
    "GradientTable",
    "QCReport",
    "RestoreFit",
    "RestoreModel",
    "__version__",
    "dti",
    "qc",
    "read_bvals_bvecs",
]
