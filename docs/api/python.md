# Python API

## Series and gradients

```{eval-rst}
.. autoclass:: cs_dmri.DWI
   :members:

.. autoclass:: cs_dmri.GradientTable
   :members:

.. autofunction:: cs_dmri.read_bvals_bvecs

.. autoclass:: cs_dmri.SpatialWarning
```

## Quality control

```{eval-rst}
.. autoclass:: cs_dmri.qc.QCReport
   :members:

.. autofunction:: cs_dmri.qc.assess
.. autofunction:: cs_dmri.qc.neighboring_dwi_correlation
.. autofunction:: cs_dmri.qc.dwi_contrast_ratio
.. autofunction:: cs_dmri.qc.outlier_slices
.. autofunction:: cs_dmri.qc.fixel_coherence
.. autofunction:: cs_dmri.qc.columns
.. autofunction:: cs_dmri.qc.column_descriptions
```

## Diffusion tensor

```{eval-rst}
.. autoclass:: cs_dmri.dti.RestoreModel
   :members:

.. autoclass:: cs_dmri.dti.RestoreFit
   :members:
```

## 3D-SHORE

```{eval-rst}
.. autoclass:: cs_dmri.shore.ShoreModel
   :members:

.. autoclass:: cs_dmri.shore.ShoreFit
   :members:
```

## Three-tissue CSD

```{eval-rst}
.. autoclass:: cs_dmri.multitissue.TissueResponse
   :members:

.. autoclass:: cs_dmri.multitissue.ResponseSet
   :members:

.. autofunction:: cs_dmri.multitissue.estimate_responses

.. autoclass:: cs_dmri.multitissue.SS3TModel
   :members:

.. autoclass:: cs_dmri.multitissue.SS3TFit
   :members:

.. autofunction:: cs_dmri.multitissue.mtnormalise
.. autofunction:: cs_dmri.multitissue.ss3t_pipeline
```
