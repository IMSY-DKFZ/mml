tta
===

The ``tta`` config group manages ``test time augmentation`` during inference. Note that this feature is still in beta
phase. By default (``tta=none``) no tta is performed. If activated the model performs multiple predictions on the same
sample via using different augmentations - after the prediction any geometrical distortion is reversed and the produced
predictions are merged. The augmentations are loaded through the
:class:`~mml.core.data_loading.augmentations.kornia.KorniaAugmentationModule` that can deal with any (unnested) list of
`kornia <https://kornia.readthedocs.io/en/latest/augmentation.html>`_ augmentations.
Note that tta is only active during testing and predicting with models - not during training nor validation.
The following examples give a good overview on the configuration options:

rotate
~~~~~~

.. autoyaml:: tta/rotate.yaml


crop
~~~~

.. autoyaml:: tta/crop.yaml