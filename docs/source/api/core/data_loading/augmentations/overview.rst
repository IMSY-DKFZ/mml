mml.core.data\_loading.augmentations
====================================

``mml.core.data_loading.augmentations`` deals with everything concerning data augmentations. The base class is for this
is :class:`~mml.core.data_loading.augmentations.augmentation_module.AugmentationModule`. Currently ``mml`` supports three
libraries for data augmentation:

  * `albumentations <https://albumentations.ai/>`_
  * `kornia <https://kornia.readthedocs.io/en/stable/>`_
  * `torchvision <https://pytorch.org/vision/stable/transforms.html>`_

The basic idea is that ``mml`` deals with all underlying details of loading, precision, device movement etc. As a user
one may use any (or even multiple!) of the libraries above and describe the sequence of desired transformations as part
of the configuration file.

The general flow is as follows:

  * image, mask, etc. are read by their respective :class:`~mml.core.data_loading.modality_loaders.ModalityLoader`
  * if preprocessing is performed (aka a preprocessing pipeline is part of the config AND the data has not yet been preprocessed):

    * the respective composed preprocessing pipeline is performed on a per sample basis on the CPU
    * WARNING: consider preprocessing the dataset for improved efficiency, see :doc:`/modes`

  * if the config contains CPU augmentations these are applied as well on a per sample basis
  * afterwards samples are turned into float precision objects and moved to the device determined by config
  * if the config contains GPU augmentations these are applied on a batch level
  * the last augmentation performed will be the normalization (if requested by config)

Certain limitations apply:
  * preprocessing is only supported for the albumentations library
  * albumentations is only supported on CPU
  * kornia is only supported on GPU

Multiple examples can be found in the ``src/mml/configs/augmentations`` folder (and ``src/mml/configs/preprocessing``).

Furthermore :mod:`~mml.core.data_loading.augmentations.mixup_cutmix` offers some lightning callbacks to perform CutMix
and MixUp during batch collation. In contrast to the torchvision/kornia versions of these those callbacks support
multi-task.

.. toctree::
    :hidden:
    :maxdepth: 1

    albumentations
    augmentation_module
    kornia
    mixup_cutmix
    torchvision