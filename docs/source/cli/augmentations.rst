augmentations
=============

The augmentations config group determines the image augmentations applied by ``mml`` during the loading of train data
(on top of any preprocessing). Currently there are three backbone libraries:
`albumentations <https://albumentations.ai/docs/>`_, `torchvision <https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended>`_ and
`kornia <https://kornia.readthedocs.io/en/latest/augmentation.html>`_. While albumentations works on CPU only,
kornia is implemented GPU only in ``mml``. Torchvision is flexible in being applied on both device types. Multiple
pipeline config files are shipped with ``mml`` as examples:

 * ``augmentations=base_rand`` - basic RandAugment
 * ``augmentations=kornia`` - a default GPU transformation pipeline with kornia
 * ``augmentations=load_imagenet_aa`` - a automatically learned augmentation pipeline (by autoalbument)
 * ``augmentations=v2`` - an example with torchvision transforms

Note that it is possible to combine CPU and GPU transforms. ``mml`` takes care of formatting, scaling, tensorizing, etc.
so the config pipelines may fully focus on the relevant image transformations. Complex pipelines are best created as
``.yaml`` files and called via file name. You may use the :ref:`config-copy` functionality to create your own
set of plain config files to modify. The relevant entries are documented below:

.. autoyaml:: augmentations/default.yaml
