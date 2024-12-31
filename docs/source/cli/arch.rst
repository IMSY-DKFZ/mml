arch
====

The arch config group determines the model architecture used by ``mml``. Currently there are two backbone libraries:
`timm <https://github.com/huggingface/pytorch-image-models>`_ and
`segmentation-models-pytorch <https://github.com/qubvel-org/segmentation_models.pytorch>`_. The former supports
classification and regression tasks while the latter supports classification and segmentation tasks.

Default config option is ``arch=timm``. Call ``arch=smp`` for segmentation models.

timm
~~~~

.. autoyaml:: arch/timm.yaml


smp
~~~

.. autoyaml:: arch/smp.yaml