mml.core.models
===============

``MML`` abstracts it's models into underlying :mod:`torch` models and overlaying :mod:`lightning` models. The ``torch``
model encapsulates the network description while the ``lightning`` wrapper controls loss computation,
optimizer, metrics and logging.

This allows very simple extensions via adding new ``torch`` models and wrapping those with the same ``lightning``
module. Two examples for very flexible ``torch`` models are the :mod:`~mml.core.models.timm` and
:mod:`~mml.core.models.smp` models that build upon the :mod:`timm` library and the :mod:`pytorch_segmentation_models`
library respectively. So far only ``single frame`` tasks are supported by the
:mod:`~mml.core.models.lightning_single_frame` ``lightning wrapper`` but video clip support is planned.


.. toctree::
    :hidden:
    :maxdepth: 1

    torch_base
    timm
    smp
    lightning_single_frame
