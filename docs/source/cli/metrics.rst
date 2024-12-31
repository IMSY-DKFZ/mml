metrics
=======


The ``metrics`` config group determines all metrics that are used to evaluate ``mml`` models. It determines a suite of
metrics per task type. More precisely the :class:`~mml.core.data_loading.task_attributes.TaskType` is matched to
a short string via :attr:`~mml.core.models.lightning_single_frame.CONFIG_ROUTES`. For each model head (that matches
a task type) the respective `metrics.SHORT_STRING` config is instantiated.

The following task type matching subdirectories exist and contain a single ``default.yaml`` file each:

  * cls (for classification tasks)
  * mlcls (for multi-label classification tasks)
  * reg (for regression tasks)
  * seg (for segmentation tasks)

The currently used metrics can be seen via ``mml --cfg=job`` (add `` | grep torchmetrics`` for reduced output). All
metrics used by ``mml`` are from `torchmetrics <https://lightning.ai/docs/torchmetrics/stable/>`_.

The main file determines some additional behaviour:

default
~~~~~~~
The default top level configuration.

.. autoyaml:: metrics/default.yaml



