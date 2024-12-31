loss
====

The ``loss`` config group determines all loss functions that are used for ``mml`` models. It determines a single loss
function per task type. More precisely the :class:`~mml.core.data_loading.task_attributes.TaskType` is matched to
a short string via :attr:`~mml.core.models.lightning_single_frame.CONFIG_ROUTES`. For each model head (that matches
a task type) the respective `loss.SHORT_STRING` config is instantiated. The main file determines some additional
behaviour:

default
~~~~~~~
The default top level configuration.

.. autoyaml:: loss/default.yaml

The following task type matching subdirectories exist:

cls
---

For classification tasks, currently shipped with a single option

ce
~~

.. autoyaml:: loss/cls/ce.yaml

mlcls
-----

For multi-label classification tasks, currently shipped with two options

bce
~~~

.. autoyaml:: loss/mlcls/bce.yaml

ce
~~

.. autoyaml:: loss/mlcls/ce.yaml


reg
---

For regression tasks, currently shipped with a single option

huber
~~~~~

.. autoyaml:: loss/reg/huber.yaml


seg
---

For segmentation tasks, currently shipped with two options

ce
~~

.. autoyaml:: loss/seg/ce.yaml

dice
~~~~

.. autoyaml:: loss/seg/dice.yaml
