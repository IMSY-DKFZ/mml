loaders
=======

The loaders config group determines the way each :class:`~mml.core.data_loading.task_attributes.Modality` is
processed by some :class:`~mml.core.data_loading.modality_loaders.ModalityLoader` to bring sample entries to tensor
ready format in ``mml``. More details on available loaders can be found in the API entry of
:mod:`~mml.core.data_loading.modality_loaders`. The config requires an equally named entry to the
:class:`~mml.core.data_loading.task_attributes.Modality` that points to the respective
:class:`~mml.core.data_loading.modality_loaders.ModalityLoader` to be used for that
:class:`~mml.core.data_loading.task_attributes.Modality`.

For most cases the ``default`` configuration should offer a sufficing solution, but specific file formats might
require a different :class:`~mml.core.data_loading.modality_loaders.ModalityLoader`. There is also some potential
for efficiency improvements through testing different :class:`~mml.core.data_loading.modality_loaders.ModalityLoader` s.

The following callbacks configuration files are currently available:

default
~~~~~~~
The default configuration determines the general fallback option for all modalities.

.. autoyaml:: loaders/default.yaml

numpy
~~~~~

.. autoyaml:: loaders/numpy.yaml

pillow
~~~~~~

.. autoyaml:: loaders/pillow.yaml

pillow_acc
~~~~~~~~~~

.. autoyaml:: loaders/pillow_acc.yaml

scikit
~~~~~~

.. autoyaml:: loaders/scikit.yaml

torch
~~~~~

.. autoyaml:: loaders/torch.yaml

uni
~~~

.. autoyaml:: loaders/uni.yaml
