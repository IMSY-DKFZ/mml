mml.core.data\_loading
======================

``mml.core.data_loading`` deals with all file and data operations except for the initial data integration part,
which is dealt with in :doc:`../data_preparation/overview`.

Data loading from a task is layered in three:
  * :class:`~mml.core.data_loading.task_struct.TaskStruct` holds only lightweight meta information of a task
  * :class:`~mml.core.data_loading.task_dataset.TaskDataset` implements mechanics to use the task description file and load individual samples
  * :class:`~mml.core.data_loading.lightning_datamodule.MultiTaskDataModule` holds different splits of (potentially) multiple task, deals with transforms and provides the dataloaders

Furthermore :mod:`~mml.core.data_loading.file_manager` holds the central :class:`~mml.core.data_loading.file_manager.MMLFileManager`.
:mod:`~mml.core.data_loading.augmentations` deals with data augmentations.
:mod:`~mml.core.data_loading.task_attributes` declares task meta information
formats as they are needed by the central :mod:`~mml.core.data_loading.task_description`.
:mod:`~mml.core.data_loading.modality_loaders` implement the reading operations for different
:class:`~mml.core.data_loading.task_attributes.Modality`'s. Lastly
:mod:`~mml.core.data_loading.utils` holds a small number of utility functions for data transformations.

.. toctree::
    :hidden:
    :maxdepth: 1

    augmentations/overview
    file_manager
    lightning_datamodule
    modality_loaders
    task_attributes
    task_dataset
    task_struct
    task_description
    utils
