mml.core.data\_preparation
==========================

``mml.core.data_preparation`` deals with the initial data integration part, for more general runtime data loading see
which is dealt with in :doc:`../data_loading/overview`.

The two core components of integrating a task into ``mml`` are
  * :class:`~mml.core.data_preparation.dset_creator.DSetCreator` to virtually arrange data at the right place
  * :class:`~mml.core.data_preparation.task_creator.TaskCreator` to aggregate full task description into a ``.json`` file

The splitting of these two concepts has the following advantages:
  * large files like images and masks are only stored once even if multiple tasks share the same files
  * support of multiple kind of tasks on the same data
  * capabilities to easily modify task descriptions without touching underlying data

The modules :mod:`~mml.core.data_preparation.dset_creator` and :mod:`~mml.core.data_preparation.task_creator` hold
the respective classes. :mod:`~mml.core.data_preparation.fake_task` is a simple instantiation of those,
that can be used during testing. The :mod:`~mml.core.data_preparation.registry` is the central spot to administrate
all :class:`~mml.core.data_preparation.dset_creator.DSetCreator`s and :class:`~mml.core.data_preparation.task_creator.TaskCreator`s.
:mod:`~mml.core.data_preparation.data_archive` provides the capability to describe and arrange raw datasets, as e.g.
to be downloaded from the web. The :mod:`~mml.core.data_preparation.archive_extractors` add support for unpacking
archives as ``zip`` or ``rar``.
Finally :class:`~mml.core.data_preparation.task_creator.utils` holds a bunch of convenience functions to be used while
creating tasks.

.. toctree::
    :hidden:
    :maxdepth: 1

    dset_creator
    data_archive
    archive_extractors
    fake_task
    registry
    task_creator
    utils
