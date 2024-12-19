mml.core.scripts
================

The ``scripts`` sub-module of ``MML`` focuses on the data processing. It's core is the
:class:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler` that provides all internal routing, logging,
object holding as well as common methods to ease objects creation. Developers can inherit from it and quickly write
new schedulers for their use case. The following inherited schedulers are shipped with ``mml-core``:

  * :class:`~mml.core.scripts.schedulers.info_scheduler.InfoScheduler` - provides basic information on ``mml``, available tasks and hyperparameter searches
  * :class:`~mml.core.scripts.schedulers.create_scheduler.CreateScheduler` - allows to automatically create (and integrate) data of new tasks
  * :class:`~mml.core.scripts.schedulers.preprocess_scheduler.PreprocessScheduler` - preprocess tasks once to speed up training
  * :class:`~mml.core.scripts.schedulers.train_scheduler.TrainingScheduler` - single task training, cross-fold validation, generate predictions

Some ``MML`` plugins ship their own schedulers with them. In addition this sub-module holds:

  * :doc:`callbacks` - some useful ``lightning`` callback implementations
  * :doc:`decorators` - some useful decorators
  * :doc:`exceptions` - exceptions defined by ``mml``
  * :doc:`notifier` - a notification system for ``mml`` monitoring
  * :doc:`model_storage` - a container to hold all relevant information of a trained model
  * :doc:`pipeline_configuration` - a container to encapsulate a training pipeline
  * :doc:`utils` - a bunch of useful stuff


.. toctree::
    :hidden:
    :maxdepth: 1

    schedulers/base_scheduler
    schedulers/info_scheduler
    schedulers/create_scheduler
    schedulers/preprocess_scheduler
    schedulers/train_scheduler
    schedulers/transfer_scheduler
    schedulers/clean_scheduler
    schedulers/upgrade_scheduler
    callbacks
    decorators
    exceptions
    model_storage
    notifier
    pipeline_configuration
    utils
