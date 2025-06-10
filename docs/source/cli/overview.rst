CLI
===

This help provides a detailed overview on CLI of MML. The basic call pattern is

.. code-block:: bash

    mml [mode] [overrides] [hydra.overrides] [hydra-flags]

Besides there are also the following mml-core CLIs (without any arguments):

  * mml-env-setup   - sets up an `mml.env` file at your current location
  * mml-copy-conf   - sets up mml configs outside the mml-core package


mode
----

Available modes include:

  * create    - Installs datasets and tasks on the workstation.
  * pp        - Preprocesses tasks with the given "preprocessing".
  * train     - Trains, tests and/or predicts (single or multi-task).
  * post      - Postprocessing via calibration and ensembling.
  * info      - Provides information on tasks, trained models, etc..
  * clean     - May be used the remove artefacts from mml.
  * upgrade   - Used to migrate mml results and data upwards.
  * downgrade - Used to migrate mml results and data downwards.

Note that mml plugins may add further modes. You can find more example usages for
modes at :doc:`/modes` and specific details on mode configuration at :doc:`mode`.

overrides
---------

MML offers a flexible system to override experiment configuration from
the command line. It is powered by `Hydra <https://hydra.cc>`_ and more
details on the syntax can be found in the respective
`documentation <https://hydra.cc/docs/advanced/override_grammar/basic/>`_.
In a nutshell
configuration options are grouped and one can either override a whole
group of options with existing config files (e.g. ``lr_scheduler=cosine``)
or set values inside a config group (e.g. ``lr_scheduler.verbose=false``).

.. note::
    Hydra configuration is presented in a simplified manner above. There are
    special cases of combining config files (e.g. ``callbacks=[early,mixup]``),
    accessing nested config files (e.g. ``loss/mlcls=ce``) or adding new keys to
    a configuration (e.g. ``+lr_scheduler.eta_min=0.01``).

This following is a list of ``mml`` config groups. To see all available current options for a group call ``mml --help``.

  * :doc:`arch` - determines model architecture
  * :doc:`augmentations` - sets the pipeline for image augmentations during training and general normalization strategy
  * :doc:`callbacks` - determines ``lightning`` callbacks during training
  * :doc:`compile` - BETA controls pytorch 2.0 ``torch.compile`` behaviour
  * :doc:`hpo` - hyperparameter optimization methodology
  * ``hydra`` - hydra internal configs, see ``hydra.overrides`` below
  * :doc:`logging` - experiment logging and other notification settings
  * :doc:`loss` - training loss
  * :doc:`lr_scheduler` - learning rate schedulers
  * :doc:`metrics` - metrics to measure model performance
  * :doc:`mode` - central component defining the scheduler and other corresponding runtime settings
  * :doc:`optimizer` - network training optimizer
  * :doc:`peft` - parameter efficient finetuning
  * :doc:`preprocessing` - image preprocessing pipeline (applied while training and predicting)
  * :doc:`reuse` - determines the reuse of previous results as well as clean up of intermediates
  * :doc:`sampling` - sets sampling strategy
  * :doc:`search_space` - defines the search space during hyperparameter optimization
  * :doc:`sys` - system properties (manages to run on different hardware)
  * :doc:`tasks` - pre-compiled task lists, pivot tasks and task tagging options
  * :doc:`trainer` - ``lightning`` trainer options
  * :doc:`tta` - BETA test time augmentation
  * :doc:`tune` - tuning options with ``lightning`` tuner

The configuration groups and overrides will be compiled to a final single
job configuration (or multiple in ``--multirun`` mode as described below).
The final configuration can be displayed with the help of hydra-flags
(see below) and is also stored in the run folder inside the ``.hydra`` subdir.

.. toctree::
   :hidden:
   :maxdepth: 1

   arch
   augmentations
   callbacks
   compile
   hpo
   loaders
   logging
   loss
   lr_scheduler
   metrics
   mode
   optimizer
   peft
   preprocessing
   remove
   reuse
   sampling
   search_space
   sys
   tasks
   trainer
   tta
   tune

.. _main-config-file:

main config file
~~~~~~~~~~~~~~~~

Furthermore ``config_mml.yaml`` (the main config file) specifies the defaults for each of these groups as well as
some other default values. These few top-level options are listed here:


.. autoyaml:: config_mml.yaml


hydra.overrides
---------------

The same override style also let's you alter configurations that directly
influence the internal behaviour of hydra. The most common use case might
be for example ``hydra.verbose=true``, which enters verbose mode and print
all logged ``debug`` messages. You can find more config groups via ``mml --help``
or follow the `hydra documentation <https://hydra.cc/docs/configure_hydra/intro/>`_.


hydra-flags
-----------

``hydra`` offers some functionality that is inherited by ``mml``. All existing
options are displayed once more if you call ``mml --help``, but here are some
noteworthy ones:

    * ``--cfg=job`` - print the compiled config (without running ``mml``)
    * ``--multirun`` - used for hyperparameter search, starts multiple jobs
    * ``--info`` - information on the defaults tree, config search paths, etc

More info in the  `hydra docs <https://hydra.cc/docs/advanced/hydra-command-line-flags/>`_.
