Getting started
===============

This section is intended to help with the learning curve of ``mml`` and let you master it as
smoothly as possible. It splits itself into the following chunks:

  * a quick overview on the **core** concepts of ``mml``
  * **basic** usage of the ``mml`` CLI, for full descriptions of all command line options see :doc:`cli/overview`
  * variants of hooking into ``mml`` to define your own scheduler, datasets, path assignments and more
  * interactive experiment pre- and post-processing

After reading through you may want to continue reading the :doc:`modes` for a more detailed overview on ``mml`` modes
and :doc:`guides` for more specific use cases.

Finally if you want to dive deeper into the ``mml`` internals, read through :doc:`api/overview` section.

Concept
-------

``mml`` is a full toolkit to be leveraged in interacting with RGB images for deep learning. It can be accessed through
the command line interface (CLI) or interactively via jupyter notebooks. An ``experiment`` (or ``run``) comprises a
single call to the ``mml`` CLI. Each experiment is assigned to a ``project``, that determines where the produced
artefacts (e.g. trained models) and experiment logs reside (see `proj`_). Experiments within a project may reuse
artefacts from a different project though (see `reuse`_). The imaging data (except for plotting and visual logs) is kept
separately from conducted experiments and stored at the location given by the ``MML_DATA_PATH`` variable. The
installation of data is performed via an explicit ``mml create`` call. ``mml pp`` allows for preprocessing data and
storing the results as well - if this step is omitted or the ``preprocessing`` configuration changes (i.e. no
preprocessed data is stored) then data will be preprocessed on the fly automatically. ``create`` and ``pp` are two
exemplary modes of ``mml`` (see `mode`_), basically top level instructions. More fine-grained configuration can be
achieved more pretty much all internal details, see :doc:`cli/overview`. Next to this configurability, the strengths
of ``mml`` lie in its extendability e.g. through adding more modes (see :doc:`extensions`) and ease of data integration
(see :doc:`guides`). ``mml`` also offers a plugin system, and ships with a selection of useful plugins to enrich the
experience (see :doc:`plugins`).


Basics
------

``mml`` has a very flexible configuration mechanism from the command line using the `hydra <https://github.com/facebookresearch/hydra>`_ framework.
To understand the usage better we tackle piece by piece of the configuration.

mode
~~~~

``mml`` has a bunch of **modes** to choose from, which determine its behaviour. The results of these modes may interact with
each other, e.g. first training a model on one task with ``train`` and afterwards reusing this model on a different
task with transfer learning via ``tl`` (the details on the ``reuse`` functionality are explained below).
Each time you call ``mml`` you should therefore specify this mode. If no mode is given ``mml`` only shows basic information.
An overview of modes can be found at :doc:`modes`. Each mode usually has further configuration options, for example
a mode may split into several **subroutines** that can be composed individually. A typical call may be
``train mode.subroutines=[train,predict] mode.cv=false mode.nested=false`` that trains a model on fold 0 (no cross
validation) of the training data and uses the same model to predict the test samples of a task.


proj
~~~~

Each ``mml`` run is assigned to a **project**, which is represented by a top level directory at the ``MML_RESULTS_PATH``.
You can use any string to define such a project (``proj=fancy_new_feature``) and also reuse existing projects. By default
the ``proj=default`` project is used, any new project name will automatically create the folder.
Multiple runs can be assigned to the same project sequentially or even in parallel. Inside the project folder (precisely
inside its ``runs`` subdir) each run will individually create a run folder that contains e.g. the ``exp.log`` file as well as
other information. Note that the results of a run are not stored at experiment level, but at project level to enable
shared usage across runs (and even across projects).

.. _continue-option:

continue
~~~~~~~~

In case a run failed (e.g. CUDA OOM) or you had for some reason to
stop the run, you may **continue** this run later on via specifying the continue flag. This is advantageous in case you are
developing a new feature or have a long series of computations already done within the run you do not want to repeat.
You can either specify the exact ``runs``
subdir (``continue=2023-03-23/11-51-42``) to continue or as a shortcut start from the latest run (``continue=latest``).
If you use ``continue`` any other argument besides ``proj`` is ignored.


tasks/task_list/pivot
~~~~~~~~~~~~~~~~~~~~~

To determine which task is used within a mode to be processed use either ``task_list=[task_a,task_b,task_c]`` or define
a `my_tasks.yaml`` config file in ``configs/tasks`` and simplify to ``tasks=my_tasks``. Many modes behave differently if
a designated **pivot task** is given via ``pivot.name=task_a``, note that providing ``pivot.name`` automatically adds
the task in the ``task_list`` (or ``tasks`` config file) if not already present.

hydra.verbose
~~~~~~~~~~~~~

For debugging purposes you may activate verbose logging (note that ``mml`` logs both to the ``stdout`` as well as to
the ``exp.log`` file of the run) by setting general ``hydra.verbose=true`` or specifying the loggers/modules you want
to debug by e.g. ``hydra.verbose=[mml.core,hydra]`` (see `hydra docs <https://hydra.cc/docs/tutorials/basic/running_your_app/logging/>`_).

reuse
~~~~~

If you have produced some result and want to reuse them in another experiment (e.g. extracted
features for a task in another task similarity experiment) you can use the ``reuse`` config option
as shown in the examples below:

.. code-block:: bash

    mml XXX proj=test reuse=none  # won't load any reusables (default)
    mml XXX proj=test reuse.models=other_proj  # loads models from project 'other_proj'
    mml XXX proj=test reuse.predictions=[other_proj,foo_proj, baz_proj]  # loads predictions from multiple projects
    mml XXX proj=test reuse.parameters=other_proj#3  # loads parameters with number 3 from project 'other_proj'

By default the most recent results within any project are reused! Appending ``#`` and some integer refers to a specific
file number (e.g. the parameter file ``model_0003.pth`` in the example above). If multiple projects are specified the
last found entry is kept (e.g. if in the example above ``other_proj`` and `foo_proj`` hold predictions for a task, but not
``baz_proj``, then the last predictions from ``foo_proj`` are reused. A fundamental exception to this mechanism are models since here
ALL models are loaded - within a project and across a given list of projects (specifying ``#`` is not allowed).

trainer
~~~~~~~

Under the hood ``mml`` uses `lightning <https://github.com/Lightning-AI/lightning>`_ to run deep learning routines. This
allows a very flexible parametrization of training behaviour through the interface of the
`lightning trainer class <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_. You can pass through any
arguments to the trainer via ``trainer.kwarg=value`` from the CLI. (Some values are set by default from ``mml`` others
are not, so you may sometimes need to add a ``+`` in front for those not used previously.)

.. code-block:: bash

    mml train proj=test trainer.accelerator=tpu  # use given TPU's for computations (default=auto)
    mml train proj=test trainer.max_epochs=40  # will stop training after 40 epochs
    mml train proj=test +trainer.profiler=advanced  # use lightning advanced profiler during training

others
~~~~~~

`Lightning <https://github.com/Lightning-AI/lightning>`_ offers more features like callbacks and model tuning
which are mapped to ``callbacks`` and ``tune`` CLI within ``mml``. Furthermore there are plenty of other
possibilities to set behaviour from CLI. To give you examples:

sampling, seed, gpus, arch, ....

.. code-block:: bash

    mml train proj=test callbacks=[mixup,swa] cbs.swa.swa_lrs=0.005  # use MixUp and SWA callbacks, set swa lr
    mml train proj=test augmentations=randaugment tune.lr=true  # use RandAugment and auto LR finder
    mml train proj=test sampling.sample_num=1000 sampling.batch_size=100  # set batch size and number of samples per epoch
    mml train proj=test seed=42 arch.name=resnet50  # use random seed 42 and a resnet50 model

Type ``mml --help`` to see all available provided config files (or look into the ``mml/configs`` folder
for more details). At all times you may add ``--cfg=job`` to your command to give you the fully compiled
config file (may interesting to detect new options and become aware of defaults).

--multirun
~~~~~~~~~~

Attaching ``--multirun`` to your command will start the job in hpo mode. Note that multirun does not offer
the ``continue`` functionality! Read more about this in :doc:`hpo`.


Hook into MML
-------------

Depending on your use case there might be necessity to hook into the ``mml`` runtime to provide your own
scheduler, datasets, path assignments and more. To make ``mml`` use a local config folder within your project
read the corresponding section in :doc:`install`. There you can already create newly available config
files or modify default configurations. But to define e.g. a new mode with a new scheduler you have to make this
scheduler available inside ``mml``. Here are multiple options:

  * call the ``mml`` CLI from inside your code
  * make your package importable and use ``hydra.instantiate`` to refer to your class/function through the configs
  * provide the ``mml`` entry point from inside your package, to load it as plugin during ``mml`` initialization
  * clone the ``mml`` source code and make your adaptions directly within ``mml``

The options are ordered by increasing complexity which means more possibilities on the one hand but also requiring
deeper understanding of ``mml``.

call mml CLI
~~~~~~~~~~~~

An example for the first option is given in the quickstart guide of :doc:`index`. It involves importing
the objects of ``mml`` you want to modify, e.g. register a data creator and finally call the ``mml.cli.main``
function to pass any CLI parameters forward. Note that as a downside ``hydra`` cannot instantiate your defined objects
unless your package is installable and you also have no runtime access to e.g. the path assignments of the file manager.

hydra.instantiate
~~~~~~~~~~~~~~~~~

The next option is to package your code. This basically requires a ``setup.cfg`` and/or ``pyproject.toml`` file in
your project. Please refer to the `packaging documentation <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_
for the details of this process. Assume your package is named ``foo`` and you have a module ``foo.bar`` defining a
class ``BuzzScheduler`` (inherited from :class:`~mml.core.scripts.base_scheduler.AbstractBaseScheduler`). Then you could
create a new config file ``buzz.yaml`` inside ``configs/mode`` as follows:

.. code-block:: yaml

    # @package _global_

    defaults:
      - override /augmentations: no_norm
      - override /sampling: extraction_default

    mode:
      id: BUZZ
      scheduler:
        _target_: foo.bar.BuzzScheduler
      subroutines:
        - a
        - b
      var_one: 1337
      var_two: 42

    sampling:
      sample_num: 1000

This will behave as follows: after hydra compiles the config with a CLI command starting like ``mml buzz`` the ``buzz.yaml``
file is included and overrides the default ``augmentation`` and ``sampling`` configs. Further it even more overwrites
``sampling.sample_num`` value and when ``MML`` starts it will use ``hydra.instantiate`` to load the ``foo.bar.BuzzScheduler``
scheduler. It may implement one or multiple subroutines determining its behaviour and also take ``cfg.mode.var_one`` and
``cfg.mode.var_two`` values into consideration. See :doc:`extensions` for more details on writing your own scheduler.

entry point
~~~~~~~~~~~

If you want to modify or extend ``MML``'s behaviour outside the scope of a a single class (like the scheduler above)
and provide e.g. additional options to some of the core functions, like a new method of
:class:`~mml.core.scripts.base_scheduler.AbstractBaseScheduler` or automatically register a
:class:`~mml.core.data_preparation.task_creator.TaskCreator` to be available in mode ``create``, you can make your
package a plugin of ``MML`` by adding the following section to your ``setup.cfg``:

.. code-block:: none

    [options.entry_points]
    mml.plugins =
        your_plugin_name = foo.bar

Each time ``MML`` starts all available plugins are loaded automatically, which means importing of ``foo.bar`` in the above case.
The ``__init__.py`` file of this module may then modify ``MML`` internals. You can find examples for this at :doc:`plugins`.

edit source
~~~~~~~~~~~

Finally consider cloning the ``MML`` repository and modify it's behaviour directly at the source. Have a look at
:doc:`api/overview` as good starting point to navigate through the internals of ``MML``.

Pre- and Post-processing
------------------------

experiment preparation
~~~~~~~~~~~~~~~~~~~~~~

Especially to planning ``MML`` experiments there is the :mod:`mml.interactive` module, offering the
:class:`~mml.interactive.planning.MMLJobDescription` class. The following snippet shows an example usage:

.. code-block:: python

    from mml.interactive import DefaultRequirements, EmbeddedJobRunner, MMLJobDescription, write_out_commands

    reqs = DefaultRequirements()
    project = 'my_project'
    all_tasks = ['task_a', 'task_b', 'task_c']
    cmds = list()
    # step one: task creation
    cmds.append(MMLJobDescription(prefix_req=reqs, mode='create', config_options={'task_list': all_tasks, 'proj': project}))
    # step two: make sure all tasks are preprocessed
    cmds.append(MMLJobDescription(prefix_req=reqs, mode='pp', config_options={'task_list': all_tasks, 'proj': project}))
    # (optional) step three: modify tasks (in this case create subsets)
    cmds.append(MMLJobDescription(prefix_req=reqs, mode='info', config_options={'task_list': all_tasks, 'tagging.all': '+subset?0_1', 'proj': project}))
    # now either put all commands into a bash file
    write_out_commands(cmd_list=cmds, name='my_commands_file.txt')
    # or run them directly
    runner = EmbeddedJobRunner()
    for job in cmds:
        runner.run(job=job)

experiment evaluation
~~~~~~~~~~~~~~~~~~~~~

The MML framework offers extensive log information, both to the console and to the ``exp.log`` file of
each run. In addition any NN training can be monitored by some experiment logger. By default
``logging.exp_logger=tensorboard`` is active. To show these information you need to install
`tensorboard <https://www.tensorflow.org/tensorboard>`_. This can for example be done via

.. code-block:: bash

    pip install tensorboard

To start tensorboard call

.. code-block:: bash

    tensorboard --logdir path/to/results

and navigate to `localhost:6006` within your preferred browser. Loss curves and metrics are shown in the ``SCALARS`` tab.
Multirun experiments may be best inspected via the ``HPARAMS`` tab, comparing specific combinations of hyperparameters
with different views. Setting ``logging.samples=n`` also logs n sample images with model predictions per epoch in the
``IMAGES`` tab. There you can also find a confusion matrix for each epoch if ``logging.cm`` is set to ``true``.

For large scale evaluation loading model storages and the corresponding pipelines works as easy as follows with the
``mml.interactive`` module:

.. code-block:: python

    import mml.interactive
    # some interactive sessions do not inherit MML_ENV_PATH env variable, you may provide this directly
    mml.interactive.init(env_path=...)
    all_models = mml.interactive.load_project_models(project='my_project')
    # this will return a dictionary with all instantiated models storages in a list assigned as value to each task as key
    model_storage = all_models['my_sample_task'][0]
    model_storage.metrics  # holds all train/val metrics across the training
    model_storage.pipeline  # holds the path to the yaml file specifying all relevant training configurations
    model_storage.parameters  # holds path to model weights after training
    model_storage.predictions  # holds paths to all predictions made with this model
    ...

