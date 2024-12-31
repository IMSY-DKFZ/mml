Extensions
==========

``mml`` supports a lot of possible extensions. You can find some examples in the :doc:`plugins`. This site serves as
a reference to list and explain some possible entry points.

Scheduler
---------

Writing a new scheduler allows you to completely design new processes inside ``mml`` while leveraging all of its
infrastructure and seamlessly integrate with existing procedures. New schedules are implemented by inheriting from the
:class:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler`. Several examples are provided in the
:mod:`~mml.core.scripts.schedulers` module, as well as scattered through some :doc:`plugins`. The first step would be to
consider the atomic steps of the scheduler. Usually they incorporate the interaction with a single or few tasks and
produce some sort of artifact as result of calculations (e.g. a trained model, extracted features, visualizations, ...).
Let's assume there are two atomic steps: ``step_a`` and ``step_b``, where the first one runs on every task of the task
list producing some artifact and the second runs a single time to aggregate all artifacts. Their methods might look like

.. code-block:: python

    class BuzzScheduler(AbstractBaseScheduler):
    def step_a(self, task_name):
        struct = self.get_struct(task_name)
        # do calculations
        datamodule = self.create_datamodule(struct)
        artifact = heavy_work(datamodule)
        # store
        path = self.fm.construct_saving_path(obj=artifact,key='foo',task_name=task_name)
        artifact.store(path)
        # attach path to struct (this will make it persistent across atomic steps)
        struct.paths['foo'] = path

    def step_b(self):
        # load artifacts
        artifacts = {}
        for task in self.cfg.task_list:
            struct = self.get_struct(task_name)
            artifacts[task] = load_my_artifact(path=struct.paths['foo'])
        # do some work
        end_result = heavy_work(artifacts)
        # store
        path = self.fm.construct_saving_path(obj=end_result,key='baz')
        end_result.store(path)

Now the only thing remaining is to tell the scheduler what subroutines are available and how they shall compose.

.. code-block:: python

    class BuzzScheduler(AbstractBaseScheduler):
    def __init__(self, cfg: DictConfig):
        # initialize
        super(BuzzScheduler, self).__init__(cfg=cfg, available_subroutines=["a", "b"])

    def create_routine(self):
        # -- add step_a commands
        if "a" in self.subroutines:  # subroutines may be called independently
            for task in self.cfg.task_list:
                self.commands.append(self.step_a)
                self.params.append([task])
        # -- add step_b command
        if "b" in self.subroutines:
            # run only once, nor args
            self.commands.append(self.step_b)
            self.params.append([])

To make your scheduler accessible you need to provide a ``mode`` config. You may either place this config inside the
original ``mml`` config folder (e.g. if you cloned the repo), create a dedicated config folder where you like (using
``mml-copy-conf`` (see :doc:`install`) or provide them through a plugin (see :doc:`plugins`). It could look like this:

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

Some more words on the interactions of such a config can be found in :doc:`usage`.

Modalities and TaskTypes
------------------------

Adding a new :class:`~mml.core.data_loading.task_attributes.Modality` and/or a new
:class:`~mml.core.data_loading.task_attributes.TaskType` requires the following steps

  * add your modality and task type to the respective ``Enum`` classes at :mod:`~mml.core.data_loading.task_attributes`
  * make sure to provide the correct entry in :meth:`~mml.core.data_loading.task_attributes.TaskType.requires`
  * if a new modality is used:
    * provide a :class:`~mml.core.data_loading.ModalityLoader` class and link it correctly in the configs (`config/loaders`)
    * write a verifier (see :mod:`~mml.core.data_preparation.task_creator`) and add it to the :attr:`~mml.core.data_preparation.task_creator.MODALITY_VERIFIER_MAP`
    * check compatibility with `kornia <https://github.com/kornia/kornia/blob/33cffcd33ee73e4a3006469d24518ca0c75c3a02/kornia/constants.py#L56>`_ and`albumentations <https://github.com/albumentations-team/albumentations/blob/main/albumentations/core/transforms_interface.py#L162>`_, specifically ensure compatibility with :attr:`~mml.core.data_loading.task_attributes.KORNIA_VALID_MODALITIES`
  * if a new task type is used:
    * add the target modality to the :class:`~mml.core.data_loading.task_struct.TaskStruct` target property
    * add heads that handle that task type to the model you intend to use
    * add a config route entry to :attr:`~mml.core.models.lightning_single_frame.CONFIGS_ROUTES`
    * create configs in `configs/metrics` as well as `configs/loss` representing suitable metrics and loss options
    * link created configs in the respective `defaults.yaml` config file of those two folders