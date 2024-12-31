remove
======

The ``remove`` config group determines the :class:`~mml.core.data_loading.file_manager.MMLFileManager`'s behaviour at
the end of an experiment. During the :meth:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.finish_exp`
routine of the experiment's scheduler the :meth:`~mml.core.data_loading.file_manager.MMLFileManager.remove_intermediates`
method of the :class:`~mml.core.data_loading.file_manager.MMLFileManager` is called and triggers the deletion of created
files according to the settings of this config
group. This is useful to get rid of intermediates or general results that are not used any more and may use up the disk
capacity.

The path assignment system of the :class:`~mml.core.data_loading.file_manager.MMLFileManager` determines a ``key`` for
each kind of path assignment. This key is used within this config to trigger deletion of files. The default path
assignments (and according keys) are defined in :attr:`~mml.core.data_loading.file_manager.DEFAULT_ASSIGNMENTS`. Here
are some examples:

  * ``models``: the :class:`~mml.core.scripts.model_storage.ModelStorage` for each trained model
  * ``parameters``: the :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` for a model
  * ``pipeline``: the :class:`~mml.core.scripts.pipeline_configuration.PipelineCfg` for a training process
  * ``predictions``: predictions of a model on a task, dictionary stored with ``torch.save``

By default (``remove=none``) no data is deleted (except for inherently temporary data, see
:meth:`~mml.core.data_loading.file_manager.MMLFileManager.remove_intermediates`). The alternative config file
``remove=all`` will take care of most of the default assignments. Any additional key can be added via prepedning a ``+``
as in ``+remove.MY_KEY=true``, which might be necessary for plugin provided path assignments.

all
~~~

.. autoyaml:: remove/all.yaml