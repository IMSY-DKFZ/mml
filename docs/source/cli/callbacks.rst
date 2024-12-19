callbacks
=========

The callbacks config group determines the lightning callbacks to be used by ``mml``, whenever lightning is invoked
(training, testing, predicting). Note that callbacks are mostly deactivated during tuning. Lightning provides comes
with some `builtin callbacks <https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#built-in-callbacks>`_
but special care has to taken since some callbacks are handled by ``mml`` itself internally. Callback creation is
handled within :meth:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.create_trainer`, there

  * :class:`~mml.core.scripts.callbacks.StopAfterKeyboardInterrupt` will be added automatically and prevents lightning from catching keyboard interrupts
  * :class:`~mml.core.scripts.callbacks.MetricsTrackerCallback` will also be added if ``metrics_callback=True`` and made accessible as :attr:`~src.mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.metrics_callback`
  * :class:`~mml.core.scripts.callbacks.MMLRichProgressBar` or :class:`~mml.core.scripts.callbacks.MMLTQDMProgressBar` are used as progress bar modifications (depending on ``logging.render`` settings)
  * :class:`~mml.core.scripts.callbacks.MMLModelCheckpoint` is added multiple times as described in :meth:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.create_trainer`

All other callbacks are coordinated through the callbacks config. Noteworthy though the yaml config files are within the
``callbacks`` folder the underlying config structure is ``cbs:{id:{**kwargs}}``. This allows the following two features:

  * stacking callbacks: ``callbacks=[early,swa]`` - here adding both early stopping and stochastic weight averaging to the callbacks
  * modify callback kwargs: ``cbs.early.patience=5`` - here setting the patience parameter of early stopping

The following callbacks configuration files are currently available:

default
~~~~~~~

.. autoyaml:: callbacks/default.yaml


early
~~~~~

.. autoyaml:: callbacks/early.yaml

none
~~~~

.. autoyaml:: callbacks/none.yaml

mixup
~~~~~

.. autoyaml:: callbacks/mixup.yaml

cutmix
~~~~~~

.. autoyaml:: callbacks/cutmix.yaml

swa
~~~

.. autoyaml:: callbacks/swa.yaml

stats
~~~~~

.. autoyaml:: callbacks/stats.yaml