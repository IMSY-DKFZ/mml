tune
====

The ``tune`` config group allows to make use of the
`lightning.Tuner <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html>`_. A scheduler
may call :meth:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.lightning_tune` before starting a
``Trainer.fit`` run. This will (depending on the configuration) allow to modify two hyperparmeters:

    * the batch size (to maximize usage of VRAM)
    * the learning rate (to stabilize learning)

default
~~~~~~~

.. autoyaml:: tune/default.yaml