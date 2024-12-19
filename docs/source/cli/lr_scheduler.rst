lr_scheduler
============

Provides the possibility to choose a learning rate scheduler. Should be compatible with any `lr_scheduler``by
``torch`` (see `here <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_ for a list). The
internal calls to the scheduler are dealt with by
`lightning <https://lightning.ai/docs/pytorch/stable/common/optimization.html>`_. Most schedulers have a couple of
hyperparameters, which are not all mapped to the ``yaml`` config files. You may add those from CLI via prepending
a ``+`` (e.g. ``lr_scheduler=plateau +lr_scheduler.cooldown=3``). Below you can find the config files, but adding more
should be straightforward from the examples provided. By default no scheduler is used.

none
~~~~

.. autoyaml:: lr_scheduler/none.yaml

cosine
~~~~~~

.. autoyaml:: lr_scheduler/cosine.yaml

cosine_restart
~~~~~~~~~~~~~~

.. autoyaml:: lr_scheduler/cosine_restart.yaml

exponential
~~~~~~~~~~~

.. autoyaml:: lr_scheduler/exponential.yaml

one_cycle
~~~~~~~~~

.. autoyaml:: lr_scheduler/one_cycle.yaml

plateau
~~~~~~~

.. autoyaml:: lr_scheduler/plateau.yaml

step
~~~~

.. autoyaml:: lr_scheduler/step.yaml