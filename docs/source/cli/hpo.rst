hpo
===

The hpo config group controls hyperparameter optimization searches. It is triggered by the ``--multirun`` flag when
invoking an mml experiment (see `hydra's documentation <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/>`_).
If using an:class:`~mml.interactive.planning.MMLJobDescription` there is the ``multirun`` boolean flag.

``use_best_params``

:doc:`/hpo`


default
~~~~~~~

.. autoyaml:: hpo/default.yaml


grid
~~~~

.. autoyaml:: hpo/grid.yaml