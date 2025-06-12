hpo
===

The hpo config group controls hyperparameter optimization searches. It is triggered by the ``--multirun`` flag when
invoking an mml experiment (see `hydra's documentation <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/>`_).

.. note::
    If using an :class:`~mml.interactive.planning.MMLJobDescription` there is the ``multirun`` boolean flag.

To leverage the results of a hyperparameter run the CLI kwarg ``use_best_params`` can be used. See
:ref:`main-config-file` for its documentation. Also note the overview given at :doc:`/hpo`.


default
~~~~~~~

.. autoyaml:: hpo/default.yaml


grid
~~~~

.. autoyaml:: hpo/grid.yaml