search_space
============

The ``search_space`` config group determines the hyperparameter search space during a ``--multirun``. See :doc:`/hpo`
for a general introduction and :doc:`hpo` for more config options on how to use this. The syntax of defining the search
space is also documented by `hydra <https://hydra.cc/docs/1.3/plugins/optuna_sweeper/#search-space-configuration>`_.
By default (``search_space=none``) no search space is defined. An example configuration is given below.

example
~~~~~~~

.. autoyaml:: search_space/example.yaml
