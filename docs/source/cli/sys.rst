sys
===

The ``sys`` config group deals with system specific settings and allows you to set ``mml`` configuration for multiple
compute hardware. By default ``mml`` ships with two systems in mind: a local setup and a remote compute cluster. For
each system the root paths need to be set somewhere - the ``mml.env`` file offers an easy solution to keep these
personal settings out of your repository but you may set those paths also directly in the ``sys`` config file. Below
you find the two configurations provided:


local
~~~~~

.. autoyaml:: sys/local.yaml

cluster
~~~~~~~

.. autoyaml:: sys/cluster.yaml