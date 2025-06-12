Hyperparameter optimization
===========================


Attaching ``--multirun`` to your command will start the job in hpo mode. Note that multirun does not offer
the ``continue`` (see :ref:`continue-option`) functionality!

.. note::
    More details on the config options can be found at :doc:`cli/hpo`.

Gridsearch
----------

This is a very easy way to start a bunch of jobs with slightly varying setting. Find the details at
`hydra multirun <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run>`_. Multiple values for a parameter are
simply given at once separated by a ``,``. You must specify ``hpo=grid`` for this variant.
An example usage to vary learning rate is given below:

.. code-block:: bash

    mml MY_MODE proj=MY_HPO_PROJ hpo=grid optimizer.lr=1.e-4,1.e-5 --multirun


Optuna
------

This hpo method is more involved. Details can be found at the `hydra optuna plugin <https://hydra.cc/docs/plugins/optuna_sweeper>`_
website as well as the website of the backend module `Optuna <https://optuna.readthedocs.io/en/stable/index.html>`_.


usage
~~~~~

Create a config file defining your search space for the hyperparameters at ``configs/search_space``.
And afterwards call the program as follows (make sure your mode has a return value to optimize using
:class:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler`'s
:attr:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.return_value`).

.. code-block:: bash

    mml MODE_WITH_RETURN_VALUE proj=MY_HPO_PROJ search_space=MY_SEARCH_SPACE --multirun

plugin
~~~~~~
The :doc:`api/plugins/sql` extends this setup with a database backend to coordinate hyperparameter optimization
trials across multiple nodes of a computing infrastructure. It further allows for results persistence in the storage.