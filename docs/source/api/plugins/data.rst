mml-data plugin
===============

The ``mml-data`` plugin enables the availability to use a set of pre-defined tasks. Install via

.. code-block:: bash

    pip install mml-data --index-url https://__token__:<your_personal_token>@git.dkfz.de/api/v4/projects/89/packages/pypi/simple

Afterwards you may call ``mml-data`` from the terminal to receive a report of available tasks and datasets. Run
``mml-data --help`` for more details.

You may now use the defined task creators simply to create the corresponding tasks locally via:

.. code-block:: bash

    mml create task_list=[...]