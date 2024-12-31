mml-tags plugin
===============

The ``mml-tags`` plugin increases possibilities to modify base tasks. Install via

.. code-block:: bash

    pip install mml-tags --index-url https://__token__:<your_personal_token>@git.dkfz.de/api/v4/projects/89/packages/pypi/simple

Afterwards you may use the tags. For a list of (all) available tags, type ``mml-tags``. Using a task tag is demonstrated here:

.. code-block:: bash

    mml task_list=[example_task+tag_one?parameter_one?parameter_two+tag_two+tag_three]