tasks
=====

The ``tasks`` config group cares about the task selection available to the scheduler. For each task listed the scheduler
automatically holds a :class:`~mml.core.data_loading.task_struct.TaskStruct` that can be interacted with to load
the task as well as attach / retrieve other artifacts.

Next to the base list of tasks
there are two important concepts:

    * the ``pivot`` task is a "highlighted" task, some schedulers behave differently depending whether a pivot is provided or specifically require one
    * the ``tag`` mechanism allows task modifications (see also the ``mml-tags`` :doc:`/plugins`) - keep in mind that for the majority of cases tagged (and multi-tagged) tasks are treated completely independent

Note that the ``tasks`` config entries
live "top level", which means although you select a config file via ``tasks=fake``, the modification of entries has no
``tasks`` prepended (aka. ´´pivot.name=MY_TASK``). Having a large pool of tasks is prone for typos during CLI
interaction which can be resolved either via creating a config file (you may use ``_template.yaml`` as a basis) or
using the :class:`~mml.interactive.planning.MMLJobDescription` jointly with a
:class:`~mml.interactive.planning.JobRunner`.


none
~~~~

By default no tasks are given.

.. autoyaml:: tasks/none.yaml

fake
~~~~

This config points to the :mod:`~mml.core.data_preparation.fake_task` and is intended for debugging and testing.

.. autoyaml:: tasks/fake.yaml
