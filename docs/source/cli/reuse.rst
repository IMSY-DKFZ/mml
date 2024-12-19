reuse
=====

The ``reuse`` config group determines the :class:`~mml.core.data_loading.file_manager.MMLFileManager`'s behaviour at
the beginning of an experiment. Through the :meth:`~mml.core.data_loading.file_manager.MMLFileManager._find_reusables`
method the :class:`~mml.core.data_loading.file_manager.MMLFileManager` loads existing data as specified. When the
:class:`~mml.core.data_loading.task_struct.TaskStructFactory` creates a
:class:`~mml.core.data_loading.task_struct.TaskStruct` in
:meth:`~mml.core.data_loading.task_struct.TaskStructFactory.create_task_struct` it uses the
:meth:`~mml.core.data_loading.task_struct.TaskStructFactory.set_task_struct_defaults` method to attach the loaded paths
to the struct. As such they can be accessed during runtime of the scheduler and any of its subroutines. Note that for
``models`` the struct will hold a list of all loaded models, while every other key will only add a single path to the
struct. More details on the syntax can be found at :doc:`/usage`. By default no data is loaded (``reuse=none``).

current
~~~~~~~

.. autoyaml:: reuse/current.yaml