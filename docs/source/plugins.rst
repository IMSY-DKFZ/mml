Plugins
=======

The concept of plugins is a general pillar of ``mml``'s architecture. Plugins provide e.g. additional schedulers,
tasks and config options. A list of available plugins can be found at :doc:`api/plugins/overview`. This section
is intended to describe the purpose and possibilities of plugins as well as their intended internal structure. It can
thus also be seen as a guide to write your own plugin.

introduction
------------

Plugins are actual python packages (and need to be such), and many are already available from the same registry
as ``mml-core``. Of course ``mml-core`` is a necessary dependency, so installing that one first is necessary
(see :doc:`install`). In a similar fashion and within the same virtual environment one may install plugins.
To make your code installable you have to add
``pyproject.toml`` and ``setup.cfg`` files ar your projects root level. See ``mml``'s ``plugins/_template`` folder
to see template files for those. More broadly the following plugin structure is recommended:

.. note::
    | root
    | ├── src
    | │ └── my_plugin
    | │   ├── configs (optional)
    | │   │  └── config_group (e.g. foo)
    | │   │    └── bar.yaml (allows this config to be used with ``foo=bar``)
    | │   ├── stuff (modules, assets, ...)
    | │   ├── ...
    | │   ├── __init__.py (should provide a __version__ string)
    | │   └── activate.py (contains all imports, adds the config search path)
    | ├── tests
    | │ └── unit (put your tests in here)
    | ├── README.md
    | ├── MANIFEST.in (declare any assets to be shipped with your plugin)
    | ├── pyproject.toml
    | └── setup.cfg

loading
-------

Plugins are loaded during the initialization of ``mml`` during :func:`mml.cli.main`, but before compiling the
hydra configuration of the experiment with the :func:`~mml.core.scripts.utils.load_mml_plugins` function. This allows
plugins to extend the search path of hydra for config files and provide config options themselves (you can check on
these search path plugins and inspect the order by calling ``mml --info searchpath``). Each plugin needs to provide an
entry point as follows in their ``setup.cfg``:

.. code-block::

    [options.entry_points]
    mml.plugins =
        my-plugin = my_plugin.activate

This assure that ``my_plugin.activate`` is sourced before running the ``mml`` core routines and modify any behaviour
before the experiment. The ``activate.py`` file may for example

  * add the plugin's config folder to the configs search path of ``mml`` to add new config options from cli
  * add path assignments to the :class:`~mml.core.data_loading.file_manager.MMLFileManager` via :meth:`~mml.core.data_loading.file_manager.MMLFileManager.add_assignment_path`
  * register :class:`~mml.core.data_preparation.dset_creator.DSetCreator`'s or :class:`~mml.core.data_preparation.task_creator.TaskCreator`'s via :func:`~mml.core.data_preparation.registry.register_dsetcreator` and :func:`~mml.core.data_preparation.registry.register_taskcreator`
  * add or overwrite methods of existing classes like :class:`~mml.core.scripts.base_scheduler.AbstractBaseScheduler`
  * or simply import modules (from ``stuff``) that do so themselves

testing
-------
``mml-core`` provides a ``pytest`` plugin so that tests (ideally within the ``tests`` folder) may use fixtures defined in
:mod:`mml.testing.fixtures`.

