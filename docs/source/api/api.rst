mml.api
=======

``mml.api`` is the fast variant to access internal entities of ``MML``. For example you may import

.. code-block:: python

    from mml.api import MMLFileManager

instead of

.. code-block:: python

    from mml.core.data_loading.file_manager import MMLFileManager

You can receive a list of all available functions, decorators, exceptions and classes in ``mml.api`` via

.. code-block:: python

    import mml.api
    print([elem for elem in dir(mml.api) if not elem.startswith('__')])

or by inspecting ``src/mml/api/__init__.py``.
