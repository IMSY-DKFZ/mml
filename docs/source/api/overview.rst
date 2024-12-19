mml-core API
============

The ``mml-core`` package makes up the core functionality of ``MML`` and implements the scaffold to operate it.

``mml-core`` is split into 5 sub-packages:

  * :doc:`api` - which serves as fast external api access with some bonus convenience functions to use ``mml`` interactively
  * :doc:`mml.configs <../cli/overview>` - which is the reference of config groups and provides the necessary ``.yaml`` files
  * ``core`` - which is the actual core source code, it splits itself into 5 sub-packages

      * :doc:`core/data_loading/overview` - all code to load, transform, store and represent data internally
      * :doc:`core/data_preparation/overview`  - all code to integrate external data into the ``mml`` framework
      * :doc:`core/models/overview`  - all code with respect to deep neural networks functionality
      * :doc:`core/scripts/overview`  - basically the ``scheduler`` mechanics of ``mml``
      * :doc:`core/visualization/overview`  - code to visualize results

  * :doc:`interactive` - which provides additional convenience classes to plan, process and visualize ``mml`` experiments
  * :doc:`testing` - which supports testing ``mml-core`` and ``mml`` :doc:`../plugins`

Furthermore there is the :doc:`cli` module to streamline command line interface access.

.. toctree::
   :hidden:
   :maxdepth: 1

   api
   cli
   core/data_loading/overview
   core/data_preparation/overview
   core/models/overview
   core/scripts/overview
   core/visualization/overview
   interactive
   testing
