Installation
============

The setup start with the preparation of the `virtual environment <Virtual environment_>`_, followed by
setting your personal system and users `local variables <Local variables_>`_. After `testing <Test installation_>`_
your installation you may want to follow some of the optional parts, like
`config copy <Config copy_>`_ within your project, `tab completion <Tab completion_>`_ to increase
CLI convenience or install any of the :doc:`plugins`.
For a quick start without manual installation you may also be interested in using the ``mml`` `docker image <Docker_>`_.


Virtual environment
-------------------

We recommend `conda <https://docs.conda.io/en/latest/miniconda.html>`_
as virtual environment manager, but remain fully ``pip`` compatible

.. code-block:: bash

    conda update -n base -c defaults conda  # OPTIONAL: update conda
    conda create --yes --name mml python=3.10  # create environment, choose python version
    conda activate mml  # activate environment, start runs later on with activated environment

Now install the core of `mml` via

.. code-block:: bash

    pip install mml-core

.. note::
    If you want to contribute to MML we suggest to install from source: clone the ``mml`` repository and install via
    ``pip install -e .[dev,docs]`` in editable mode with the extras for documentation generation and linting tools.

.. note::
    From version ``0.10.0`` the ``mml`` framework was split into the separate installable packages ``mml-core`` and
    diverse plugins. Please do **NOT** install as ``pip install ... mml`` anymore!

Local variables
---------------

``mml`` relies on a ``mml.env`` file for relevant environment variables. There are multiple possibilities to locate this:

 - within the ``mml`` installation itself (e.g. for ``mml`` developers, called the ``default`` location)
 - within your project folder (e.g. for separation of ``mml`` installations),
 - within your home folder or similar (e.g. for shared ``mml`` configs across installations)

The actual localization of the file is done as follows:

 - ``mml`` searches for a ``MML_ENV_PATH`` environment variable, if found follow this path
 - else check the installation path of the currently running instance and search there for ``src/mml/mml.env`` (the ``default`` location), if found follow this path
 - else raise an error

default location
~~~~~~~~~~~~~~~~

If you want to use the ``default`` location, go to ``PROJECT_ROOT/src/mml`` for cloned projects or your interpreters
site-packages for pip installed packages (you can find that with ``pip show mml-core | grep Location``), do the following

 * copy and rename ``example.env`` to ``mml.env``

new non default location
~~~~~~~~~~~~~~~~~~~~~~~~

If else you want to go with any non-``default`` location you can use ``mml-env-setup`` from the command line at the location
you want to place your ``mml.env`` file:

.. code-block:: bash

    mml-env-setup

Afterward make sure to set ``MML_ENV_PATH``, if you use conda environments, there is a
`convenience setup <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables>`_
for this:

.. code-block:: bash

    conda env config vars set MML_ENV_PATH=/path/to/your/mml.env
    # if your file is located at the current working directory, you may instead use
    # pwd | conda env config vars set MML_ENV_PATH=$(</dev/stdin)/mml.env
    # requires re-activation of environment
    conda activate mml

setting the variables
~~~~~~~~~~~~~~~~~~~~~

 - open ``mml.env`` in your preferred editor
 - set ``MML_DATA_PATH`` to the path you want to store downloaded or generated datasets later on
 - set ``MML_RESULTS_PATH`` to be the location you want to save your experiments in later on (plots, trained network parameters, calculated distances, etc.).
 - set ``MML_LOCAL_WORKERS`` to be the number of usable (virtual) cpu cores
 - all other variables are optional

re-use some previous config file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``MML_ENV_PATH``, if you use conda environments, there is a `convenience setup <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables>`_
for this:

.. code-block:: bash

    conda env config vars set MML_ENV_PATH=/path/to/your/mml.env
    # requires re-activation of environment
    conda activate mml

Test installation
-----------------

To test the installation you may simply call

.. code-block:: bash

    mml

from the command line and check the output.

.. _config-copy:

Config copy
-----------

Since version ``0.6`` ``mml`` supports using a separate configs folder than the one shipped within ``mml`` itself.
This comes in useful if you want to define new config files for your application and/or want to version control
a specific combination of configs that differ from the defaults.
If this is the case or you just want more fine-grained control on the config options, it is possible to create an own
``configs`` folder within your project to control ``mml`` behaviour from there (this is of course not necessary if
``mml`` has been cloned from the repository).
Conveniently this can be achieved by simply navigating to your desired configs root folder (likely your project root folder)
and type ``mml-copy-conf`` to navigate you through this process. Recall that your ``mml.env`` file (see
`above <Local variables_>`_) **remains** at the specified location.

.. note::
    Alternatively if you make your code installable, you can write :doc:`plugins` and just add custom config parts
    to the standard config groups of ``mml``!

Tab completion
--------------

The hydra config system allows for tab completion in multiple shells, see
`here <https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/>`_ for details.
Roughly as example for ``bash`` you can install mml tab completion with ``eval "$(mml -sc install=bash)"``.


Docker
------

.. note::
    The ``mml`` docker image is not yet published to a public container-registry. You may build it yourself though.

prerequisites
~~~~~~~~~~~~~

For building docker images and running containers you will need docker installed on your system. As ``mml`` depends on using a GPU you will need the nvidia-container-toolkit as well - see `install guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ for installation instructions and further information.


building the mml image
~~~~~~~~~~~~~~~~~~~~~~

We recommend using prebuilt images from the ``mml`` container-registry. However, if you need a custom configuration or just want to build the image yourself you can do so. Adapt the ``Dockerfile`` in the base directory of ``mml`` to your requirements. This will most probably be the used python version, as well as the number of workers and other parameters in the ``mml.env`` file.
Build the image by running:

.. code-block:: bash

    docker build . -t <image-name>


running the container
~~~~~~~~~~~~~~~~~~~~~

If you have access to the ``mml`` container-registry or built the image yourself and installed all necessary prerequisites you're ready to go.
To start your container just run:

.. code-block:: bash

    docker run -v <host-data-path>:/data -v <host-results-path>:/results --ipc=host --gpus=all -i -t <image-name>

Inside the container you can just start using ``mml`` like you would from your local bash.

.. note::
    The paths you use for running the docker container are mounted from your host computer, so data that's already present can be used directly and all changes in data and results are accessible from your host and persist after closing the container.