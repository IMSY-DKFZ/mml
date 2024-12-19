mml-sql plugin
==============

Adds persistence and multi-node support to hyperparmeter optimization. After installation simply run your ``MML``
experiment with ``mml hpo=sql ...``.


One big advantage of using Optuna together with the MySQL backend is the possibility of starting multiple jobs on
different nodes in parallel, while aggregating results automatically. To make sure to attach another run to the same
Optuna study the study name as unique identifier has to be passed. The study name is created from the first call
and if not specified from CLI (``hydra.sweeper.study_name=MY_HPO_STUDY``) follows the default pattern ``{proj}_${now:%Y-%m-%d_%H-%M-%S}`` (
e.g. ``MY_HPO_PROJ_2021-06-29_09-58-42``). The study name is logged at the very beginning of a run. Attaching another run to the same study
works as follows:

.. code-block:: bash

    mml MODE_WITH_RETURN_VALUE proj=MY_HPO_PROJ hpo=sql hydra.sweeper.study_name=MY_HPO_PROJ_2021-06-29_09-58-42 search_space=MY_SEARCH_SPACE --multirun

Viewing the current status of the study, including top results can be done from info-mode as follows:

.. code-block:: bash

    mml info proj=MY_HPO_PROJ hpo=sql mode.study_name=MY_HPO_PROJ_2021-06-29_09-58-42 tasks=none

After the study has run, you may want to use the best results that the study offered. You may either manually create
a config file with the parameters identified via info mode, or use the ``use_best_params`` functionality:

.. code-block:: bash

    mml PROBABLY_SAME_MODE proj=ANY_PROJ use_best_params=STUDY_NAME

Note that these parameters are applied last after hydra config compilation and overwrite any other given config values.
So if a search space comprised ``example.parameter`` providing both ``example.parameter=42`` and ``use_best_params`` will
ignore the given value of 42.

install
~~~~~~~

Install MySQL and enter interactive MySQL session:

.. code-block:: bash

    sudo apt-get install mysql-server default-libmysqlclient-dev
    sudo mysql -u root -p

Create MySQL user and database (you can use different names for database, user and password):

.. code-block:: bash

    mysql> CREATE DATABASE IF NOT EXISTS mml_hpo;
    mysql> CREATE USER 'mml_user'@'%' IDENTIFIED WITH mysql_native_password BY 'password123';
    mysql> GRANT ALL PRVILEGES ON mml_hpo.* TO 'mml_user'@'%';
    mysql> FLUSH PRIVILEGES;

Make sure to set the MySQL variables accordingly to your chosen values within ``mml.env`` (see :doc:`../../install`):

.. code-block::

    export MML_MYSQL_USER=mml_user
    export MML_MYSQL_PW=password123
    export MML_MYSQL_PORT=3306
    export MML_HOSTNAME_OF_MYSQL_HOST=localhost
    export MML_MYSQL_DATABASE=mml_hpo

access
~~~~~~

This part is optional and only required if you want other machines to access your local database (e.g. from a remote cluster):

.. code-block:: bash

    sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf
    # change the line 'bind-adress = ...' to be a comment by adding a hashtag in front
    # do not forget to save changes!
    service mysql restart