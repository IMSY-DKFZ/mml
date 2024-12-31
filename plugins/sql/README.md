# MML SQL plugin

This plugin provides SQL support for the optuna multiruns of MML.

# Install

Install the plugin with pip. In addition, you require the `mysqlclient` package.

```commandline
pip install mysqlclient
```

You also need to prepare the sql backend and add some information to your mml.env file.


## set up my sql database

Install MySQL and enter interactive MySQL session:


```commandline
    sudo apt-get install mysql-server default-libmysqlclient-dev
    sudo mysql -u root -p
```

Create MySQL user and database (you can use different names for database, user and password):

```commandline
    mysql> CREATE DATABASE IF NOT EXISTS mml_hpo;
    mysql> CREATE USER 'mml_user'@'%' IDENTIFIED WITH mysql_native_password BY 'password123';
    mysql> GRANT ALL PRVILEGES ON mml_hpo.* TO 'mml_user'@'%';
    mysql> FLUSH PRIVILEGES;
```

## set up secrets

This plugin expects the following secrets in the `mml.env` file (adapt to your previously chosen values):

```
export MML_MYSQL_USER=mml_user
export MML_MYSQL_PW=password123
export MML_MYSQL_PORT=3306
export MML_HOSTNAME_OF_MYSQL_HOST=localhost
export MML_MYSQL_DATABASE=mml_hpo
```

## grant access to other workstations

This part is optional and only required if you want other machines to access your local database (e.g. from a remote cluster):

```commandline
    sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf
    # change the line 'bind-adress = ...' to be a comment by adding a hashtag in front
    # do not forget to save changes!
    service mysql restart
```

# Usage

Instead of `hpo=default` just use `hpo=sql` when starting your `mml ... --multirun`. 

# Note

Since `mysqlclient` has a GPL-2.0 license this feature is outsourced to an internal plugin instead of `mml-core`.