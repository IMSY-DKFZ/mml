logging
=======

The ``logging`` config group allows to modify the logging behaviour of mml. Next to sum options at the top level, there
are three sub-configurations:

  * ``exp_logger`` will determine the experiment logger used by ``lightning``
  * ``notifier`` will determine whether and how certain events will be messaged
  * ``render`` will determine the rendering backend for logging


default
~~~~~~~
The default top level configuration is stored in ``log.yaml``.

.. autoyaml:: logging/log.yaml


exp_logger
----------

Currently ``mml`` only provides a config for `tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_.

tensorboard
~~~~~~~~~~~

.. autoyaml:: logging/exp_logger/tensorboard.yaml


notifier
--------

``mml`` ships with two notifiers, but can easily be extended. There are three configuration files: ``none.yaml`` (the
default, which does not provide any notifier), ``slack.yaml`` which allows notification via Slack and ``email.yaml`` for
email notification. Multiple notifiers can be combined via ``logging/notifier=[email,slack]``. For each notifier you
can independently determine the events to send notifications (e.g. ``logging.notifier.slack.on_start=true``).

slack
~~~~~

.. autoyaml:: logging/notifier/slack.yaml

email
~~~~~

.. autoyaml:: logging/notifier/email.yaml

render
------

The backend for rendering the logs. Can be either `colorlog <https://github.com/borntyping/python-colorlog>`_
or `rich <https://github.com/Textualize/rich>`_. The default is colorlog and support for ``rich`` might not be as
throughout as for ``colorlog``. Change the renderer via ``logging/render=rich``.