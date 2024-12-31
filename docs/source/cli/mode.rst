mode
====

The ``mode`` config group has a special role in ``mml`` and is required to be passed as the first argument (and without
``mode=...``) e.g. as ``mml train ...`` or ``mml create ...``. Some examples can be found at :doc:`/modes`. Note that any
additional configurations must be prepended by the ``mode`` keyword as usual (e.g. ``mode.subroutines=[x,y,z]``).
A mode basically maps to a subclass of :class:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler` using
the ``mode.scheduler._target_`` config entry. This specific scheduler will be instantiated by ``mml`` and the
:meth:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.run` method will be invoked to carry out any
computations. Sometimes different configs may point to the same scheduler with slightly different default parameters.
Such cases are marked below.

clean
~~~~~

.. autoyaml:: mode/clean.yaml

create
~~~~~~

.. autoyaml:: mode/create.yaml

downgrade
~~~~~~~~~

The downgrade config is a convenience entry to call the upgrade scheduler with the downgrade subroutine.

.. autoyaml:: mode/downgrade.yaml

info
~~~~

.. autoyaml:: mode/info.yaml

post
~~~~

.. autoyaml:: mode/post.yaml

pp
~~

.. autoyaml:: mode/pp.yaml

predict
~~~~~~~

The predict config is a convenience entry to call the train scheduler with the predict subroutine.

.. autoyaml:: mode/predict.yaml

test
~~~~

The test config is a convenience entry to call the train scheduler with the test subroutine.

.. autoyaml:: mode/test.yaml

tl
~~

.. autoyaml:: mode/tl.yaml

train
~~~~~

.. autoyaml:: mode/train.yaml

upgrade
~~~~~~~

.. autoyaml:: mode/upgrade.yaml