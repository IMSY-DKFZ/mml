optimizer
=========

The ``optimizer`` config group determines the ``torch-optim`` optimizer used for backpropagation during model training.
For now only a single optimizer and single parameter group is supported.


adam
~~~~
The default optimizer by ``mml``.

.. autoyaml:: optimizer/adam.yaml

sgd
~~~

.. autoyaml:: optimizer/sgd.yaml