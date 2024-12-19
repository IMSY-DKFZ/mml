preprocessing
=============

Data preprocessing is the first transformation to data after loading from disk. It should comprise deterministic
one-one mappings that can e.g. be used to unify data (resizing operations to stack image tensors later on). The
preprocessing config determines the pipeline for this process. Note that preprocessing can be done on the fly -
which is ideal for exploration - but for number crunching a single call to the ``pp`` mode can drastically improve
efficiency (in exchange for disk space). Note that in contrast to :doc:`augmentations` only the
`albumentations <https://albumentations.ai/docs/>`_ backend is supported!

default
~~~~~~~

.. autoyaml:: preprocessing/default.yaml

example
~~~~~~~

.. autoyaml:: preprocessing/example.yaml

none
~~~~

.. autoyaml:: preprocessing/none.yaml

size224
~~~~~~~

.. autoyaml:: preprocessing/size224.yaml

size256
~~~~~~~

.. autoyaml:: preprocessing/size256.yaml

size336
~~~~~~~

.. autoyaml:: preprocessing/size336.yaml

size384
~~~~~~~

.. autoyaml:: preprocessing/size384.yaml

size512
~~~~~~~

.. autoyaml:: preprocessing/size512.yaml

size528
~~~~~~~

.. autoyaml:: preprocessing/size528.yaml