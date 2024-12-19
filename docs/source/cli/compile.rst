compile
=======

The compile config group determines whether to activate ``torch.compile`` upon the model. This may speed up model
training but has limitations (it requires an initial optimization routine when the model starts training) that leads to
overhead which may not be regained by (potentially lacking) training speed improvements. Furthermore there have been
issues with the :ref:`continue-option` functionality. Thus it is not enabled by default.

.. autoyaml:: compile/default.yaml
