mml-lsf plugin
==============

Features for (DKFZ-)LSF in combination with ``MML``. Currently supports:

- derive number of workers by host (does not interfere with local host settings)
- plan :class:`~mml.interactive.planning.MMLJobDescription` accordingly with ``LSFSubmissionRequirements``
- submit jobs from notebooks leveraging ``LSFJobRunner``

More details can be found in the `README <https://github.com/IMSY-DKFZ/mml/tree/main/plugins/lsf>`_ of the plugin.