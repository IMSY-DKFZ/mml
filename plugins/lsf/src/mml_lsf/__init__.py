# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from mml_lsf.requirements import LSFSubmissionRequirements
from mml_lsf.runner import LSFJobRunner

VERSION = (0, 5, 1)
__version__ = ".".join(map(str, VERSION))
__all__ = ["__version__", "LSFJobRunner", "LSFSubmissionRequirements"]
