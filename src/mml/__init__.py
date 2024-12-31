# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import os

# deactivate warning on non-up-to-date albumentations version
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

VERSION = (1, 0, 0)
__version__ = ".".join(map(str, VERSION))
__all__ = ["__version__"]
