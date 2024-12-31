# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

modules = Path(__file__).parent.glob("*.py")
# this will automatically import all valid submodules, so the tasks are registered automatically
__all__ = [module.stem for module in modules if module.is_file() and not module.stem.startswith("_")]

from . import *  # noqa F403, F401, E402
