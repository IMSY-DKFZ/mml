# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import mml_tasks.creators  # noqa
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin


# register plugin configs
class MMLTasksSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Sets the search path for mml with copied config files
        search_path.append(provider="mml-tasks", path="pkg://mml_tasks.configs")


Plugins.instance().register(MMLTasksSearchPathPlugin)
