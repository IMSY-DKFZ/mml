# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin


# register plugin configs
# ATTENTION:
#  - this is not required if you are not providing any new entries in the "configs" folder next to this file
#  - you may then remove the configs folder and the lines below
#  - ANYWAY everything that should be registered/loaded from within MML automatically must be done from here
#  - IF you provide configs, replace INSERTPLUGINNAME with a suitable string for your plugin
#  - AND replace the "TODO" such that it matches with the name of this plugin
class MMLINSERTPLUGINNAMESearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Sets the search path for mml with copied config files
        search_path.append(provider="mml-TODO", path="pkg://mml_TODO.configs")


Plugins.instance().register(MMLINSERTPLUGINNAMESearchPathPlugin)
