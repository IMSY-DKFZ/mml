# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

from mml.core.data_loading.file_manager import MMLFileManager

MMLFileManager.add_assignment_path(
    obj_cls=int,
    key="dimension",
    path=Path("PROJ_PATH") / "DIMENSIONS" / "TASK_NAME" / "dims.txt",
    enable_numbering=True,
    reusable=True,
)
MMLFileManager.add_assignment_path(
    obj_cls=None,
    key="dimension_plot",
    path=Path("PROJ_PATH") / "PLOTS" / "DIMENSION" / "dims_versus_perf.html",
    enable_numbering=True,
    reusable=False,
)


# register plugin configs
class MMLDimensionalitySearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Sets the search path for mml with copied config files
        search_path.append(provider="mml-dimensionality", path="pkg://mml_dimensionality.configs")


Plugins.instance().register(MMLDimensionalitySearchPathPlugin)
