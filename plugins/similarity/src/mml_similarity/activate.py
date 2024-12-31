# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from mml_similarity.scripts.sample_based_scheduler import SAMPLE_BASED_MODES

from mml.core.data_loading.file_manager import MMLFileManager

# set distance file and plotting file for all distance measures
for measure in ["fed", "ens", "semantic"] + SAMPLE_BASED_MODES:
    MMLFileManager.add_assignment_path(
        obj_cls=pd.DataFrame,
        key=measure,
        path=Path("PROJ_PATH") / "DISTANCES" / measure / "distances.csv",
        enable_numbering=False,
        reusable=MMLFileManager.GLOBAL_REUSABLE,
    )
    MMLFileManager.add_assignment_path(
        obj_cls=None,
        key=f"2D_{measure}",
        path=Path("PROJ_PATH") / "PLOTS" / "task_space" / f"graph-{measure}.png",
        enable_numbering=True,
    )

# the fed method requires to store fim and models tuned on the fc
MMLFileManager.add_assignment_path(
    obj_cls=dict,
    key="fim",
    path=Path("PROJ_PATH") / "FIMS" / "TASK_NAME" / "fim.pkl",
    enable_numbering=True,
    reusable=True,
)
MMLFileManager.add_assignment_path(
    obj_cls=torch.nn.Module,
    key="fc_tuned",
    path=Path("PROJ_PATH") / "FC_TUNED" / "TASK_NAME" / "fc_tuned.pth",
    enable_numbering=True,
    reusable=True,
)
# sample based methods require to extract features
MMLFileManager.add_assignment_path(
    obj_cls=np.ndarray,
    key="features",
    path=Path("PROJ_PATH") / "FEATURES" / "TASK_NAME" / "features.npy",
    enable_numbering=True,
    reusable=True,
)


# register plugin configs
class MMLSimilaritySearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Sets the search path for mml with copied config files
        search_path.append(provider="mml-similarity", path="pkg://mml_similarity.configs")


Plugins.instance().register(MMLSimilaritySearchPathPlugin)
