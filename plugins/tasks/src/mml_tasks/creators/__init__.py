# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging

from mml.core.data_preparation.registry import _DATASET_CREATORS, _TASKCREATORS

_pre_length = len(_DATASET_CREATORS), len(_TASKCREATORS)

import mml_tasks.creators.classification  # noqa: F401, E402
import mml_tasks.creators.segmentation  # noqa: F401, E402

logger = logging.getLogger(__name__)

logger.debug(
    f"Detected {len(_DATASET_CREATORS) - _pre_length[0]} dataset creators and "
    f"{len(_TASKCREATORS) - _pre_length[1]} task creator functions from mml_tasks plugin."
)
