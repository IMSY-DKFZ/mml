# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging

from mml_similarity.scripts.abstract_task_distance_scheduler import AbstractTaskDistanceScheduler
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class SemanticScheduler(AbstractTaskDistanceScheduler):
    """
    AbstractBaseScheduler implementation for semantic simiarlity based upon task keywords. Includes the following
    subroutines:
    - distance computation
    - (plotting, not as full subroutine, but as part of the finish experiment routine)
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(SemanticScheduler, self).__init__(cfg=cfg, available_subroutines=["distance"])

    def create_routine(self) -> None:
        """
        This scheduler implements a single subroutine (distance).

        :return: None
        """
        # -- add distance computation commands
        self.create_default_distances_routine()

    def compute_distance_impl(self, source_task_name: str, target_task_name: str) -> float:
        # load datasets
        source_struct = self.get_struct(source_task_name)
        target_struct = self.get_struct(target_task_name)

        union = set(source_struct.keywords).union(target_struct.keywords)
        intersection = set(source_struct.keywords).intersection(target_struct.keywords)

        return 1 - (len(intersection) / len(union))

    def is_sym_distance(self) -> bool:
        return True
