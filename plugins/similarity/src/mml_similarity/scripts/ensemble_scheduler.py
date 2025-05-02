# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from typing import Dict

import pandas as pd
import scipy
from mml_similarity.scripts.abstract_task_distance_scheduler import AbstractTaskDistanceScheduler
from omegaconf import DictConfig

from mml.core.scripts.exceptions import MMLMisconfigurationException

logger = logging.getLogger(__name__)


class EnsembleDistancesScheduler(AbstractTaskDistanceScheduler):
    """
    AbstractBaseScheduler implementation for the ensembling of task similarities setup. Includes the following
    subroutines:
    - distance

    Not to be confused with the ensembling of models by the PostProcessingScheduler!
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(EnsembleDistancesScheduler, self).__init__(cfg=cfg, available_subroutines=["distance"])
        self.source_distances: Dict[str, pd.DataFrame] = {}
        # load distances
        for dist in self.cfg.distance.sources:
            if dist not in self.fm.global_reusables:
                raise MMLMisconfigurationException(
                    f"Did not find {dist} in reusables. Make sure to set reuse.{dist} accordingly."
                )
            df = pd.read_csv(self.fm.global_reusables[dist], index_col=0)
            if self.cfg.distance.normalize:
                _data = scipy.stats.zscore(df.to_numpy().astype(float), axis=None, nan_policy="omit")
                df = pd.DataFrame(_data, columns=df.columns, index=df.index)
            self.source_distances[dist] = df
        if len(self.source_distances) < 2:
            raise MMLMisconfigurationException("At least two source distances are required to merge.")
        # set weights accordingly
        if self.cfg.distance.weights is None:
            # no weights given, weigh equally
            self.weights = [1.0] * len(self.source_distances)
        else:
            self.weights = self.cfg.distance.weights
        if len(self.weights) != len(self.source_distances):
            raise MMLMisconfigurationException("Lengths of weights and source distances does not match!")
        if any(not isinstance(w, float) for w in self.weights):
            raise MMLMisconfigurationException("Weights must all be floats!")
        if any(w <= 0.0 for w in self.weights):
            raise MMLMisconfigurationException("Non-positive weights are likely a typo!")

    def create_routine(self) -> None:
        """
        This scheduler implements only a single subroutine that will be called once.

        :return: None
        """
        # -- there is only a single command
        self.create_default_distances_routine()

    def compute_distance_impl(self, source_task_name: str, target_task_name: str) -> float:
        running_sum = 0.0
        for weight, (dist, df) in zip(self.weights, self.source_distances.items()):
            try:
                if df.isna().at[source_task_name, target_task_name]:
                    raise KeyError
                val = df.at[source_task_name, target_task_name]
            except KeyError:
                raise RuntimeError(
                    f"Distance {source_task_name} -> {target_task_name} not yet computed for distance {dist}."
                )
            running_sum += weight * val
        return running_sum

    def is_sym_distance(self) -> bool:
        # may be incorrect depending on sources
        return False
