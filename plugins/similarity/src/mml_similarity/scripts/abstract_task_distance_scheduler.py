# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import shutil
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from mml_similarity.visualization.plot_2D import plot_task_embedding
from omegaconf import DictConfig

from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import TAG_SEP

logger = logging.getLogger(__name__)


class AbstractTaskDistanceScheduler(AbstractBaseScheduler, ABC):
    def __init__(self, cfg: DictConfig, available_subroutines: List[str]):
        assert "distance" in available_subroutines
        self.distance_measure = cfg.distance.name
        super(AbstractTaskDistanceScheduler, self).__init__(
            cfg=cfg, available_subroutines=available_subroutines + ["plot"]
        )
        self.distance_file = self.fm.construct_saving_path(obj=pd.DataFrame(), key=self.distance_measure)

    def load_distances(self) -> pd.DataFrame:
        """
        Convenience function to load distance file.

        :return: dataframe of distances
        """
        assert self.distance_file.exists(), "Distance file not found!"
        df = pd.read_csv(self.distance_file, index_col=0)
        return df

    def create_default_distances_routine(self) -> None:
        """
        To be leveraged in any inheriting scheduler to add all necessary distance computation commands. Simply implement
        :meth:`compute_distance_impl` in the inheriting scheduler and call :meth:`create_default_distances_routine` as
        part of  :meth:`create_routine` when the `distance` subroutine may be translated to commands.
        """
        # -- add distance computation commands
        if "distance" in self.subroutines:
            # pivot case - only compute cases involving transferring to pivot task
            if self.pivot:
                for source in self.cfg.task_list:
                    self.commands.append(self._compute_distance)
                    self.params.append([source, self.pivot])
            # non pivot variant of a symmetric distance, simplify to necessary computations
            elif self.is_sym_distance():
                for source, target in combinations(self.cfg.task_list, 2):
                    self.commands.append(self._compute_distance)
                    self.params.append([source, target])
            # non pivot and non symmetric distance, need to add all possible computations
            else:
                for source in self.cfg.task_list:
                    for target in self.cfg.task_list:
                        self.commands.append(self._compute_distance)
                        self.params.append([source, target])

    def _compute_distance(self, source: str, target: str) -> None:
        """
        Method of AbstractTaskDistanceScheduler that will wrap :meth:`compute_distance_impl`. No need to manipulate
        in downstream schedulers. If the scaffold provided does not suit, simply do not call
        :meth:`create_default_distances_routine`.

        :param str source: source task name
        :param str target: target task name
        :return: None, will save the result of :meth:`compute_distance_impl` in the internal data frame
        """
        logger.info(
            "Starting distance computation for tasks "
            + self.highlight_text(source)
            + " -> "
            + self.highlight_text(target)
        )
        distance = self.compute_distance_impl(source_task_name=source, target_task_name=target)
        # saving of distance
        df = self.load_distances()
        # x = row (=index) and y = column within dataframe represent distance from x to y
        df.at[source, target] = distance
        df.to_csv(self.distance_file)
        logger.info(
            "Successfully finished distance computation for tasks "
            + self.highlight_text(source)
            + " -> "
            + self.highlight_text(target)
        )

    def compute_distance_impl(self, source_task_name: str, target_task_name: str) -> float:
        """
        Computes the distance between two tasks. To be implemented in downstream scheduler.

        :param source_task_name: name of the source task
        :param target_task_name: name of the target task
        :return: float value to represent the distance between two tasks
        """
        raise NotImplementedError(
            'If using "create_default_distances_routine" you must implement '
            '"compute_distance_impl" - do not manipulate the wrapper "_compute_distance"!'
        )

    def after_preparation_hook(self) -> None:
        # load results file
        if "distance" in self.subroutines:
            # if in continue mode, the files may already exist with some results, so load and check
            if self.continue_status:
                # check if file matches experiment setting
                df = self.load_distances()
                logger.info(
                    f"Found existing results file and will continue using it! {sum(list(df.count()))} "
                    f"out of {df.shape[0] * df.shape[1]} entries already present."
                )
            else:
                if self.distance_file.exists():
                    backup_path = self.fm.construct_saving_path(
                        obj=None, key="backup", file_name=self.distance_file.name
                    )
                    shutil.copyfile(self.distance_file, backup_path)
                    warnings.warn(
                        f"Found existing distance file (type {self.distance_measure}). These are not "
                        f"versioned, but will be extended / overridden with repeated mml runs. The updated "
                        f"file is at {self.distance_file}, for your convenience a backup is stored at "
                        f"{backup_path}."
                    )
                else:
                    # first run for this distance measure in this project - create and save an empty frame
                    df = pd.DataFrame()
                    df.to_csv(self.distance_file)

    def before_finishing_hook(self) -> None:
        if "distance" in self.subroutines:
            # if skipped before in symmetric distance case, now fill the missing entries in result file
            if self.is_sym_distance():
                logger.info("Filling in symmetric results in results dataframe.")
                df = self.load_distances()
                # adding indirectly given distances for symmetric case
                for source in df.index:
                    for target in df.columns:
                        # x = row (=index) and y = column within dataframe represent distance from x to y
                        if source == target:
                            df.at[source, target] = 0.0
                        elif not df.isna().at[source, target]:
                            df.at[target, source] = df.at[source, target]
                        elif not df.isna().at[target, source]:
                            df.at[source, target] = df.at[target, source]
                # saving
                df.to_csv(self.distance_file)
            logger.info(f"Computed all distances. Results of may be found at: {self.distance_file}.")
        if "plot" in self.subroutines:
            # if chosen in the configs do a final plotting
            plot_task_embedding(
                distances=self.load_distances(),
                structs=self.task_factory.container,
                plot_config=self.cfg.plotting.distance,
                distance_key=self.distance_measure,
            )
        # determine return value that can be used for optimization, only available if no pivot was used
        if "distance" in self.subroutines and not self.pivot:
            # find identity, duplicates , subsets variants of tasks
            task_groups = defaultdict(list)
            for task in self.cfg.task_list:
                if any(
                    [
                        mod_string in task
                        for mod_string in [TAG_SEP + "identity", TAG_SEP + "duplicate", TAG_SEP + "subset"]
                    ]
                ):
                    base_name = task[: task.find(TAG_SEP)]
                else:
                    base_name = task
                task_groups[base_name].append(task)
            # check if there are actual groups for optimization
            if all([len(group) == 1 for group in task_groups.values()]):
                # no groups return a "bad" value
                logger.debug("No task grouping detected, this prohibits optimizing upon task similarity.")
                self.return_value = 1.0
            else:
                # calculate mean distances within each group
                intra_group_distances = {}
                df = self.load_distances()
                for base_name in task_groups:
                    group = task_groups[base_name]
                    if len(group) == 1:
                        continue
                    intra_group_distances[base_name] = []
                    for task_1, task_2 in combinations(group, 2):
                        intra_group_distances[base_name].append(df.at[task_1, task_2])
                        intra_group_distances[base_name].append(df.at[task_2, task_1])
                group_means = {
                    base_name: np.mean(intra_group_distances[base_name]) for base_name in intra_group_distances
                }
                logger.debug(
                    f"Detected {len(group_means)} task groups that comprise: "
                    f"{[task_groups[base_name] for base_name in group_means]}."
                )
                # return value will be average group mean (should be small) divided by overall mean (should be high)
                overall_mean = df.mask(np.eye(len(df.index), dtype=bool)).replace([np.inf], np.nan).mean().mean()
                self.return_value = np.mean(list(group_means.values())) / overall_mean

    @abstractmethod
    def is_sym_distance(self) -> bool:
        """
        Helper function used to identify if distances have to be calculated in both directions.

        :return: (bool) True is indicating symmetric behaviour
        """
        pass
