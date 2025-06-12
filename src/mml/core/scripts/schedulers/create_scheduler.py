# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

from _collections import OrderedDict
from omegaconf import DictConfig

import mml.core.data_preparation.task_creator
from mml.core.data_preparation.dset_creator import WIP_FILE_NAME
from mml.core.data_preparation.registry import (
    _DATASET_CREATORS,
    _TASKCREATORS,
    get_dset_creator,
    get_dset_for_task,
    get_task_creator,
)
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import TAG_SEP

logger = logging.getLogger(__name__)


class CreateScheduler(AbstractBaseScheduler):
    """
    AbstractBaseScheduler implementation for the Dataset and Task creation process. Includes the following subroutines:
    - dataset
    - task
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(CreateScheduler, self).__init__(cfg=cfg, available_subroutines=["dataset", "task"])
        assert self.cfg.preprocessing.id == "none", (
            f"Create mode only possible for preprocessing=none, gave "
            f"{self.cfg.preprocessing.id}. Use pp mode to create preprocessed"
            f" version afterwards."
        )
        # when starting mml from __main__.py plugins are already loaded, but here we check for other ways of starting
        if len(_TASKCREATORS) == 0 or len(_DATASET_CREATORS) == 0:
            raise RuntimeError(
                "Was not able to find any task creators and/or dataset creators! If you rely on plugins "
                "to provide data or task creators make sure to call "
                "mml.core.scripts.utils.load_mml_plugins() before."
            )
        if not isinstance(self.cfg.mode.n_folds, int) or self.cfg.mode.n_folds < 2:
            raise MMLMisconfigurationException("mode.n_folds must be an integer larger 1!")
        mml.core.data_preparation.task_creator.DEFAULT_N_FOLDS = self.cfg.mode.n_folds
        if not isinstance(self.cfg.mode.ensure_balancing, bool):
            raise MMLMisconfigurationException("mode.n_folds must be a boolean!")
        mml.core.data_preparation.task_creator.DEFAULT_ENSURE_BALANCED = self.cfg.mode.ensure_balancing

    def prepare_exp(self) -> None:
        """
        Prepare experiment expects tasks to be present and loads these into task factory container. Here this should
        be avoided.
        """
        logger.info("Starting task creation!")

    def create_routine(self):
        """
        This scheduler implements two subroutines, one for dataset preparation and one for task preparation.

        :return: None
        """
        # determine
        filtered_tasks = []
        for task in self.cfg.task_list:
            if TAG_SEP in task:
                logger.critical(
                    f"Task {task} is a tagged task and should not be created via create scheduler!"
                    f"Please create the base task and run any other mode with the tagged version to "
                    f"create the tagged task."
                )
                continue
            if task in self.fm.task_index.keys() and "none" in self.fm.task_index[task]:
                logger.info(
                    f"Skipping creation of task {task} because there already seems to be a RAW version of that."
                )
                continue
            filtered_tasks.append(task)
        # -- add download commands
        if "dataset" in self.subroutines:
            all_dsets_req = list(OrderedDict.fromkeys([get_dset_for_task(task) for task in filtered_tasks]))
            for dset in all_dsets_req:
                self.commands.append(self.prepare_dataset)
                self.params.append([dset])
        # -- add task creation commands
        if "task" in self.subroutines:
            for task in filtered_tasks:
                self.commands.append(self.create_task)
                self.params.append([task])

    def prepare_dataset(self, dset_name):
        logger.info("Starting preparing dataset " + self.highlight_text(dset_name))
        all_dsets = self.fm.get_all_dset_names()
        if dset_name in all_dsets["none"]:
            logger.info(
                f"Dataset {dset_name} already downloaded and prepared. If you encounter problems with this "
                f"dataset, delete {all_dsets['none'][dset_name]} and rerun."
            )
            dset_path = all_dsets["none"][dset_name]
            if (dset_path / WIP_FILE_NAME).exists():
                raise RuntimeError(
                    f"Something must have gone wrong during a previous data set creation for "
                    f"{dset_name}. It is necessary to remove the folder {dset_path} before retrying."
                )
        else:
            dset_creator = get_dset_creator(dset_name=dset_name)
            # run creator
            output = dset_creator()
            if isinstance(output, Path):
                logger.debug(f"Dataset created @ {output}.")
                dset_path = output
            else:
                raise RuntimeError(
                    f"Registered creator {dset_creator.__name__} for dataset {dset_name} did not "
                    f"provide a path, but {type(output)}."
                )
        logger.debug(f"Find dataset {dset_name} @ {dset_path}.")
        logger.info("Finished preparing dataset " + self.highlight_text(dset_name))

    def create_task(self, task_name):
        logger.info("Starting preparing task " + self.highlight_text(task_name))
        all_dsets = self.fm.get_all_dset_names()
        dset_name = get_dset_for_task(task_name=task_name)
        assert dset_name in all_dsets["none"], f"Dataset {dset_name} not available to start preparing {task_name}."
        dset_path = all_dsets["none"][dset_name]
        task_creator = get_task_creator(task_name)
        output = task_creator(dset_path=dset_path)
        if output is None or isinstance(output, Path):
            logger.debug(f"Task {task_name} fully created by task creator f{task_creator.__name__}.")
        else:
            raise RuntimeError(
                f"Registered creator {task_creator.__name__} for task {task_name} output did not match expectations, "
                f"it provided {type(output)}."
            )
        logger.info("Finished preparing task " + self.highlight_text(task_name))
