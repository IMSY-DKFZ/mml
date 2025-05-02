# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging

import numpy as np
from mml_dimensionality.scripts.mle import calc_mle
from mml_dimensionality.visualization import plot_dimensions_versus_baseline
from omegaconf import DictConfig

from mml.core.scripts.decorators import beta
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import LearningPhase

logger = logging.getLogger(__name__)


@beta("Dimension estimation is in beta!")
class DimensionalityScheduler(AbstractBaseScheduler):
    """ "
    AbstractBaseScheduler implementation for the estimation of task dimensionality. Includes the following subroutines:
    - estimate

    The method and partially also the code is reused from:
    "The Intrinsic Dimensionaity of Images and Its Impact On Learning"
    by Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, Tom Goldstein (ICLR 2021, spotlight)
    code: https://github.com/ppope/dimensions
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(DimensionalityScheduler, self).__init__(cfg=cfg, available_subroutines=["estimate", "plot"])
        if self.cfg.mode.k < 3:
            raise MMLMisconfigurationException("mode.k must be at least 3.")
        if self.cfg.augmentations.normalization is not None:
            raise MMLMisconfigurationException(
                "Must deactivate normalization for dimensionality estimation. Best set augmentations=no_norm."
            )

    def create_routine(self):
        """
        This scheduler implements one subroutine, which estimates a task's dimensionality.

        :return: None
        """
        # -- add preprocess command
        if "estimate" in self.subroutines:
            if self.pivot:
                logger.info("Dimensionality mode with pivot task will only estimate this task!")
                self.commands.append(self.estimate_task_dimensionality)
                self.params.append([self.pivot])
            else:
                for task in self.cfg.task_list:
                    self.commands.append(self.estimate_task_dimensionality)
                    self.params.append([task])

    def before_finishing_hook(self):
        if "plot" in self.subroutines and not self.pivot:
            task_list = []
            # check if performances are available
            for task in self.cfg.task_list:
                struct = self.get_struct(task_name=task)
                if len(struct.models) > 0:
                    task_list.append(struct)
                else:
                    logger.error(
                        f"No performance found for task {task} to produce dimension versus performance plot, "
                        f"you may want to use reuse.models=SOME_PROJ_NAME to provide performances."
                    )

            if len(task_list) > 0:
                plot_path = self.fm.construct_saving_path(obj=None, key="dimension_plot")
                plot_dimensions_versus_baseline(all_tasks=task_list, store_path=plot_path)
                logger.info(f"Plotted task dimensionality against performance at {plot_path}.")

    def estimate_task_dimensionality(self, task_name: str):
        logger.info("Starting estimating dimensionality data for task " + self.highlight_text(task_name))
        task_struct = self.get_struct(task_name)
        datamodule = self.create_datamodule(task_structs=task_struct)
        datamodule.setup(stage="fit")
        dataset = datamodule.task_datasets[task_name][LearningPhase.TRAIN]
        val = calc_mle(cfg=self.cfg, dataset=dataset)
        dim = -1 if np.isnan(val) else int(val)
        path = self.fm.construct_saving_path(obj=dim, key="dimension", task_name=task_struct.name)
        with open(path, "w+") as f:
            f.write(str(dim))
        task_struct.paths["dimension"] = path
        self.task_factory.dump()
        logger.info("Finished dimensionality estimation for task " + self.highlight_text(task_name))
