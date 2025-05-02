# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import random

import lightning
import torch
from omegaconf import DictConfig

from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.schedulers.train_scheduler import TrainingScheduler

logger = logging.getLogger(__name__)


class TransferScheduler(TrainingScheduler):
    """
    Inherited from TrainingScheduler this scheduler supports the same subroutines:
    - model training
    - model prediction
    - model testing

    But it adds the option to finetune an existing model by choosing a mode.pretrain_task in the config.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        if self.cfg.mode.pretrain_task not in self.cfg.task_list:
            raise MMLMisconfigurationException(
                f"Current pretraining source {self.cfg.mode.pretrain_task} was not "
                f"found within tasks. Make sure consistency."
            )

    def before_training_hook(
        self,
        datamodule: lightning.LightningDataModule,
        model: lightning.LightningModule,
        trainer: lightning.Trainer,
        fold: int,
        task_name: str,
    ) -> None:
        """
        Implements the weight loading logic.
        """
        # load pretrained weights
        source_task_struct = self.get_struct(self.cfg.mode.pretrain_task)
        if len(source_task_struct.models) == 0:
            raise MMLMisconfigurationException(
                f"No previous trained model for task {self.cfg.mode.pretrain_task} "
                f"found, either change mode.pretrain_task value or run train "
                f"on the pretrain task beforehand in this project or use the "
                f"reuse.models option to load from a previous project."
            )
        logger.info(
            f"Found {len(source_task_struct.models)} existing models for pretraining task "
            f"{self.cfg.mode.pretrain_task}."
        )
        if self.cfg.mode.model_selection == "performance":
            storage = min(source_task_struct.models, key=lambda x: x.performance)
            logger.info(f"Chose pretrain model based on performance (best: {storage.performance}).")
        elif self.cfg.mode.model_selection == "random":
            select_idx = random.randrange(len(source_task_struct.models))
            storage = source_task_struct.models[select_idx]
            logger.info(f"Chose pretrain model randomly (rolled: {select_idx}).")
        elif self.cfg.mode.model_selection == "created":
            storage = max(source_task_struct.models, key=lambda x: x.created)
            logger.info(f"Chose pretrain model based on creation date (latest: {storage.created}).")
        else:
            raise MMLMisconfigurationException("mode.model_selection must be one of [performance, random, created].")
        state = torch.load(f=storage.parameters, weights_only=False)["state_dict"]
        # remove metrics and heads
        to_be_removed = []
        for key in state.keys():
            if any(
                [
                    key.startswith(prefix)
                    for prefix in [
                        "model.heads",
                        "train_metrics",
                        "val_metrics",
                        "test_metrics",
                        "train_cms",
                        "val_cms",
                        "test_cms",
                    ]
                ]
            ):
                to_be_removed.append(key)
        for key in to_be_removed:
            del state[key]
        # load module and continue with training
        model.load_state_dict(state_dict=state, strict=False)
        logger.info(f"Successfully loaded pretrain weights from task {self.cfg.mode.pretrain_task}.")
        if self.cfg.mode.freeze:
            model.model.freeze_backbone()
            logger.info("Froze model backbone and continue with linear probing of model heads.")

    def after_training_hook(
        self,
        datamodule: lightning.LightningDataModule,
        model: lightning.LightningModule,
        trainer: lightning.Trainer,
        fold: int,
        task_name: str,
    ) -> None:
        """
        If necessary unfreezes the model backbone.
        """
        if self.cfg.mode.freeze:
            model.model.unfreeze_backbone()
            logger.info("Unfroze model backbone after linear probing of model heads.")
