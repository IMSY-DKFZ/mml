# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import lightning
import lightning.pytorch.tuner.lr_finder
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, TQDMProgressBar

from mml.core.scripts.utils import LearningPhase

logger = logging.getLogger(__name__)


class StopAfterKeyboardInterrupt(lightning.Callback):
    """
    Ensures pytorch lightning to really shut down after keyboard interrupt. This is the new variant of this Callback
    aimed to be used by most recent pytorch lightning versions.
    """

    def on_exception(
        self, trainer: "lightning.Trainer", pl_module: "lightning.LightningModule", exception: BaseException
    ) -> None:
        if trainer.interrupted and isinstance(exception, KeyboardInterrupt):
            raise InterruptedError(
                "Trainer has been interrupted by keyboard! "
                "Will stop running MML - no graceful shutdown, ongoing epoch results are lost! "
                "Run MML in continue mode to start from the checkpoint of last epochs end."
            )


class MetricsTrackerCallback(lightning.Callback):
    """
    Keeps track of all metrics, at the end of each epoch.
    """

    def __init__(self):
        self.metrics: List[Dict[str, float]] = []

    def state_dict(self) -> Dict[str, Any]:
        return {"metrics": self.metrics}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.metrics = state_dict["metrics"]

    def on_train_epoch_end(self, trainer: "lightning.Trainer", pl_module: "lightning.LightningModule") -> None:
        self.copy_metrics(trainer=trainer, phase=LearningPhase.TRAIN)

    def on_validation_epoch_end(self, trainer: "lightning.Trainer", pl_module: "lightning.LightningModule") -> None:
        self.copy_metrics(trainer=trainer, phase=LearningPhase.VAL)

    # we need to gather test metrics only at the end (after module.test_epoch_end) to wait for bootstrapped computation
    def on_test_end(self, trainer: "lightning.Trainer", pl_module: "lightning.LightningModule") -> None:
        self.copy_metrics(trainer=trainer, phase=LearningPhase.TEST)

    def copy_metrics(self, trainer: lightning.Trainer, phase: LearningPhase) -> None:
        phase_metrics = copy.deepcopy(trainer.callback_metrics)  # Dict[str, torch.Tensor]
        phase_metrics = {
            name: metric_tensor.item()
            for name, metric_tensor in phase_metrics.items()
            if name.startswith(str(phase.value))
        }
        logger.debug(f"{phase=}, logged metrics {phase_metrics.keys()}")
        if len(self.metrics) == trainer.current_epoch:
            # first time copying for this epoch
            self.metrics.append(phase_metrics)
        elif len(self.metrics) == trainer.current_epoch + 1:
            # updating this epoch, for example adding train to val metrics
            self.metrics[trainer.current_epoch].update(phase_metrics)
        else:
            # we might have missed some epochs?
            diff = trainer.current_epoch - len(self.metrics)
            logger.error(f"There is a discrepancy of {diff} between metrics recorded and the current epoch!")
            for _ in range(diff):
                self.metrics.append({})
            self.metrics.append(phase_metrics)


class MMLRichProgressBar(RichProgressBar):
    """
    Slight modification of the Lightning rich progress bar, showing the correct experiment name.
    """

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["exp"] = "/".join(Path(os.getcwd()).parts[-2:])
        return items


class MMLTQDMProgressBar(TQDMProgressBar):
    """
    Slight modification of the Lightning tqdm progress bar, showing the correct experiment name.
    """

    def __init__(self, refresh_rate=1):
        super().__init__(refresh_rate=refresh_rate)
        self.experiment_name = "/".join(Path(os.getcwd()).parts[-2:])

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["exp"] = self.experiment_name
        return items


class MMLModelCheckpoint(ModelCheckpoint):
    """
    Slight modification of the lightning ModelCheckpoint (see
    https://github.com/Lightning-AI/pytorch-lightning/issues/20245).
    """

    def _save_last_checkpoint(self, trainer: "lightning.Trainer", monitor_candidates: Dict[str, torch.Tensor]) -> None:
        """Only update last checkpoint in case there has just been a new checkpoint."""
        if self._last_global_step_saved == trainer.global_step:
            super()._save_last_checkpoint(trainer=trainer, monitor_candidates=monitor_candidates)

    def on_train_epoch_end(self, trainer: "lightning.Trainer", pl_module: "lightning.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
                self._save_last_checkpoint(trainer, monitor_candidates)

    def on_validation_end(self, trainer: "lightning.Trainer", pl_module: "lightning.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                monitor_candidates = self._monitor_candidates(trainer)
                self._save_topk_checkpoint(trainer, monitor_candidates)
                self._save_last_checkpoint(trainer, monitor_candidates)
