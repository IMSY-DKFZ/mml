# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

"""
Adapted from
    https://github.com/veritable-tech/pytorch-lightning-spells/blob/master/pytorch_lightning_spells/callbacks.py
Which was adapted from
    https://github.com/rwightman/pytorch-image-models/blob/8c9814e3f500e8b37aae86dd4db10aba2c295bd2/timm/data/mixup.py
Which was partly adapted from
    https://github.com/clovaai/CutMix-PyTorch

    Papers:
        MixUp: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

    References:
        `rwightman/pytorch-image-models/`
        `veritable-tech/pytorch-lightning-spells`
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import lightning
import numpy as np
import torch
from torchmetrics import MetricCollection

from mml.core.data_loading.task_attributes import Modality, TaskType
from mml.core.data_loading.task_struct import TaskStruct

logger = logging.getLogger(__name__)


class MixingCallback(lightning.Callback):
    def __init__(self, alpha: float = 0.4, label_smoothing: float = 0.0):
        """
        Base class for MML data mixing callbacks.

        :param float alpha: controls the mixing factor (between 0 and 1)
        :param float label_smoothing: if greater than 0 activates label smoothing
        """
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.target_modalities: Dict[str, str] = {}
        self.num_classes: Dict[str, int] = {}

    def setup(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule, stage: Optional[str] = None
    ) -> None:
        """
        During set up the task structs are inspected for compatibility and the number of classes is extracted. Due to
        data mixing during training the torchmetrics train metrics are deactivated.
        """
        task_structs: List[TaskStruct] = trainer.datamodule.task_structs
        for struct in task_structs:
            task = struct.name
            if struct.task_type != TaskType.CLASSIFICATION:
                warnings.warn(f"DataMixing not supported for task type {struct.task_type} yet")
                continue
            self.target_modalities[task] = pl_module.targets[task]
            self.num_classes[task] = struct.num_classes
            logger.info(f"Deactivating train metrics for classification task {task} due to data mixing.")
            pl_module.train_metrics[task] = MetricCollection([])

    def smooth_one_hot(self, x: torch.Tensor, task: str) -> torch.Tensor:
        """
        One hot encoding for a tasks targets with smoothing controlled via
        :attr:`~mml.core.data_loading.augmentations.mixup_cutmix.MixingCallback.label_smoothing`.

        :param torch.Tensor x: batched task targets
        :param str task: name of the task
        :return: one hot encoded task targets, smoothed if
          :attr:`~mml.core.data_loading.augmentations.mixup_cutmix.MixingCallback.label_smoothing` > 0
        """
        off_value = self.label_smoothing / self.num_classes[task]
        on_value = 1.0 - self.label_smoothing + off_value
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], self.num_classes[task]), off_value, device=x.device).scatter_(1, x, on_value)

    def mixup_targets(self, targets: torch.Tensor, lambdas: np.ndarray, task: str) -> torch.Tensor:
        """
        Takes care of mixing targets.

        :param torch.Tensor targets: batched task targets, first target will be mixed with last, second with second
          to last, etc.
        :param np.ndarray lambdas: actual fractions of each mix
        :param str task: name of the task
        :return: (optionally) smoothed and then mixed targets
        """
        y1 = self.smooth_one_hot(targets, task=task)
        y2 = self.smooth_one_hot(targets.flip(0), task=task)
        lam = targets.new(lambdas).view(-1, *[1 for _ in range(len(y1.size()) - 1)])
        return y1 * lam + y2 * (1.0 - lam)


class CutMixCallback(MixingCallback):
    def __init__(self, alpha: float = 0.4, label_smoothing: float = 0.0, minmax: Optional[Tuple[float, float]] = None):
        """
        Callback that performs CutMix augmentation on training batches. Incorporates two strategies:
          * bounding box sizes are either controlled via a beta ditribution controlled by parameter alpha
          * or if set minmax controls relative bbox ratios and the distribution is more uniformly

        :param float alpha: if minmax is None this value controls the beta distribution
        :param float label_smoothing: if greater than 0 activates label smoothing
        :param Optional[Tuple[float, float]] minmax: min and max bbox ratios (as percent of image size),
          typical values for minmax are in the .2-.3 for min and .8-.9 range for max.
        """
        super().__init__(alpha=alpha, label_smoothing=label_smoothing)
        self.ratio_minmax = minmax

    def on_train_batch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        batch: Dict[str, Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Will be triggered on each training batch."""
        for task in batch:
            images: torch.Tensor = batch[task][Modality.IMAGE.value]
            targets: torch.Tensor = batch[task][self.target_modalities[task]]
            images_flipped = images.flip(0).clone()
            lambdas = np.random.beta(self.alpha, self.alpha, images.size(0))
            for i in range(images.shape[0]):
                (yl, yh, xl, xh), lambd_tmp = self.get_bbox_and_lam(images.shape, lambdas[i])
                lambdas[i] = lambd_tmp
                # fill in the cut regions
                images[i, :, yl:yh, xl:xh] = images_flipped[i, :, yl:yh, xl:xh]
            new_targets = self.mixup_targets(targets, lambdas, task=task)
            batch[task][Modality.IMAGE.value] = images
            batch[task][self.target_modalities[task]] = new_targets

    def get_bbox_and_lam(
        self, img_shape: Tuple, lam: float
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Generate bbox and apply lambda correction."""
        if self.ratio_minmax is not None:
            yl, yu, xl, xu = self.rand_bbox_minmax(img_shape)
        else:
            yl, yu, xl, xu = self.rand_bbox(img_shape, lam)
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
        return (yl, yu, xl, xu), lam

    def rand_bbox(
        self, img_shape: Tuple, lam: float, margin: float = 0.0, count: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Standard CutMix bounding-box. Generates a random square bbox based on lambda value. This implementation includes
        support for enforcing a border margin as percent of bbox dimensions.

        :param img_shape: image shape as tuple
        :param lam: cutmix lambda value
        :param margin: percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        :param count: number of bbox to generate
        :return:
        """
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape[-2:]
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
        cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
        cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yh = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xh = np.clip(cx + cut_w // 2, 0, img_w)
        return yl, yh, xl, xh

    def rand_bbox_minmax(
        self, img_shape: Tuple[float, float, float], count: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Alternative min-max cutmix bounding-box. Inspired by Darknet cutmix implementation, generates a random
        rectangular bbox based on min/max percent values applied to each dimension of the input image.

        :param img_shape: image shape as tuple
        :param count: number of bbox to generate
        :return: bounding box positions for the full batch
        """
        assert len(self.ratio_minmax) == 2
        img_h, img_w = img_shape[-2:]
        cut_h = np.random.randint(int(img_h * self.ratio_minmax[0]), int(img_h * self.ratio_minmax[1]), size=count)
        cut_w = np.random.randint(int(img_w * self.ratio_minmax[0]), int(img_w * self.ratio_minmax[1]), size=count)
        yl = np.random.randint(0, img_h - cut_h, size=count)
        xl = np.random.randint(0, img_w - cut_w, size=count)
        yu = yl + cut_h
        xu = xl + cut_w
        return yl, yu, xl, xu


class MixUpCallback(MixingCallback):
    def __init__(self, alpha: float = 0.4, label_smoothing: float = 0.0):
        """
        Callback that performs MixUp augmentation on training batches.

        :param float alpha: controls the mixing factor (between 0 and 1)
        :param float label_smoothing: if greater than 0 activates label smoothing
        """
        super().__init__(alpha=alpha, label_smoothing=label_smoothing)

    def on_train_batch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        batch: Dict[str, Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Will be triggered on each training batch."""
        for task in batch:
            images: torch.Tensor = batch[task][Modality.IMAGE.value]
            targets: torch.Tensor = batch[task][self.target_modalities[task]]
            images_flipped = images.flip(0).clone()
            lambdas = np.random.beta(self.alpha, self.alpha, images.size(0))
            # Create the tensor and expand (for batch inputs)
            lambdas_tensor = images.new(lambdas).view(-1, *[1 for _ in range(len(images.size()) - 1)])
            # Combine input batch
            new_images = images * lambdas_tensor + images_flipped * (1 - lambdas_tensor)
            new_targets = self.mixup_targets(targets, lambdas, task=task)
            batch[task][Modality.IMAGE.value] = new_images
            batch[task][self.target_modalities[task]] = new_targets
