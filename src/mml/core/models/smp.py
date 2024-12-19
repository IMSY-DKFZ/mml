# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from typing import Any, Dict, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.base import ClassificationHead, SegmentationHead, initialization

from mml.core.data_loading.task_attributes import RGBInfo, TaskType
from mml.core.models.torch_base import BaseHead, BaseModel

logger = logging.getLogger(__name__)


class SMPGenericModel(BaseModel):
    def __init__(self, **kwargs):
        self.arch_name: Optional[str] = None  # architecture, set during _init_model
        self.encoder_name: Optional[str] = None  # encoder, set during _init_model
        self.weights: Optional[str] = None  # encoder pretraining weights, set during _init_model
        self.feature_channels: Optional[int] = None  # encoder output size, set during _init_model
        self.out_channels: Optional[int] = None  # decoder output size, set during _init_model
        super(SMPGenericModel, self).__init__(**kwargs)
        # only used for feature extraction
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone.encoder(x)
        decoder_output = self.backbone.decoder(*features)
        return {
            name: head(decoder_output if head.task_type == TaskType.SEMANTIC_SEGMENTATION else features[-1])
            for name, head in self.heads.items()
        }

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooling(self.backbone.encoder(x)[-1]).squeeze(3).squeeze(2)

    def _init_model(self, arch: str, weights: Optional[str], encoder: str = "resnet34", **kwargs: Any) -> None:
        model = smp.create_model(
            arch=arch,
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=1,  # default segmentation head will be discarded
        )
        settings = smp.encoders.encoders[encoder]["pretrained_settings"][weights if weights else "imagenet"]
        self.input_size = settings.get("input_size")
        if weights:
            self.required_mean = RGBInfo(*settings.get("mean"))
            self.required_std = RGBInfo(*settings.get("std"))
        self.arch_name = arch
        self.encoder_name = encoder
        self.weights = weights
        self.backbone = model
        self.feature_channels = self.backbone.encoder.out_channels[-1]
        self.out_channels = self.backbone.segmentation_head[1].in_channels

    def _create_head(self, task_type: TaskType, num_classes: int, **kwargs: Any) -> BaseHead:
        return SMPHead(
            task_type=task_type,
            num_classes=num_classes,
            num_features=self.out_channels if task_type == TaskType.SEMANTIC_SEGMENTATION else self.feature_channels,
        )

    def supports(self, task_type: TaskType) -> bool:
        """SMP support classification and segmentation tasks."""
        return task_type in [
            TaskType.CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.SEMANTIC_SEGMENTATION,
        ]


class SMPHead(BaseHead):
    def __init__(self, task_type: TaskType, num_classes: int, num_features: int):
        super().__init__(task_type=task_type, num_classes=num_classes)
        if task_type == TaskType.SEMANTIC_SEGMENTATION:
            self.head = SegmentationHead(in_channels=num_features, out_channels=num_classes, activation="softmax2d")
        else:
            self.head = ClassificationHead(in_channels=num_features, classes=num_classes)
        initialization.initialize_head(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
