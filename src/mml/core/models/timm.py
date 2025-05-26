# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from typing import Any, Dict, Optional

import timm
import timm.data
import torch
import torch.nn as nn
from huggingface_hub.utils import HfHubHTTPError

from mml.core.data_loading.task_attributes import RGBInfo, TaskType
from mml.core.models.torch_base import BaseHead, BaseModel


class TimmGenericModel(BaseModel):
    def __init__(self, **kwargs):
        self.out_channels: Optional[int] = None  # number of backbone output features, set during _init_model
        self.name: str = kwargs["name"]  # backbone name
        self.drop_rate: float = kwargs["drop_rate"]  # heads dropout rate
        super().__init__(**kwargs)  # init requires all kwargs to be stored

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)  # type: ignore[union-attr]
        return {name: head(features) for name, head in self.heads.items()}

    def _init_model(self, name: str, pretrained: bool, drop_rate: float) -> None:
        try:
            self.backbone = timm.create_model(
                model_name=name,
                pretrained=pretrained,
                num_classes=0,  # create heads individually
                in_chans=3,  # maybe support other channel nums in future
            )
        except HfHubHTTPError:
            raise RuntimeError(
                "Huggingface hub appears to be down, you may check: https://status.huggingface.co/ "
                "to re-assure. If the specified backbone has been loaded before you may prepend "
                "HF_HUB_OFFLINE=1 to your mml call (or to your environment variables via "
                "export HF_HUB_OFFLINE=1) and try to rerun."
            )
        if pretrained:
            cfg = timm.data.resolve_data_config(model=self.backbone)
            self.required_mean = RGBInfo(*cfg["mean"])
            self.required_std = RGBInfo(*cfg["std"])
            self.input_size = cfg["input_size"]
        else:
            self.input_size = self.backbone.default_cfg["input_size"]
        self.out_channels = self.backbone.num_features

    def _create_head(self, task_type: TaskType, num_classes: int, **kwargs: Any) -> BaseHead:
        return TimmHead(
            task_type=task_type, num_classes=num_classes, num_features=self.out_channels, drop_rate=self.drop_rate
        )

    def supports(self, task_type: TaskType) -> bool:
        """TimmModel supports classification and regression tasks."""
        return task_type in [TaskType.CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION, TaskType.REGRESSION]


class TimmHead(BaseHead):
    def __init__(self, task_type: TaskType, num_classes: int, num_features: int, drop_rate: float):
        super().__init__(task_type=task_type, num_classes=num_classes)
        self.drop = nn.Dropout(drop_rate)
        # only a single head for regression tasks
        n_heads = 1 if task_type == TaskType.REGRESSION else num_classes
        self.linear = nn.Linear(num_features, n_heads, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        return self.linear(x)
