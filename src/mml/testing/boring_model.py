from typing import Any, Dict

import torch
import torch.nn

from mml.core.data_loading.task_attributes import TaskType
from mml.core.models.torch_base import BaseHead, BaseModel


class BoringModel(BaseModel):
    """
    A simple model that just processes a single pixel per image.
    """

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # create indexing tuple: ([0, 1, ..., B-1], 0, 0, ...)
        indices = [torch.arange(batch_size)]
        indices.extend([0] * (x.ndim - 1))
        # extract values and reshape to (B, 1)
        return self.backbone(x[tuple(indices)].unsqueeze(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.forward_features(x)
        return {name: head(features) for name, head in self.heads.items()}

    def _init_model(self, **kwargs) -> None:
        self.backbone = torch.nn.Linear(in_features=1, out_features=1)

    def _create_head(self, task_type: TaskType, num_classes: int, **kwargs: Any) -> BaseHead:
        return BoringHead(task_type=task_type, num_classes=num_classes, drop_rate=0.1)

    def supports(self, task_type: TaskType) -> bool:
        """BoringModel supports classification and regression tasks."""
        return task_type in [TaskType.CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION, TaskType.REGRESSION]


class BoringHead(BaseHead):
    def __init__(self, task_type: TaskType, num_classes: int, drop_rate: float):
        super().__init__(task_type=task_type, num_classes=num_classes)
        self.drop = torch.nn.Dropout(drop_rate)
        # only a single head for regression tasks
        n_heads = 1 if task_type == TaskType.REGRESSION else num_classes
        self.linear = torch.nn.Linear(1, n_heads, bias=True)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        return self.linear(x)
