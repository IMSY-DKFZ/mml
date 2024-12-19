# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import torch

from mml.core.data_loading.task_attributes import Modality
from mml.core.scripts.decorators import beta


@beta("Test Time Augmentation and PredictionMerger are in beta.")
class PredictionMerger:
    def __init__(self, mode: str, modality: Modality):
        """Merges the logits of predictions by variations introduced via test time augmentation (TTA)."""
        if mode not in ["mean"]:
            raise ValueError(f"PredictionMerger mode {mode} is not supported")
        self.mode = mode
        self.modality = modality
        self.n = 0
        self.merged = None

    def update(self, prediction: torch.Tensor) -> None:
        """Add a prediction to the merger."""
        self.n += 1
        if self.merged is None:
            self.merged = prediction
        elif self.mode == "mean":
            if self.modality in [Modality.CLASS, Modality.SOFT_CLASSES, Modality.CLASSES, Modality.MASK]:
                self.merged += prediction
            else:
                raise ValueError(f"PredictionMerger mode {self.mode} does not support modality {self.modality}")

    def compute(self) -> torch.Tensor:
        """Let the merger compute the merged result."""
        if self.merged is None:
            raise RuntimeError("PredictionMerger has seen no predictions")
        if self.mode == "mean":
            return self.merged / self.n
        else:
            raise RuntimeError(f"PredictionMerger mode {self.mode} is not supported")
