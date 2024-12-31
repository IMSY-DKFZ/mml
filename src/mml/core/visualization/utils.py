# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import torch

from mml.core.data_loading.task_attributes import RGBInfo

# list of easily distinguishable colors
COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#ffe119",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffffff",
    "#000000",
]


def undo_image_normalization(images: torch.Tensor, means: RGBInfo, stds: RGBInfo) -> torch.Tensor:
    """
    Undoes the augmentation step of image normalization. This allows to visualize already augmented images.

    :param torch.Tensor images: batch of images
    :param ~mml.core.data_loading.task_attributes.RGBInfo means: channel means used to normalize
    :param ~mml.core.data_loading.task_attributes.RGBInfo stds: channel stds used to normalize
    :return: batch of images, interpretable for matplotlib, BUT: not clipped nor stretched to [0,255] range
    :rtype: torch.Tensor
    """
    raw_images = images.clone()
    mean = torch.tensor(means.get_rgb()).view(3, 1, 1).to(raw_images.device)
    std = torch.tensor(stds.get_rgb()).view(3, 1, 1).to(raw_images.device)
    raw_images.mul_(std).add_(mean)
    return raw_images
