# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import torch
from torch.nn.functional import one_hot


def one_hot_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Expects a batched segmentation mask (B x H x W) and returns a batched one-hot encoding like (B x C x H x W). Handles
    pseudo entry 255 within.

    :param torch.Tensor mask: the mask to be transformed (B x H x W), values must be below num_classes
    :param int num_classes: the number of classes
    :raises ValError: if mask values are outside [0, ..., num_classes - 1]
    :return: a batched one-hot encoding like (B x C x H x W)
    :rtype: torch.Tensor
    """
    if mask.max() >= num_classes or mask.min() < 0:
        raise ValueError("Mask values must be within 0 and num_classes - 1.")
    # replace 255 value with pseudo-class
    target_remapped = torch.where(mask == 255, num_classes, mask)
    # one-hot encode with pseudo-class
    target_one_hot = one_hot(target_remapped, num_classes=num_classes + 1).permute(0, 3, 1, 2)
    # remove pseudo class
    target_one_hot = target_one_hot[:, :-1]
    return target_one_hot
