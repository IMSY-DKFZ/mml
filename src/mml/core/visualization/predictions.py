# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

"""Functions that allow nice plotting of model predictions."""

from itertools import cycle
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
from torchvision.utils import draw_segmentation_masks

from mml.core.data_loading.task_attributes import TaskType
from mml.core.data_loading.utils import one_hot_mask
from mml.core.visualization.utils import COLORS


def render_predictions(
    raw_images: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, classes: List[str], task_type: TaskType
) -> plt.Figure:
    """
    Wrapper function to access task prediction renderers.

    :param torch.Tensor raw_images: non-normalized but potentially augmented (e.g. rotated) images
    :param torch.Tensor logits: prediction logits
    :param torch.Tensor targets: underlying targets
    :param List[str] classes: class strings (order must match target indices)
    :param TaskType task_type: the corresponding tasks task type
    :return: a matplotlib figure that shows some model predictions
    """
    if task_type not in PREDICTION_RENDERERS:
        raise RuntimeError(f"Could not render predictions of task type {task_type}.")
    func = PREDICTION_RENDERERS[task_type]
    return func(raw_images=raw_images, logits=logits, targets=targets, classes=classes)


def render_classification_predictions(
    raw_images: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, classes: List[str]
) -> plt.Figure:
    """
    Implements prediction rendering for classification tasks.
    """
    normalized_preds = torch.nn.functional.softmax(logits, dim=1)
    grid = []
    for img_idx in range(raw_images.size(0)):
        img = raw_images[img_idx]
        row_list = []
        # raw image
        row_list.append(img)
        # reference
        row_list.append(classes[targets[img_idx].cpu().item()])
        # iterate over classes
        for cls_idx in range(len(classes)):
            row_list.append(f"{normalized_preds[img_idx][cls_idx].cpu().item():.2f}")
        grid.append(row_list)
    return render_labeled_grid(img_grid=grid, col_labels=["Image", "Reference"] + classes)


def render_segmentation_predictions(
    raw_images: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, classes: List[str]
) -> plt.Figure:
    """
    Implements prediction rendering for segmentation tasks.
    """
    color_cycle = cycle(COLORS)
    color_list = [next(color_cycle) for _ in classes]
    normalized_masks = torch.nn.functional.softmax(logits, dim=1)
    one_hot_targets = one_hot_mask(mask=targets, num_classes=logits.size(1)).to(dtype=torch.bool, device="cpu")
    grid = []
    for img_idx in range(raw_images.size(0)):
        img = raw_images[img_idx]
        row_list = []
        # raw image
        row_list.append(img)
        # reference
        reference = draw_segmentation_masks(
            image=(img * 255).to(dtype=torch.uint8, device="cpu"),
            masks=one_hot_targets[img_idx],
            alpha=0.8,
            colors=color_list,
        )
        row_list.append(reference)
        # iterate over classes
        for cls_idx in range(len(classes)):
            row_list.append(normalized_masks[img_idx][cls_idx])
        grid.append(row_list)
    return render_labeled_grid(img_grid=grid, col_labels=["Image", "Reference"] + classes)


def render_labeled_grid(
    img_grid: Sequence[Sequence[Union[torch.Tensor, np.ndarray, str]]], col_labels: Optional[List[str]]
) -> plt.Figure:
    """
    Takes a grid of images and strings and returns a matplotlib figure.

    :param Sequence[Sequence[Union[torch.Tensor, np.ndarray, str]]] img_grid: list of lists, containing images as
      numpy arrays or torch tensors and potentially strings
    :param Optional[List[str]] col_labels: (optional) labels for the columns of the grid
    :return: a matplotlib figure with given column titles and rendered images / text
    :rtype: plt.Figure
    """
    if col_labels is None:
        col_labels = [None for _ in img_grid[0]]
    assert all([len(col_labels) == len(img_grid[i]) for i in range(len(img_grid))])
    fig, axs = plt.subplots(ncols=len(img_grid[0]), nrows=len(img_grid), squeeze=False)
    for row_idx, row_imgs in enumerate(img_grid):
        for col_idx, img in enumerate(row_imgs):
            # transform tensors to array
            if isinstance(img, torch.Tensor):
                img = img.detach()
                img = F.to_pil_image(img)
                img = np.asarray(img)
            if isinstance(img, np.ndarray):
                axs[row_idx, col_idx].imshow(np.asarray(img))
            elif isinstance(img, str):
                axs[row_idx, col_idx].text(0.5, 0.5, img, horizontalalignment="center", verticalalignment="center")
            axs[row_idx, col_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if col_labels[col_idx] and row_idx == 0:
                axs[0, col_idx].set_title(
                    label=col_labels[col_idx], rotation=45, fontsize="small", fontstretch="condensed"
                )
    fig.subplots_adjust(hspace=0, wspace=0)
    return fig


# access this dict from outside to provide additional rendering or modify existing ones
PREDICTION_RENDERERS = {
    TaskType.CLASSIFICATION: render_classification_predictions,
    TaskType.MULTILABEL_CLASSIFICATION: render_classification_predictions,
    TaskType.SEMANTIC_SEGMENTATION: render_segmentation_predictions,
}
