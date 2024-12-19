# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def render_confusion_matrix(cm: npt.NDArray[np.float64], classes: List[str]) -> plt.Figure:
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    :param ~np.ndarray cm: the n by n confusion matrix with integer entries
    :param List[str] classes: list of class names (same order as in the axes of the cm)
    :return: matplotlib figure rendered the cm
    """
    # plot background on true counts (interpretable from color bar)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    # name rows and columns
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # normalize each row (true class) and print relative values into cm cells
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="red")
    # name axes
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig
