# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging

import torch
import torch.nn.functional as Functional
from scipy.spatial.distance import jensenshannon
from torch import Tensor

logger = logging.getLogger(__name__)


def compute_task_distance(emb_1: Tensor, emb_2: Tensor, metric: str = "cosine") -> float:
    if metric == "cosine":
        return cosine_distance(emb_1, emb_2)
    elif metric == "euclidean":
        return euclidean_distance(emb_1, emb_2, normalize=False)
    elif metric == "normalized_euclidean":
        return euclidean_distance(emb_1, emb_2, normalize=True)
    elif metric == "jensenshannon":
        return jensenshannon(emb_1.cpu().numpy(), emb_2.cpu().numpy())
    else:
        logger.error(f"Invalid distance metric {metric} for distance computation.")
        return float("NaN")


def cosine_distance(emb_1: Tensor, emb_2: Tensor) -> float:
    return 1.0 - Functional.cosine_similarity(emb_1, emb_2, dim=0).item()


def euclidean_distance(emb_1: Tensor, emb_2: Tensor, normalize: bool = True) -> float:
    if normalize:
        emb_1 = Functional.normalize(emb_1, dim=0)
        emb_2 = Functional.normalize(emb_2, dim=0)
    return torch.linalg.vector_norm(emb_1 - emb_2, ord=2).item()
