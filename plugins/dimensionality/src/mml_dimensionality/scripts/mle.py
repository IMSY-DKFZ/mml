# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

"""
The method and partially also the code is reused from:
"The Intrinsic Dimensionaity of Images and Its Impact On Learning"
by Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, Tom Goldstein (ICLR 2021, spotlight)
code: https://github.com/ppope/dimensions
"""

import logging
import random
import warnings
from typing import Tuple

import numpy as np
import torch
from mml_dimensionality.models.knn import KNNComputer
from torch.utils.data import DataLoader
from tqdm import tqdm

from mml.core.data_loading.task_attributes import Modality
from mml.core.data_loading.task_dataset import TaskDataset

logger = logging.getLogger(__name__)


def calc_mle(cfg, dataset: TaskDataset) -> float:
    device = torch.device("cuda") if cfg.allow_gpu else torch.device("cpu")
    subsets = create_random_subsets(data_set=dataset, subset_size=cfg.sampling.sample_num)[: cfg.mode.max_subsets]
    if len(subsets[-1]) < cfg.mode.subset_min_size:
        subsets = subsets[:-1]
        if len(subsets) == 0:
            logger.error(
                f"Not enough samples to build a suitable subset. Current min size is "
                f"{cfg.mode.subset_min_size}, but task only offers {len(dataset)} samples!"
            )
            return float("Nan")
    subset_dims = []
    for s_set in tqdm(subsets, desc="Iterate subsets"):
        if len(s_set) < cfg.sampling.sample_num:
            warnings.warn(f"Subset only has {len(s_set)} samples!")
        model = run_knn(
            dataset=s_set, k=cfg.mode.k, batch_size=cfg.sampling.batch_size, n_workers=cfg.num_workers, device=device
        )
        dist = model.min_dists.cpu().numpy()

        mle_res, inv_mle_res = [], []
        for k in range(3, cfg.mode.k + 1):
            mle_results, invmle_results = intrinsic_dim_sample_wise_double_mle(k, dist)
            mle_res.append(mle_results.mean())
            inv_mle_res.append(1.0 / invmle_results.mean())
            logger.debug(f"Subset estimate for K={k}: MLE -> {mle_res[-1]}, INV-MLE -> {inv_mle_res[-1]}")
        if cfg.mode.inv_mle:
            subset_dims.append(inv_mle_res[-1])
        else:
            subset_dims.append(mle_res[-1])
    subset_dims = [d for d in subset_dims if np.isfinite(d)]
    if len(subset_dims) == 0:
        raise RuntimeError(
            "No valid subset dim found! Consider either increasing  mode.max_subsets, "
            "mode.subset_min_size or switching to mode.inv_mle=true"
        )
    logger.debug(f"all subset estimates: {subset_dims}")
    return sum(subset_dims) / len(subsets)


def run_knn(dataset: torch.utils.data.Subset, k: int, batch_size: int, n_workers: int, device: torch.device):
    model = KNNComputer(sample_num=len(dataset), k=k + 1)
    anchor_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    new_image_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    anchor_counter = 0
    model = model.to(device)
    # ignoring the labels
    with torch.no_grad():
        for a_batch in tqdm(anchor_loader, desc="Compute KNN", leave=False):
            a_images = a_batch[Modality.IMAGE.value].to(device)
            new_img_counter = 0
            for new_batch in new_image_loader:
                new_images = new_batch[Modality.IMAGE.value].to(device)
                # forward through model
                model(a_images, anchor_counter, new_images, new_img_counter)
                new_img_counter += new_images.size(0)

                # equiv_flag = (model.min_dists[anchor_start_idx:anchor_start_idx + a_images.size(0),
                #               0] == 0) & (model.min_dists[
                #                           anchor_start_idx:anchor_start_idx + a_images.size(0), 1] == 0)
                # if torch.any(equiv_flag):
                #     raise Exception("Identical data detected!")

            anchor_counter += a_images.size(0)
    return model.cpu()


def create_random_subsets(data_set, subset_size):
    indices = [i for i in range(len(data_set))]
    random.shuffle(indices)
    n_subsets = (len(data_set) // subset_size) + 1
    subset_idxes = [indices[i * subset_size : (i + 1) * subset_size] for i in range(n_subsets)]

    return [torch.utils.data.Subset(data_set, sidxes) for sidxes in subset_idxes]


def intrinsic_dim_sample_wise_double_mle(k: int, dist: np.ndarray) -> Tuple:
    """
    Returns Levina-Bickel dimensionality estimation and the correction by MacKay-Ghahramani.

    :param int k: nearest neighbors to use
    :param np.ndarray dist: array of the nearest distances
    :return: 2 dimensionality estimates
    """
    dist = dist[:, 1 : (k + 1)]
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1 : k] / dist[:, 0 : k - 1])
    d = d.sum(axis=1) / (k - 2)
    inv_mle = d.copy()
    d = 1.0 / d
    mle = d
    return mle, inv_mle
