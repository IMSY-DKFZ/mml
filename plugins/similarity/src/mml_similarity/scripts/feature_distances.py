# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging

import numpy as np
import scipy.linalg
import torch

from mml.core.data_loading.task_attributes import Modality

try:
    from geomloss import SamplesLoss

    _GEOMLOSS_AVAILABLE = True
except ImportError:
    SamplesLoss = None
    _GEOMLOSS_AVAILABLE = False
from omegaconf import DictConfig
from scipy.stats import entropy, wasserstein_distance
from tqdm import tqdm

from mml.core.data_loading.lightning_datamodule import MultiTaskDataModule
from mml.core.models.lightning_single_frame import SingleFrameLightningModule

logger = logging.getLogger(__name__)


def sample_features(
    model: SingleFrameLightningModule, datamodule: MultiTaskDataModule, device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """
    Uses the model as feature extractor and returns sampled features  of the datamodule in a numpy array.

    :param SingleFrameLightningModule model: model to extract features from raw samples
    :param MultiTaskDataModule datamodule: datamodule to sample images from
    :param torch.device device: device to perform the extraction (cpu recommended)
    :return: numpy array of features, should be 2-dimensional with shape [sample_num, num_features]
    """
    if len(model.task_structs) != 1:
        raise RuntimeError("For backward compatibility only single headed models are supported so far.")
    task = list(model.task_structs.keys())[0]
    model = model.to(device).eval()
    features_list = []
    for batch, batch_idx, dataloader_idx in tqdm(iter(datamodule.train_dataloader())):
        batch_images = batch[task][Modality.IMAGE.value].to(device)
        features = model.forward_features(batch_images).detach().cpu().numpy()
        # add features to list
        features_list.append(features)
    return np.concatenate(features_list, axis=0)


def compute_feature_distance(
    features_1: np.ndarray, features_2: np.ndarray, distance_cfg: DictConfig, mode_id: str
) -> float:
    """
    Wrapper function for all feature based computations.

    :param features_1: features of task 1 [sample_num, num_features]
    :param features_2: features of task 2 [sample_num, num_features]
    :param distance_cfg: distance config, determining computation parameters
    :return: calculated distance
    """
    fct_map = {
        "mmd": calculate_maximum_mean_discrepancy,
        "emd": calculate_wasserstein_distance,
        "fid": calculate_frechet_inception_distance,
        "kld": calculate_kullbach_leibler_divergence,
    }
    if mode_id not in fct_map:
        raise ValueError(f"Mode {mode_id} is not a valid feature distance option!")
    fct = fct_map[mode_id]
    return fct(features_1, features_2, distance_cfg)


def calculate_maximum_mean_discrepancy(
    features_1: np.ndarray, features_2: np.ndarray, distance_cfg: DictConfig
) -> float:
    """
    Returns the Maximum Mean Discrepancy of the two data-samples. Splits the computation in chunks if sample num is
    too large, afterward averages results.
    """
    if features_1.shape[0] > distance_cfg.chunk_size:
        logger.warning(f"High sample size {features_1.shape[0]} causes mmd calculation to be done chunk-wise.")
    splits = (features_1.shape[0] // distance_cfg.chunk_size) + 1
    results = []
    device = torch.device("cuda") if distance_cfg.allow_gpu else torch.device("cpu")
    for sub_source, sub_target in zip(np.array_split(features_1, splits), np.array_split(features_2, splits)):
        source = torch.as_tensor(sub_source).to(device)
        target = torch.as_tensor(sub_target).to(device)
        if distance_cfg.kernel == "cauchy":
            results.append(torch.mean(mmd_cauchy(source, target, a=distance_cfg.blur).view(-1)).item())
        elif distance_cfg.kernel in ["energy", "gaussian", "laplacian"]:
            if not _GEOMLOSS_AVAILABLE:
                raise ImportError("Did not find geomloss package. Please install >pip install geomloss<.")
            mmd = SamplesLoss(loss=distance_cfg.kernel, blur=distance_cfg.blur, backend="auto")
            results.append(mmd(source, target).item())
        else:
            raise ValueError(f"Kernel {distance_cfg.kernel} not implemented for MMD computation.")
    return torch.tensor(results).mean().item()


def mmd_cauchy(x: torch.Tensor, y: torch.Tensor, a: float) -> torch.Tensor:
    """
    Maximum Mean Discrepancy Cauchy Kernel computation. Code by Tim Adler (& IWR).

    :param x: (samples x features) tensor
    :param y: (samples x features) tensor
    :param a: scalar parameter
    :return: cauchy kernel maximum mean discrepancy
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    XX = a * (a + (rx.t() + rx - 2.0 * xx)) ** -1
    YY = a * (a + (ry.t() + ry - 2.0 * yy)) ** -1
    XY = a * (a + (rx.t() + ry - 2.0 * zz)) ** -1
    return XX + YY - 2.0 * XY


def calculate_frechet_inception_distance(
    features_1: np.ndarray, features_2: np.ndarray, distance_cfg: DictConfig
) -> float:
    """
    Returns the Frechet Inception Distance of two data-samples. Code adapted from torch-fidelity.
    See https://github.com/toshas/torch-fidelity/blob/1e4eaa478fd42aeb31f8476ef7d5181ead9ead37/torch_fidelity/metric_fid.py
    """
    assert len(features_1.shape) == 2 and len(features_2.shape) == 2
    mu1 = np.mean(features_1, axis=0)
    mu2 = np.mean(features_2, axis=0)
    sigma1 = np.cov(features_1, rowvar=False)
    sigma2 = np.cov(features_2, rowvar=False)
    assert mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        logger.warning(
            f"fid calculation produces singular product; adding {distance_cfg.eps} to diagonal of cov estimates"
        )
        offset = np.eye(sigma1.shape[0]) * distance_cfg.eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            msg = f"Imaginary component {m}."
            logger.error(msg)
            raise ValueError(msg)
        covmean = covmean.real
    # compute fid and return
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def calculate_wasserstein_distance(features_1: np.ndarray, features_2: np.ndarray, distance_cfg: DictConfig) -> float:
    """
    Return wasserstein distance of two feature matrices.
    """
    assert (
        len(features_1.shape) == len(features_2.shape) == 2
    ), "wasserstein distance computation expects (samples x features) array"
    assert (
        features_1.shape[1] == features_2.shape[1]
    ), "source and target data are expected to have identical feature shapes"
    if distance_cfg.method == "sinkhorn":
        if not _GEOMLOSS_AVAILABLE:
            raise ImportError("Did not find geomloss package. Please install >pip install geomloss<.")
        sinkhorn = SamplesLoss(
            "sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized"
        )  # todo optimize parameters
        return sinkhorn(torch.as_tensor(features_1), torch.as_tensor(features_2))
    elif distance_cfg.method == "default":
        total = 0
        for dim in range(features_1.shape[1]):
            total += wasserstein_distance(features_1[:, dim], features_2[:, dim])
        return total / features_1.shape[1]
    elif distance_cfg.method == "binned":
        n_bins = 1000
        data_1 = torch.from_numpy(features_1)
        data_2 = torch.from_numpy(features_2)
        # create bins
        hist1 = torch.zeros((data_1.shape[1], n_bins), dtype=torch.long)
        hist2 = torch.zeros((data_2.shape[1], n_bins), dtype=torch.long)
        for c in range(data_1.shape[1]):
            hist1[c] = torch.histc(data_1[:, c].flatten(), bins=n_bins)
            hist2[c] = torch.histc(data_2[:, c].flatten(), bins=n_bins)
        # normalize
        hist1 = hist1 / torch.sum(hist1, dim=1, dtype=torch.double, keepdim=True)
        hist2 = hist2 / torch.sum(hist2, dim=1, dtype=torch.double, keepdim=True)
        # see https://en.wikipedia.org/wiki/Earth_mover%27s_distance#Computing_the_EMD
        diff = hist1 - hist2
        y = torch.cumsum(diff, dim=1)
        return torch.mean(torch.sum(torch.abs(y), dim=1)).item()
    else:
        raise ValueError("distance.method must be either 'sinkhorn' or 'default' or 'binned'")


def calculate_kullbach_leibler_divergence(
    features_1: np.ndarray, features_2: np.ndarray, distance_cfg: DictConfig
) -> float:
    """
    Return kullbach leibler divergence of the feature means of two sample feature matrices.
    See https://arxiv.org/pdf/1908.07630.pdf for details.
    """
    assert (
        len(features_1.shape) == len(features_2.shape) == 2
    ), "kullbach leibler distance computation expects (samples x features) array"
    assert (
        features_1.shape[1] == features_2.shape[1]
    ), "source and target data are expected to have identical feature shapes"
    # average features
    source_avg = np.mean(features_1, axis=0)
    target_avg = np.mean(features_2, axis=0)
    return entropy(source_avg, target_avg)
