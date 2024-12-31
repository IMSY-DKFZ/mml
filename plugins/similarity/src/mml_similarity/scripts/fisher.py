# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
from hydra.core.config_store import ConfigStore
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM_MonteCarlo
from nngeometry.object import PMatDiag
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from mml.core.data_loading.task_attributes import Modality, TaskType
from mml.core.data_loading.task_dataset import TaskDataset, TupelizedTaskDataset
from mml.core.data_preparation.utils import WIPBar
from mml.core.models.torch_base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class FimConfig:
    samples: int = 10000
    empirical: bool = False
    ignore_bias: bool = True
    ignore_downsample: bool = True
    ignore_bn: bool = False
    average_filters: bool = True
    final_fraction: float = 0.6
    nngeom: bool = False


cs = ConfigStore.instance()
cs.store(name="fim_config", node=FimConfig)


def compute_fisher_information_embedding(
    fim_cfg: FimConfig,
    model: BaseModel,
    ds: TaskDataset,
    loader_kwargs: dict,
    device: torch.device,
    task_type: TaskType,
) -> Dict[str, Tensor]:
    """
    Wrapper for computing the fisher information embedding of a task.

    :param fim_cfg: subnode of config to deal with settings on fim computation (should reside at cfg.mode.fim)
    :param model: tuned model on the task (torch.nn.Module), needs to be a subclass of mml.core.models.torch_modules.base.BaseModel
    :param loader_kwargs: kwargs for the data loader, e.g. specifying num_workers
    :param ds: the dataset in use
    :param device: torch device to compute on
    :param task_type: TaskType of the task
    :return: diagonal of the Fisher information matrix as dict
    """
    assert task_type in [
        TaskType.CLASSIFICATION,
        TaskType.SEMANTIC_SEGMENTATION,
    ], f"Task_type {task_type} not supported during fim computation yet!"

    loader_kwargs["sampler"] = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=fim_cfg.samples)
    loader_kwargs["drop_last"] = False  # make sure to use all samples
    if fim_cfg.nngeom:
        ds = TupelizedTaskDataset(ds)
    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)
    param_names = select_embedding_params(fim_cfg, model)
    func = nngeom_fim if fim_cfg.nngeom else fim_diag
    embedding = func(
        model,
        data_loader=loader,
        fim_params=param_names,
        empirical=fim_cfg.empirical,
        device=device,
        task_type=task_type,
    )
    return postprocess_embedding(fim_cfg, embedding)


def select_embedding_params(fim_cfg: FimConfig, model: BaseModel) -> List[str]:
    """
    Chooses the parameter names to be considered as elements of the embedding.

    :param fim_cfg: subnode of config to deal with settings on fim computation (should reside at cfg.mode.fim)
    :param model: tuned model on the task (torch.nn.Module), needs to be a subclass of mml.core.models.torch_modules.base.BaseModel
    :return: list of str with names of the relevant parameters
    """
    # first shrink to encoder params
    encoder_pars = [
        name for name, param in model.named_parameters() if name.startswith("backbone.") and param.requires_grad
    ]
    # now, based on config drop some
    to_drop = []
    # excluding bias terms
    if fim_cfg.ignore_bias:
        to_drop += [name for name in encoder_pars if "bias" in name]
    # ignore downsample weights
    if fim_cfg.ignore_downsample:
        to_drop += [name for name in encoder_pars if "downsample" in name]
    # ignore batchnorm weights
    if fim_cfg.ignore_bn:
        to_drop += [name for name in encoder_pars if ".bn." in name]
    # only take into consideration a fraction based on the final model layers
    if fim_cfg.final_fraction < 1:
        # TODO: beware naming compatibility with other models
        layers = [name for name, _ in model.backbone.named_children() if "fc" not in name and "pool" not in name]
        layers_to_keep = layers[int(len(layers) * (1 - fim_cfg.final_fraction)) :]
        logger.debug(
            f"Fim final fraction results in keeping the following encoder layers: {layers_to_keep} and "
            f"dropping these encoder layers: {[name for name in layers if name not in layers_to_keep]}"
        )
        to_drop += [name for name in encoder_pars if name.split(".")[1] not in layers_to_keep]
    # make sure to not drop multiple times by making a set
    for name in set(to_drop):
        encoder_pars.remove(name)
    return encoder_pars


def embedding_to_tensor(embedding: Dict[str, Tensor]) -> Tensor:
    """
    Transforms an embedding (dict of tensors) to a single tensor.

    :param embedding: any dict with tensor values
    :return: a single tensor
    """
    return torch.cat(tuple([t.view(-1) for t in embedding.values()]))


def postprocess_embedding(fim_cfg: FimConfig, embedding: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Averaging convolutional layers if specified. Also sets precision and checks for NaN values.

    :param fim_cfg: subnode of config to deal with settings on fim computation (should reside at cfg.mode.fim)
    :param embedding: embedding as produced by fim_diag function
    :return: postprocessed embedding
    """
    # if specified average filters within a convolution layer
    if fim_cfg.average_filters:
        # TODO: better alternative is to use actual layer class (gathered earlier)
        conv_params = [
            name for name in embedding.keys() if "conv" in name and "weight" in name and embedding[name].dim() == 4
        ]
        for conv_param in conv_params:
            embedding[conv_param] = torch.mean(embedding[conv_param], dim=[2, 3])  # Average over kernel size
    # finally setting precision and checking for missing values
    for param in embedding.keys():
        embedding[param].to(torch.double)
        if torch.max(torch.isnan(embedding[param])) == 1:
            logger.error("Detected NaN value(s) in Fisher Embedding!")
    return embedding


def nngeom_fim(
    model: Module,
    data_loader: DataLoader,
    fim_params: List[str],
    task_type: TaskType,
    empirical: bool = False,
    device: torch.device = None,
) -> Dict[str, Tensor]:
    if empirical:
        raise NotImplementedError
    model.eval()
    model.to(device)
    with WIPBar() as bar:
        bar.desc = "Computing FIM"
        fim_layers = list(set(".".join(name.split(".")[:-1]) for name in fim_params))
        layer_col = LayerCollection()
        for layer, mod in model.named_modules():
            if len(list(mod.children())) == 0 and len(list(mod.parameters())) > 0:
                if layer in fim_layers:
                    layer_col.add_layer_from_model(model, mod)
        variant = None
        if task_type == TaskType.CLASSIFICATION:
            variant = "classif_logits"
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            variant = "segmentation_logits"

        def loading_func(*d):
            return next(iter(model(d[0].to(device)).values()))

        diag_vals = FIM_MonteCarlo(
            model=model,
            loader=data_loader,
            representation=PMatDiag,
            device=device,
            variant=variant,
            function=loading_func,
            layer_collection=layer_col,
        ).get_diag()
        fim = {name: None for name in fim_params}
        _, module_to_layerid = layer_col.get_layerid_module_maps(model)
        for layer, mod in model.named_modules():
            if layer not in fim_layers:
                continue
            collection_layer_id = module_to_layerid[mod]
            idx = layer_col.p_pos[collection_layer_id]
            size = layer_col.layers[collection_layer_id].numel()
            vals = diag_vals[idx : idx + size]
            param_counts = {layer + "." + name: param.numel() for name, param in mod.named_parameters()}
            param_links = {layer + "." + name: param for name, param in mod.named_parameters()}
            assert size == sum(param_counts.values())
            delimiter = 0
            for param_name, count in param_counts.items():
                if param_name in fim_params:
                    fim[param_name] = (
                        vals[delimiter : delimiter + count].clone().detach().view_as(param_links[param_name])
                    )
                delimiter += count
    assert None not in fim.values(), "something went wrong..."
    return fim


def fim_diag(
    model: Module,
    data_loader: DataLoader,
    fim_params: List[str],
    task_type: TaskType,
    empirical: bool = False,
    device: torch.device = None,
) -> Dict[str, Tensor]:
    """
    Computes the diagonal elements of the fisher information matrix for a given task.

    :param model: tuned model for the task
    :param data_loader: data loader for the task
    :param fim_params: list of parameter names that should be considered
    :param task_type: TaskType of the task
    :param empirical: if activated uses the empirical fisher (uses true targets instead of sampled model estimations)
    :param device: torch device to run computations on
    :return: diagonal elements of the fisher as parameter name to value mapping
    """
    model.eval()
    model.to(device)
    fim = {}
    for name, param in model.named_parameters():
        if name in fim_params:
            fim[name] = torch.zeros_like(param)
        else:
            param.requires_grad = False
    # logging infos
    tic = time.time()
    seen, last = 0, 0
    # iterate once over loader
    with tqdm(total=len(data_loader.sampler), desc="compute fisher") as pbar:
        for batch in data_loader:
            # extract batch
            images = batch[Modality.IMAGE.value]
            if empirical:
                targets = (
                    batch[Modality.CLASS.value] if task_type == TaskType.CLASSIFICATION else batch[Modality.MASK.value]
                )
            # send to device
            if device is not None:
                images = images.to(device)
                if empirical:
                    targets = targets.to(device)
            # forward image
            logits = next(iter(model(images).values()))
            if empirical:
                outdx = targets.unsqueeze(1)
            elif task_type == TaskType.CLASSIFICATION:
                outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
            else:
                outdx = None
                raise NotImplementedError("SEMANTIC_SEGMENTATION FIM not yet implemented!")
            samples = logits.gather(1, outdx)

            # iterate once over batch
            for idx in range(images.size(0)):
                model.zero_grad()
                torch.autograd.backward(samples[idx], retain_graph=True)
                for name, param in model.named_parameters():
                    if name in fim_params:
                        fim[name] += param.grad**2
                        fim[name].detach_()
                seen += 1
                pbar.update(1)
                if seen % 100 == 0:
                    toc = time.time()
                    fps = float(seen - last) / (toc - tic)
                    tic, last = toc, seen
                    logger.debug(f"Samples: {seen:5d} Fps: {fps:2.4f} samples/s.")

    # average fim_diag elements
    for name, grad2 in fim.items():
        grad2 /= float(seen)
    return fim
