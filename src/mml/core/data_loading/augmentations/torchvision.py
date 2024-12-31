# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf
from torchvision import tv_tensors
from torchvision.transforms import v2

from mml.core.data_loading.augmentations.augmentation_module import AugmentationModule, DataFormat
from mml.core.data_loading.task_attributes import Modality, RGBInfo


class TorchvisionAugmentationModule(AugmentationModule):
    """
    Torchvision V2 augmentation module.
    """

    def __init__(
        self,
        device: str,
        cfg: ListConfig,
        is_first: bool,
        is_last: bool,
        means: Optional[RGBInfo],
        stds: Optional[RGBInfo],
        num_classes: Optional[int] = None,
    ):
        self.num_classes = num_classes
        super().__init__(device=device, cfg=cfg, is_first=is_first, is_last=is_last, means=means, stds=stds)
        # prevent collating and precision conversions to change type
        tv_tensors.set_return_type("TVTensor")

    def _build_pipeline(self):
        t_list = []
        if self.device == "cpu":
            # first format transforms on cpu
            t_list.extend([v2.ToDtype(torch.uint8, scale=True)])
        t_list.extend(self.from_cfg(self.cfg, on_cpu=self.device == "cpu", num_classes=self.num_classes))
        if self.means is None and self.stds is None:
            if self.is_last or self.device == "cpu":  # need to transfer floats
                t_list.append(v2.ToDtype(torch.float32, scale=True))
        elif sum([x is None for x in [self.means, self.stds]]) == 1:
            raise RuntimeError(
                "Was presented either only STD or only MEAN normalization values. " "Require either none or both!"
            )
        else:
            # Normalize expects float input
            t_list.append(v2.ToDtype(torch.float32, scale=True))
            t_list.append(v2.Normalize(mean=self.means.get_rgb(), std=self.stds.get_rgb()))
        if self.is_last:
            t_list.append(v2.ToPureTensor())
        aug = v2.Compose(t_list)
        # see if necessary, should be deactivated by default
        # deactivate gradients of augmentations forward
        # aug.forward = torch.no_grad()(aug.forward)
        self.pipeline = aug

    def _forward_impl(self, inpt: Any) -> Any:
        if self.data_format == DataFormat.BATCHED_SAMPLE_DICTS or self.data_format == DataFormat.SINGLE_SAMPLE_DICT:
            inpt = {"dummy": inpt}
        outpt = self.pipeline(apply_tv_tensor_types(inpt))
        if self.data_format == DataFormat.BATCHED_SAMPLE_DICTS or self.data_format == DataFormat.SINGLE_SAMPLE_DICT:
            outpt = outpt["dummy"]
        return outpt

    def _sanity_check(self, inpt: Any) -> None:
        pass

    @staticmethod
    def from_cfg(aug_config: ListConfig, on_cpu: bool, num_classes: Optional[int] = None) -> List[v2.Transform]:
        """
        Takes a config and returns a list of corresponding transforms.

        :param DictConfig aug_config: see configs/augmentations/torchvision.yaml for an example, both the "cpu" and the
            "gpu" attribute may be passed to this function.
        :param bool on_cpu: determines if the transforms will be performed on cpu (single sample in a worker) or gpu
            (batched)
        :param Optional[int] num_classes: optional parameter required for MixUp and CutMix transforms
        :return: a list of torchvision v2 transforms
        """
        aug_config = OmegaConf.to_container(aug_config, resolve=True)
        transforms = []
        for transform_args in aug_config:
            transform_name = transform_args.pop("name")
            if transform_name in ["CutMix", "MixUp"]:
                if on_cpu:
                    raise ValueError("CutMix and MixUp transforms need to be performed in batched mode on gpu!")
                if not num_classes:
                    raise ValueError("CutMix and MixUp transforms require the num_classes parameter to be given.")
                transform_args["num_classes"] = num_classes
                transform_args["labels_getter"] = mixup_cutmix_labels_getter
            _dict = {"_target_": "torchvision.transforms.v2." + transform_name}
            _dict.update(**transform_args)
            transform = instantiate(_dict)
            transforms.append(transform)
        return transforms


def mixup_cutmix_labels_getter(batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Helper function to extract labels from a batch for torchvision v2 MixUp and CutMix transforms.
    See https://pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html#non-standard-input-format.

    :param Dict[str, Dict[str, torch.Tensor]] batch: full batch as returned by dataloader, expects a single task
    :return: classification labels of the single task
    """
    task_batch = next(iter(batch.values()))
    for modality_cand in [Modality.CLASS, Modality.CLASSES, Modality.SOFT_CLASSES]:
        if modality_cand.value in task_batch:
            return task_batch[modality_cand.value]
    raise RuntimeError(
        f"Mixup_cutmix_labels_getter did not find a suitable target in batch with keys:" f" {task_batch.keys()}."
    )


def apply_tv_tensor_types(inpt: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Turns plain tensors organised in Modality Dict structure to corresponding TV-Tensors.

    See https://pytorch.org/vision/stable/tv_tensors.html

    :param Dict[str, Dict[str, Any]] inpt: input, must be in DataFormat.MULTI_TASK_SAMPLE_DICTS
    :return: same batch, but all tensors are wrapped by corresponding tv_tensors
    """
    for task in inpt:
        if Modality.IMAGE.value in inpt[task]:
            inpt[task][Modality.IMAGE.value] = tv_tensors.Image(inpt[task][Modality.IMAGE.value], requires_grad=False)
        if Modality.MASK.value in inpt[task]:
            inpt[task][Modality.MASK.value] = tv_tensors.Mask(inpt[task][Modality.MASK.value], requires_grad=False)
    return inpt
