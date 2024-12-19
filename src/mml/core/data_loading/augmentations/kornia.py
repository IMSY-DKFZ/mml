# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from typing import Any, List, Optional

import kornia as K
import torch
from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf

from mml.core.data_loading.augmentations.augmentation_module import AugmentationModule, DataFormat
from mml.core.data_loading.task_attributes import Modality, RGBInfo

KORNIA_VALID_MODALITIES = ["image", "mask", "bbox", "keypoints"]


class KorniaAugmentationModule(AugmentationModule):
    """
    Kornia augmentation module.

    Basic supported dict entries are ['image', 'mask', 'bbox', 'keypoints']
    """

    def __init__(
        self,
        device: str,
        cfg: ListConfig,
        is_first: bool,
        is_last: bool,
        means: Optional[RGBInfo],
        stds: Optional[RGBInfo],
    ):
        self.data_keys = None
        super().__init__(device=device, cfg=cfg, is_first=is_first, is_last=is_last, means=means, stds=stds)
        self.run_inverse = False

    def _build_pipeline(self):
        transforms = self.from_cfg(self.cfg)
        norm_trans = []
        if self.means is None and self.stds is None:
            pass
        elif sum([x is None for x in [self.means, self.stds]]) == 1:
            raise RuntimeError(
                "Was presented either only STD or only MEAN normalization values. " "Require either none or both!"
            )
        else:
            norm_trans.append(K.augmentation.Normalize(mean=self.means.get_rgb(), std=self.stds.get_rgb()))
        aug = K.augmentation.container.AugmentationSequential(*transforms, *norm_trans)
        # deactivate gradients of augmentations forward
        aug.forward = torch.no_grad()(aug.forward)
        self.pipeline = aug

    def _forward_impl(self, inpt: Any) -> Any:
        if self.data_format == DataFormat.BATCHED_SAMPLE_DICTS:
            inpt = {"dummy": inpt}
        for task in inpt:
            sub_batch = inpt[task]
            # use order from dataset
            aug_modalities = [mod for mod in sub_batch if mod in KORNIA_VALID_MODALITIES]
            if Modality.MASK.value in aug_modalities:
                # kornia requires mask to be float and same dimensions as image
                sub_batch[Modality.MASK.value] = sub_batch[Modality.MASK.value].unsqueeze(1).float()
            # disassemble batch
            batch_list = [sub_batch[mod] for mod in aug_modalities]
            # augment
            if self.run_inverse:
                augmented = self.pipeline.inverse(*batch_list, data_keys=aug_modalities)
            else:
                augmented = self.pipeline(*batch_list, data_keys=aug_modalities)
            # re-assemble
            if len(aug_modalities) == 1:
                # kornia treats this case differently and returns tensor instead of list
                sub_batch.update({aug_modalities[0]: augmented})
            else:
                sub_batch.update({mod: augmented[ix] for ix, mod in enumerate(aug_modalities)})
            if Modality.MASK.value in aug_modalities:
                # undo kornia requirement of mask to be float and same dimensions as image
                sub_batch[Modality.MASK.value] = sub_batch[Modality.MASK.value].squeeze(1).long()
        if self.data_format == DataFormat.BATCHED_SAMPLE_DICTS:
            inpt = inpt["dummy"]
        return inpt

    def inverse(self, inpt: Any) -> Any:
        """
        API to use the feature of kornia augmentations to invertible. Reverts the last augmentation. Useful for TTA.

        :param inpt: a transformed batch (potentially after inference by a model)
        :return: the model results with undone geometry
        """
        self.run_inverse = True
        outpt = self.__call__(inpt)
        self.run_inverse = False
        return outpt

    def _sanity_check(self, inpt: Any) -> None:
        assert self.device == "gpu", "so far only gpu is supported for KorniaAugmentationModule"
        assert self.data_format in [DataFormat.MULTI_TASK_SAMPLE_DICTS, DataFormat.BATCHED_SAMPLE_DICTS]
        modalities = []
        if self.data_format == DataFormat.MULTI_TASK_SAMPLE_DICTS:
            for task in inpt:
                modalities.extend(list(inpt[task].keys()))
        else:
            modalities = list(inpt.keys())
        modalities = [
            mod
            for mod in modalities
            if mod
            not in [Modality.CLASS.value, Modality.CLASSES.value, Modality.SOFT_CLASSES.value, Modality.SAMPLE_ID.value]
        ]
        if any([mod not in KORNIA_VALID_MODALITIES for mod in modalities]):
            raise ValueError(f"Some modality might not be supported by kornia backend: {modalities}")
        self.data_keys = modalities

    @staticmethod
    def from_cfg(aug_config: ListConfig) -> List[K.augmentation.AugmentationBase2D]:
        """
        Takes a config and returns a list of corresponding transforms.

        :param DictConfig aug_config: see configs/augmentations/kornia.yaml for an example
        :return: a list of kornia transforms
        """
        aug_config = OmegaConf.to_container(aug_config, resolve=True)
        transforms = []
        for transform_args in aug_config:
            transform_name = transform_args.pop("name")
            if transform_name == "RandAugment":
                transform = K.auto.RandAugment(**transform_args)
            elif transform_name == "AutoAugment":
                transform = K.auto.AutoAugment(**transform_args)
            elif transform_name == "TrivialAugment":
                transform = K.auto.TrivialAugment(**transform_args)
            else:
                _dict = {"_target_": "kornia.augmentation._2d." + transform_name}
                _dict.update(**transform_args)
                transform = instantiate(_dict)
            transforms.append(transform)
        return transforms
