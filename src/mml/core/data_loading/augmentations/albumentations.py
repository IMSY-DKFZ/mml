# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import hashlib
from typing import Any, Dict, List, Optional

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from omegaconf import ListConfig, OmegaConf

from mml.core.data_loading.augmentations.augmentation_module import (
    IMAGENET_AA_PATH,
    AugmentationModule,
    DataFormat,
    Transform,
)
from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.data_loading.task_attributes import RGBInfo

ALBUMENTATIONS_VALID_MODALITIES = ["image", "mask"]


class AlbumentationsAugmentationModule(AugmentationModule):
    """
    Albumentations augmentation module.

    Basic supported dict entries are ['image', 'mask', 'bboxes', 'keypoints'].

    An AutoAlbument generated pipeline is available via the Identifier "ImageNetAA" (no parameters). "RandAugment" with
    parameters is also provided on top. See :meth:`get_rand_augment` for details.

    Note that once bboxes and keypoints will be supported than composition will include the respective parameters
    https://albumentations.ai/docs/api_reference/core/composition/. Futhermore "additional_targets" might need to be
    defined. To check whether a certain augmentation supports a specific target type see
    https://albumentations.ai/docs/getting_started/transforms_and_targets/.
    """

    def __init__(
        self,
        device: str,
        cfg: ListConfig,
        is_first: bool,
        is_last: bool,
        means: Optional[RGBInfo],
        stds: Optional[RGBInfo],
        floatify: bool = False,
        tensorize: bool = True,
    ):
        self.tensorize = tensorize
        self.floatify = floatify
        if floatify and not tensorize:
            raise ValueError("floatify is to ensure float tensors are moved to device")
        super().__init__(device=device, cfg=cfg, is_first=is_first, is_last=is_last, means=means, stds=stds)

    def _build_pipeline(self):
        transforms = self.from_cfg(self.cfg)
        norm_trans = []
        _is_float = False
        if self.means is None and self.stds is None:
            if self.is_last:
                # no normalization requested, but this is the last transform, we need to make sure to have float values
                norm_trans.append(A.ToFloat(max_value=255))
                _is_float = True
        elif sum([x is None for x in [self.means, self.stds]]) == 1:
            raise RuntimeError(
                "Was presented either only STD or only MEAN normalization values. Require either none or both!"
            )
        else:
            # default case: requested normalization
            norm_trans.append(A.Normalize(mean=self.means.get_rgb(), std=self.stds.get_rgb()))
            _is_float = True
        # if this is the last transform before moving to gpu we need to make sure to have float tensors
        # to support lightning precision
        if self.floatify and not _is_float:
            norm_trans.append(A.ToFloat(max_value=255))
        if self.tensorize:
            norm_trans.append(ToTensorV2())
        self.pipeline = A.Compose([*transforms, *norm_trans])

    def _forward_impl(self, inpt: Dict[str, Any]) -> Dict[str, Any]:
        # albumentations only handles single sample inputs, receive a dict and return a dict
        clone = {mod: inpt[mod] for mod in ALBUMENTATIONS_VALID_MODALITIES if mod in inpt}
        outpt = self.pipeline(**clone)
        outpt.update({k: v for k, v in inpt.items() if k not in ALBUMENTATIONS_VALID_MODALITIES})
        return outpt

    def _sanity_check(self, inpt: Any) -> None:
        # albumentations sanity checks
        assert self.device == "cpu"
        assert self.data_format == DataFormat.SINGLE_SAMPLE_DICT

    def __hash__(self):
        """MD5 Hash value of the pipeline."""
        path = MMLFileManager.instance().construct_saving_path(None, key="temp", file_name="hash_dump.json")
        A.save(self.pipeline, str(path))
        block_size = 65536
        hasher = hashlib.md5()
        with open(str(path), "rb") as file:
            buf = file.read(block_size)
            while len(buf) > 0:
                hasher.update(buf)
                buf = file.read(block_size)
        return hasher.hexdigest()

    @staticmethod
    def from_cfg(aug_config: ListConfig) -> List[Transform]:
        """
        Takes a config and returns a list of corresponding transforms.

        :param DictConfig aug_config: see configs/augmentations/default.yaml for an example, the "cpu" attribute is what
            has to be passed to this function.
        :return: a list of albumentation transforms
        """
        aug_config = OmegaConf.to_container(aug_config, resolve=True)
        transforms = []
        for transform_args in aug_config:
            transform_name = transform_args.pop("name")
            if transform_name == "RandAugment":
                transform = AlbumentationsAugmentationModule.get_rand_augment(**transform_args)
            elif transform_name == "ImageNetAA":
                transform: Transform = A.load(IMAGENET_AA_PATH)  # type: ignore
            else:
                transform = A.from_dict(  # type: ignore
                    {
                        "transform": {
                            "__class_fullname__": "albumentations.augmentations.transforms." + transform_name,
                            **transform_args,
                        }
                    }
                )
            transforms.append(transform)
        return transforms

    @staticmethod
    def compose_albumentations_transforms(
        pp_trans: List[A.BasicTransform],
        aug_trans: List[A.BasicTransform],
        mean: Optional[RGBInfo] = None,
        std: Optional[RGBInfo] = None,
    ) -> A.Compose:
        """
        Composes transforms from preprocessing, data augmentation, normalization and tensorization.

        :param pp_trans: list of preprocessing transforms, can be empty
        :param aug_trans: list of data augmentation transforms, can be empty
        :param mean: mean channel values to normalize input data
        :param std: standard deviation values per channel to normalize input data
        :return: a composed albumentation pipeline
        """
        norm_trans = []
        if mean is None and std is None:
            norm_trans.append(A.ToFloat(max_value=255))
        elif sum([x is None for x in [mean, std]]) == 1:
            raise RuntimeError(
                "Was presented either only STD or only MEAN normalization values. Require either none or both!"
            )
        else:
            mean = mean.get_rgb()
            std = std.get_rgb()
            norm_trans.append(
                A.Normalize(
                    mean=mean,
                    std=std,
                )
            )
        return A.Compose([*pp_trans, *aug_trans, *norm_trans, ToTensorV2()])

    @staticmethod
    def get_rand_augment(
        number: int, magnitude: int, p: float = 1.0, mode: str = "all", cut_out: bool = False
    ) -> A.BaseCompose:
        """
        Gets RandAugment transform. For details see https://arxiv.org/abs/1909.13719.

        :param number: number of transforms to be applied (excluding cut_out if active)
        :param magnitude: int between 0 and 9 determining strength of transformation
        :param p: probability to apply RandAugment
        :param mode: either 'geo' for geometrical transforms, 'color' for color transforms or 'all' for both of them
        :param cut_out: indicating if cutout should be applied
        :return: an albumentations transform
        """
        MAX_MAGNITUDE = 10
        assert 0 < magnitude < MAX_MAGNITUDE, f"magnitude range is 1 - 9, was given {magnitude}"
        assert mode in ["geo", "color", "all"], f"incorrect RandAugment mode {mode}, provide one of [color, geo, all]"

        ops = [  # 0 - geometrical
            A.Affine(
                translate_percent=(
                    float(-np.linspace(0, 1, MAX_MAGNITUDE)[magnitude]),
                    float(np.linspace(0, 1, MAX_MAGNITUDE)[magnitude]),
                ),
                p=p,
                cval_mask=255,
            ),
            A.Affine(
                rotate=(
                    float(-np.linspace(0, 45, MAX_MAGNITUDE)[magnitude]),
                    float(np.linspace(0, 45, MAX_MAGNITUDE)[magnitude]),
                ),
                p=p,
                cval_mask=255,
            ),
            A.Affine(
                scale=(
                    1 + float(-np.linspace(0, 0.5, MAX_MAGNITUDE)[magnitude]),
                    1 + float(np.linspace(0, 0.5, MAX_MAGNITUDE)[magnitude]),
                ),
                p=p,
                cval_mask=255,
                keep_ratio=True,
            ),
            A.Affine(
                shear=(
                    float(-np.linspace(0, 35, MAX_MAGNITUDE)[magnitude]),
                    float(np.linspace(0, 35, MAX_MAGNITUDE)[magnitude]),
                ),
                p=p,
                cval_mask=255,
            ),
            # 4 - Color Based
            A.InvertImg(p=p),
            A.Equalize(p=p),
            A.Solarize(threshold=float(np.linspace(0, 256, MAX_MAGNITUDE)[magnitude]), p=p),
            A.Posterize(num_bits=int(np.linspace(0, 8, MAX_MAGNITUDE)[magnitude]), p=p),
            A.RandomBrightnessContrast(
                brightness_limit=float(np.linspace(0, 0.8, MAX_MAGNITUDE)[magnitude]), contrast_limit=0.0, p=p
            ),
            A.RandomBrightnessContrast(
                contrast_limit=float(np.linspace(0, 0.8, MAX_MAGNITUDE)[magnitude]), brightness_limit=0.0, p=p
            ),
            A.Sharpen(
                alpha=(0.1, float(np.linspace(0.1, 0.9, MAX_MAGNITUDE)[magnitude])),
                lightness=(0.4, float(np.linspace(0.4, 1.0, MAX_MAGNITUDE)[magnitude])),
                p=p,
            ),
        ]

        if mode == "geo":
            ops = ops[:4]
        elif mode == "color":
            ops = ops[5:]
        else:
            ops = ops
        transforms = A.SomeOf(transforms=ops, n=number, replace=True, p=1)
        if cut_out:
            transforms = A.Sequential(
                transforms=[
                    transforms,
                    A.CoarseDropout(
                        num_holes_range=(4, 8),
                        hole_height_range=(0.05, float(np.linspace(0.05, 0.2, MAX_MAGNITUDE)[magnitude])),
                        hole_width_range=(0.05, float(np.linspace(0.05, 0.2, MAX_MAGNITUDE)[magnitude])),
                        mask_fill_value=255,
                        p=p,
                    ),
                ]
            )
        return transforms
