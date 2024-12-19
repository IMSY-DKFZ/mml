# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import omegaconf
import pytest

from mml.core.data_loading.augmentations.albumentations import AlbumentationsAugmentationModule
from mml.core.data_loading.task_attributes import Modality


def test_valid_transform_from_config(image):
    conf_str = """
    pipeline:
     - name: ShiftScaleRotate
       shift_limit: 0.2
       scale_limit: 0.2
       rotate_limit: 30
       p: 0.5
     - name: RGBShift
       r_shift_limit: 15
       g_shift_limit: 15
       b_shift_limit: 15
       p: 0.5
     - name: RandomBrightnessContrast
       p: 0.5
    """
    conf = omegaconf.OmegaConf.create(conf_str)
    transforms = AlbumentationsAugmentationModule(
        cfg=conf["pipeline"],
        device="cpu",
        is_last=False,
        is_first=False,
        means=None,
        stds=None,
        floatify=False,
        tensorize=False,
    )
    assert len(transforms) == 3
    # apply transform
    transforms(**{Modality.IMAGE.value: image})


def test_invalid_transform_from_config():
    conf_str = """
        pipeline:
          - name: MyImaginedTransform
            p: 1.5
          - name: RandomBrightnessContrast
            p: 0.5
        """
    conf = omegaconf.OmegaConf.create(conf_str)
    with pytest.raises(KeyError, match="MyImaginedTransform"):
        AlbumentationsAugmentationModule(
            cfg=conf["pipeline"], device="cpu", is_last=False, is_first=False, means=None, stds=None
        )
