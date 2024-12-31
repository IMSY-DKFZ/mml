# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from hydra.utils import instantiate

from mml.core.data_loading.augmentations.mixup_cutmix import CutMixCallback, MixUpCallback
from mml.core.models.lightning_single_frame import SingleFrameLightningModule


def test_mixup_callback(mml_config, datamodule):
    mixup = MixUpCallback()
    trainer = instantiate(mml_config.trainer, callbacks=[mixup], fast_dev_run=True)
    model = SingleFrameLightningModule(task_structs=datamodule.task_structs, cfg=mml_config)
    trainer.fit(model, datamodule=datamodule)


def test_cutmix_callback(mml_config, datamodule):
    cutmix = CutMixCallback()
    trainer = instantiate(mml_config.trainer, callbacks=[cutmix], fast_dev_run=True)
    model = SingleFrameLightningModule(task_structs=datamodule.task_structs, cfg=mml_config)
    trainer.fit(model, datamodule=datamodule)
