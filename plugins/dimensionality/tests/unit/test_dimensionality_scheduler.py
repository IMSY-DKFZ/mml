# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest
from mml_dimensionality.scripts.dimensionality_scheduler import DimensionalityScheduler
from omegaconf import OmegaConf


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("test_task_name", ["test_task_a", "test_task_d"])
def test_train_probe_model(test_task_monkeypatch, mml_config, test_task_name) -> None:
    OmegaConf.set_struct(mml_config, False)
    mml_config.allow_gpu = False
    mml_config.mode.subroutines = ["estimate"]
    mml_config.mode.k = 5
    mml_config.mode.max_subsets = 2
    mml_config.mode.subset_min_size = 5
    mml_config.mode.inv_mle = True
    mml_config.sampling.sample_num = 10
    mml_config.augmentations.normalization = None
    mml_config.augmentations.pipeline = {}
    OmegaConf.set_struct(mml_config, True)
    scheduler = DimensionalityScheduler(cfg=mml_config)
    scheduler.estimate_task_dimensionality(task_name=test_task_name)
    assert "dimension" in test_task_monkeypatch[test_task_name].paths
