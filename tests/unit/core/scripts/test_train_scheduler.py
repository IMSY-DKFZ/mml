# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest
from omegaconf import OmegaConf

from mml.core.scripts.pipeline_configuration import PIPELINE_CONFIG_PARTS
from mml.core.scripts.schedulers.train_scheduler import TrainingScheduler


@pytest.mark.gpu
@pytest.mark.parametrize("test_task_name", ["test_task_a"])  # TODO add 'test_task_d'
def test_train_probe_model(test_task_monkeypatch, mml_config, test_task_name) -> None:
    OmegaConf.set_struct(mml_config, False)
    mml_config.mode.subroutines = ["train"]
    mml_config.pivot.name = test_task_name
    mml_config.mode.cv = False
    mml_config.mode.nested = False
    mml_config.mode.multitask = False
    mml_config.mode.store_parameters = False
    mml_config.mode.use_blueprint = False
    mml_config.mode.pipeline_keys = PIPELINE_CONFIG_PARTS
    mml_config.mode.store_best = True
    mml_config.mode.eval_on = None
    mml_config.mode.task_weights = None
    OmegaConf.set_struct(mml_config, True)
    with pytest.warns(UserWarning, match="MMLFileManager was not created by BaseScheduler"):
        scheduler = TrainingScheduler(cfg=mml_config)
    scheduler.train_fold(task_name=test_task_name, fold=0)
    # check if model storage has been created
    assert len(scheduler.get_struct(test_task_name).models) == 1
