# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest
from omegaconf import OmegaConf

from mml.core.scripts.schedulers.train_scheduler import TrainingScheduler


@pytest.mark.gpu
@pytest.mark.benchmark(group="training_epoch")
def test_training_speed(mml_config, benchmark, file_manager, fake_task):
    OmegaConf.set_struct(mml_config, False)
    mml_config.mode.subroutines = ["train"]
    mml_config.pivot.name = "mml_fake_task"
    mml_config.mode.cv = False
    mml_config.mode.nested = False
    mml_config.mode.multitask = False
    mml_config.mode.store_parameters = False
    mml_config.mode.use_blueprint = False
    mml_config.mode.pipeline_keys = None
    mml_config.mode.store_best = True
    mml_config.mode.eval_on = None
    mml_config.mode.task_weights = None
    mml_config.trainer.max_epochs = 1
    OmegaConf.set_struct(mml_config, True)

    scheduler = TrainingScheduler(mml_config)
    scheduler.prepare_exp()

    benchmark(scheduler.train_fold, task_name="mml_fake_task", fold=0)
