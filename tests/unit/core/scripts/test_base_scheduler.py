# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import os
import warnings
from pathlib import Path

import pytest

import mml.core.scripts.callbacks
import mml.core.scripts.schedulers.base_scheduler as base_scheduler_module
import mml.core.scripts.utils
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler


@pytest.fixture
def base_scheduler(monkeypatch, mml_config, file_manager):
    # enables testing of abstract method, also ensures that the file manager is set correctly before testing scheduler
    monkeypatch.setattr(AbstractBaseScheduler, "__abstractmethods__", set())
    mml_config["allow_gpu"] = False
    yield AbstractBaseScheduler(cfg=mml_config, available_subroutines=["test"])


def test_routine(base_scheduler, file_manager):
    with open(base_scheduler.planned_schedule, "r") as file:
        lines = file.readlines()
        assert lines == ["method: prepare_exp / []\n", "method: finish_exp / []\n"]
    with open(base_scheduler.status_log, "r") as file:
        lines = file.readlines()
        assert lines[:2] == ["HEADER\n", "Timepoint of beginning\n"]
        assert lines[3] == "START\n"


def test_lock_is_set(base_scheduler, file_manager):
    lock_path = Path(os.getcwd()) / "lock.tmp"
    assert lock_path.exists()


def test_lock_prevents_scheduler_race(base_scheduler, file_manager, mml_config):
    with pytest.raises(RuntimeError, match="lock"):
        AbstractBaseScheduler(cfg=mml_config, available_subroutines=["test"])


def test_create_trainer_with_callbacks(base_scheduler, mml_config):
    trainer = base_scheduler.create_trainer(monitor=None, metrics_callback=True)
    assert base_scheduler.checkpoint_callback in trainer.callbacks
    assert base_scheduler.metrics_callback in trainer.callbacks
    assert any([isinstance(cb, mml.core.scripts.callbacks.StopAfterKeyboardInterrupt) for cb in trainer.callbacks])


def test_after_init_hooks(base_scheduler, monkeypatch):
    def dummy(scheduler):
        warnings.warn("success!")

    monkeypatch.setattr(base_scheduler_module, "AFTER_SCHEDULER_INIT_HOOKS", [dummy])
    with pytest.warns(match="success"):
        base_scheduler._run_after_init_hooks()
