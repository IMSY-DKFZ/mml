# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import omegaconf
import pytest

from mml.core.data_loading.lightning_datamodule import MultiTaskDataModule
from mml.core.data_loading.task_struct import TaskStructFactory


@pytest.fixture()
def dummy_factory(file_manager):
    dummy_cfg = omegaconf.OmegaConf.create({"preprocessing": {"id": "none"}})
    yield TaskStructFactory(cfg=dummy_cfg, load=False)


@pytest.fixture(scope="function", params=["a", "b", "c"])
def datamodule(mml_config, test_task_monkeypatch, request):
    yield MultiTaskDataModule(task_structs=[test_task_monkeypatch[f"test_task_{request.param}"]], cfg=mml_config)
