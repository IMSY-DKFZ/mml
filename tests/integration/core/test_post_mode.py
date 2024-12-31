# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import shutil

import pytest
import torch
from pytest_console_scripts import ScriptRunner

from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.scripts.model_storage import ModelStorage


@pytest.mark.gpu
def test_postprocess_console(
    script_runner: ScriptRunner,
    fake_task,
    no_plugins,
    file_manager: MMLFileManager,
    monkeypatch,
    dummy_fake_model_storage_path,
    dummy_fake_predictions_path,
):
    # need to monkeypatch prediction loading
    orig_load = torch.load

    def mock_torch_load(path):
        return orig_load(dummy_fake_predictions_path)

    monkeypatch.setattr(torch, "load", mock_torch_load)

    # schedule updates existing predictions - we avoid this update
    def mock_torch_save(*args, **kwargs):
        pass

    monkeypatch.setattr(torch, "save", mock_torch_save)
    # create dummy model storage
    destination_dummy_storage = file_manager.proj_path / "MODELS" / "mml_fake_task" / "dummy.json"
    destination_dummy_storage.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(src=dummy_fake_model_storage_path, dst=destination_dummy_storage)
    dummy_model_storage = ModelStorage.from_json(path=destination_dummy_storage)
    # attach models to file_manager
    file_manager.reusables["mml_fake_task"] = {}
    file_manager.reusables["mml_fake_task"]["models"] = [dummy_model_storage] * 10
    # next try to postprocess
    ret = script_runner.run(
        [
            "mml",
            "post",
            "tasks=fake",
            "reuse=current",
            "mode.subroutines=[calibrate,ensemble]",
        ],
        print_result=True,
    )
    assert ret.success
