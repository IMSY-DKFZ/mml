# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import shutil

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mml.core.data_loading.task_struct import TaskStructFactory
from mml.core.scripts.model_storage import ModelStorage


@pytest.mark.gpu
def test_suggest_mode_console(
    script_runner, test_task_monkeypatch, file_manager, dummy_fake_pipeline_path, dummy_fake_model_storage_path
):
    # dump some models and pipelines
    task_list = ["test_task_a"]
    for char in "bc":
        task = f"test_task_{char}"
        task_list.append(task)
        struct = TaskStructFactory.get_by_name(self=None, name=task)  # this is monkeypatched by test_task_monkeypatch
        for _ in range(3):
            p = file_manager.construct_saving_path(OmegaConf.create({}), key="pipeline", task_name=task)
            shutil.copy2(src=dummy_fake_pipeline_path, dst=p)
            model = ModelStorage.from_json(dummy_fake_model_storage_path)
            model.pipeline = p
            model_path = file_manager.construct_saving_path(
                model, key="models", task_name=task, file_name="model_storage.json"
            )
            model.store(path=model_path)
            struct.models.append(model)  # provide models while side passing reuse functionality
    df = pd.DataFrame(data=np.random.rand(len(task_list), len(task_list)), columns=task_list, index=task_list)
    # ensure saving path is assigned
    import mml_similarity.activate  # noqa

    # store dummy distances
    df.to_csv(file_manager.construct_saving_path(df, key="fed"))
    # since file manager is already initialized, need to manually trigger reuse
    file_manager.reuse_cfg = OmegaConf.create({"fed": file_manager.proj_path.name})
    file_manager._find_reusables()
    ret = script_runner.run(
        [
            "mml",
            "suggest",
            "pivot.name=test_task_a",  # mml should deduce the other tasks as source tasks
            # f"+reuse.fed={file_manager.proj_path.name}",  # is triggered above
            # f"reuse.models={file_manager.proj_path.name}",  # is triggered above
        ],
        print_result=True,
    )
    assert ret.success
