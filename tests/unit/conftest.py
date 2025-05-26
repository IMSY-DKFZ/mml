# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest
from omegaconf import OmegaConf

from mml.core.data_loading.task_attributes import Keyword, Modality, RGBInfo, Sizes, TaskType
from mml.core.data_loading.task_struct import TaskStruct
from mml.core.scripts.model_storage import ModelStorage
from mml.testing.boring_model import BoringModel


@pytest.fixture()
def dummy_task_struct():
    yield TaskStruct(
        name="dummy",
        task_type=TaskType.CLASSIFICATION,
        means=RGBInfo(*[0.1, 0.2, 0.3]),
        stds=RGBInfo(*[0.4, 0.5, 0.6]),
        sizes=Sizes(*[1, 2, 3, 4]),
        relative_root="dummy_root",
        class_occ={"one": 1, "two": 2},
        preprocessed="none",
        keywords=[Keyword.ARTIFICIAL],
        idx_to_class={0: "one", 1: "two"},
        modalities={Modality.CLASS: "stuff"},
    )


@pytest.fixture()
def dummy_model_storage(file_manager, dummy_task_struct):
    par_path = file_manager.construct_saving_path(BoringModel(), key="parameters", task_name=dummy_task_struct.name)
    pip_path = file_manager.construct_saving_path(
        obj=OmegaConf.create({}), key="pipeline", task_name=dummy_task_struct.name
    )
    pred_paths = [
        file_manager.construct_saving_path(
            obj=dict(), key="predictions", task_name=dummy_task_struct.name, file_name="preds-test-fold-0.pt"
        )
        for _ in range(3)
    ]
    s = ModelStorage(
        pipeline=pip_path,
        parameters=par_path,
        performance=0.42,
        training_time=13.37,
        predictions={k: v for k, v in zip("abc", pred_paths)},
        metrics=[{"loss": 0.5, "acc": 0.1}, {"loss": 0.4, "acc": 0.2}],
    )
    yield s
