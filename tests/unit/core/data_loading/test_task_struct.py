# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from mml.core.data_loading.task_struct import TaskStruct, undup_names
from mml.core.scripts.exceptions import TaskNotFoundError
from mml.core.scripts.model_storage import ModelStorage
from mml.core.scripts.utils import ARG_SEP, TAG_SEP


def test_paths_representation(dummy_task_struct):
    attr = TaskStruct.non_permanent_task_attributes()
    test_path = Path("this/is/a/test")
    dummy_task_struct.paths["test"] = test_path
    path_representer, path_instantiator = attr["paths"]
    representation = path_representer(dummy_task_struct.paths)
    recreated = path_instantiator(representation)
    assert dummy_task_struct.paths == recreated


def test_models_representation(dummy_task_struct, file_manager):
    attr = TaskStruct.non_permanent_task_attributes()
    par_path = file_manager.construct_saving_path(ModelCheckpoint(), key="parameters", task_name=dummy_task_struct.name)
    pip_path = file_manager.construct_saving_path(
        obj=OmegaConf.create({}), key="pipeline", task_name=dummy_task_struct.name
    )
    test_model = ModelStorage(parameters=par_path, pipeline=pip_path, performance=0.3, metrics=[{"test": 0.3}])
    test_model.store(task_struct=dummy_task_struct)
    dummy_task_struct.models.append(test_model)
    model_representer, modelinstantiator = attr["models"]
    representation = model_representer(dummy_task_struct.models)
    recreated = modelinstantiator(representation)
    assert dummy_task_struct.models == recreated


@pytest.mark.parametrize(
    "task_name", ["test", f"test{TAG_SEP}duplicate{ARG_SEP}123", f"test{TAG_SEP}stuff{TAG_SEP}duplicate{ARG_SEP}0"]
)
def test_undup_names(task_name):
    result = undup_names([task_name])[0]
    assert f"{TAG_SEP}duplicate" not in result
    assert undup_names([result])[0] == result
    if f"{TAG_SEP}duplicate" not in task_name:
        assert result == task_name


def test_get_by_name(dummy_factory, dummy_task_struct):
    dummy_factory.container.append(dummy_task_struct)
    response = dummy_factory.get_by_name("dummy")
    assert response is dummy_task_struct
    assert not dummy_factory.check_exists("another_dummy")
    with pytest.raises(TaskNotFoundError):
        dummy_factory.get_by_name("foo")


def test_create_task(dummy_meta_class_path, dummy_factory):
    dummy_factory.fm.task_index["dummy_task"] = {}
    dummy_factory.fm.task_index["dummy_task"]["none"] = dummy_meta_class_path
    struct = dummy_factory.create_task_struct(name="dummy_task", return_ref=True)
    assert struct in dummy_factory.container
    assert dummy_factory.sizes.to_list() == struct.sizes.to_list()


def test_dumping_and_loading(dummy_meta_class_path, dummy_factory):
    assert not dummy_factory.fm.task_dump_path.exists()
    dummy_factory.fm.task_index["dummy_task"] = {}
    dummy_factory.fm.task_index["dummy_task"]["none"] = dummy_meta_class_path
    dummy_factory.create_task_struct(name="dummy_task", return_ref=False)
    dummy_factory.dump(clear_container=True)
    assert dummy_factory.fm.task_dump_path.exists()
    assert len(dummy_factory.container) == 0
    assert dummy_factory.sizes.min_width == dummy_factory.sizes.min_height == 100000
    dummy_factory.loading_old_dump()
    assert len(dummy_factory.container) == 1
    assert dummy_factory.sizes.min_height <= dummy_factory.sizes.max_height
    assert dummy_factory.sizes.min_width <= dummy_factory.sizes.max_width
