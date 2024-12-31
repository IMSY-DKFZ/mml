# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import filecmp
from pathlib import Path

import pytest
from omegaconf import DictConfig, open_dict

from mml.core.data_loading.file_manager import DSET_PREFIX, MMLFileManager, RemoveConfig
from mml.core.data_loading.task_description import TaskDescription


# dummy class
@pytest.fixture
def foo():
    class Foo:
        pass

    return Foo


# tests for path assignments
@pytest.mark.parametrize(argnames="numbering", argvalues=(True, False))
def test_add_new_assignment(file_manager, foo, numbering):
    file_manager.add_assignment_path(foo, key="test", path="TEMP_PATH/test.txt", enable_numbering=numbering)
    constructed = file_manager.construct_saving_path(obj=foo(), key="test")
    assert constructed.exists() == numbering


def test_assignment_numbering(file_manager):
    p1 = file_manager.construct_saving_path(obj=None, key="temp", file_name="dummy.txt")
    p2 = file_manager.construct_saving_path(obj=None, key="temp", file_name="dummy.txt")
    assert p1.exists()
    assert p2.exists()
    assert p1 != p2


def test_assignment_replace_file_name(file_manager):
    p = file_manager.construct_saving_path(obj=None, key="temp", file_name="test123.txt")
    assert p.name == "test123_0001.txt"


def test_assignment_replace_task_id(file_manager, foo):
    file_manager.add_assignment_path(foo, key="test", path=Path("TEMP_PATH") / "TASK_NAME" / "test.txt")
    task_name = "my_test_task"
    constructed = file_manager.construct_saving_path(obj=foo(), key="test", task_name=task_name)
    assert constructed.parent.name == task_name


def test_add_assignment_incorrect_path_beginning(file_manager, foo):
    with pytest.raises(ValueError):
        file_manager.add_assignment_path(foo, key="test", path=Path("/tmp/") / "test.txt")


def test_add_assignment_multiple_keys(file_manager, foo):
    file_manager.add_assignment_path(foo, key="test", path=Path("TEMP_PATH") / "test.txt")
    with pytest.raises(KeyError):
        file_manager.add_assignment_path(foo, key="test", path=Path("TEMP_PATH") / "test.txt")


@pytest.mark.parametrize(
    "path",
    [
        Path("PROJ_PATH") / "TASK_NAME" / "file_name.sth",
        Path("TEMP_PATH") / "TEST" / "TASK_NAME" / "file_name.sth",
        Path("PROJ_PATH") / "TEST" / "ID" / "file_name.sth",
        Path("PROJ_PATH") / "TEST" / "TEST_2" / "TASK_NAME" / "file_name.sth",
    ],
)
def test_add_assignment_reuse_value_errors(file_manager, foo, path):
    with pytest.raises(ValueError):
        file_manager.add_assignment_path(foo, key="test", path=path, reusable=True)


# test other file manager paths
def test_download_path(file_manager):
    d_path = file_manager.get_download_path(dset_name="myTestDataset")
    assert isinstance(d_path, Path)
    assert d_path.exists()
    another_call = file_manager.get_download_path(dset_name="myTestDataset")
    assert d_path == another_call


def test_get_dataset_path(file_manager):
    new_path = file_manager.get_dataset_path(dset_name="myTestDataset")
    assert new_path.exists()
    assert new_path.is_dir()
    assert len(list(new_path.iterdir())) == 0


def test_get_dataset_preprocessing_path(file_manager):
    raw_dset_path = file_manager.raw_data / f"{DSET_PREFIX}_myTestDataset"
    pp_path = file_manager.get_dataset_path(raw_path=raw_dset_path, preprocessing="test_pp")
    with pytest.warns(UserWarning, match="Dataset myTestDataset has already partly existing preprocessing"):
        pp_path_again = file_manager.get_dataset_path(raw_path=raw_dset_path, preprocessing="test_pp")
    assert pp_path == pp_path_again
    assert file_manager.preprocessed_data in pp_path.parents


def test_remove_intermediates(file_manager, foo):
    file_manager.add_assignment_path(foo, key="temp_test", path=Path("TEMP_PATH") / "TEST" / "test.txt")
    file_manager.add_assignment_path(foo, key="proj_test", path=Path("PROJ_PATH") / "TEST" / "test.txt")
    p_1 = file_manager.construct_saving_path(obj=foo(), key="temp_test")
    assert p_1.exists()
    p_2 = file_manager.construct_saving_path(obj=foo(), key="proj_test")
    assert p_2.exists()
    r_cfg = DictConfig(RemoveConfig())
    with open_dict(r_cfg):
        r_cfg.proj_test = True
    file_manager.remove_cfg = r_cfg
    file_manager.remove_intermediates()
    assert not p_1.exists()
    assert not p_2.exists()


def test_load_meta(dummy_meta_class_path):
    task_description = MMLFileManager.load_task_description(dummy_meta_class_path)
    assert isinstance(task_description, TaskDescription)


def test_write_meta(tmp_path):
    store_path = tmp_path / "dummy.json"
    assert not store_path.exists()
    dummy_description = TaskDescription(idx_to_class={0: "a", 1: "b"})
    MMLFileManager.write_task_description(path=store_path, task_description=dummy_description)
    assert store_path.exists()
    with pytest.warns(UserWarning, match="Overwriting"):
        MMLFileManager.write_task_description(path=store_path, task_description=dummy_description)


def test_load_write_meta_consistency(tmp_path, dummy_meta_class_path):
    # dummy_meta_path is formatted human friendly, not as written by FM
    task_description = MMLFileManager.load_task_description(dummy_meta_class_path)
    store_path_1 = tmp_path / "dummy_1.json"
    MMLFileManager.write_task_description(path=store_path_1, task_description=task_description)
    loaded_description = MMLFileManager.load_task_description(store_path_1)
    store_path_2 = tmp_path / "dummy_2.json"
    MMLFileManager.write_task_description(path=store_path_2, task_description=loaded_description)
    assert filecmp.cmp(store_path_1, store_path_2, shallow=False)


@pytest.mark.parametrize("enable_numbering", [True, False])
def test_find_global_reusables(file_manager: MMLFileManager, enable_numbering):
    reuse_cfg = {}
    property_name = "abc"
    file_manager.add_assignment_path(
        obj_cls=None,
        key=property_name,
        path=Path("PROJ_PATH") / property_name / "FILE_NAME",
        enable_numbering=enable_numbering,
        reusable=file_manager.GLOBAL_REUSABLE,
    )
    p = file_manager.construct_saving_path(obj=None, key=property_name, file_name=f"some_{property_name}_file.file")
    if not enable_numbering:
        p.touch(exist_ok=False)
    file_manager.construct_saving_path(obj=None, key=property_name, file_name=f"some_{property_name}_file.file")
    reuse_cfg[property_name] = file_manager.proj_path.name
    file_manager.reuse_cfg = DictConfig(reuse_cfg)
    file_manager._find_reusables()
    assert file_manager.GLOBAL_REUSABLE in file_manager.reusables
    assert property_name in file_manager.global_reusables
    assert isinstance(file_manager.global_reusables[property_name], Path)
    assert file_manager.global_reusables[property_name].exists()


# TODO test_get_all_dset_names, test_reload_task_index, test_get_task_path, ...
