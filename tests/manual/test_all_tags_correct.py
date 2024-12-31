# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import os
from pathlib import Path

import pytest

import mml.core.data_preparation.task_creator as task_creator_module
import mml.core.data_preparation.utils as prep_utils_module
from mml.core.data_loading.file_manager import MMLFileManager


@pytest.mark.skipif("not config.getoption('check_tasks')")
@pytest.mark.env  # use real mml.env
@pytest.mark.plugin  # load actual plugins
def test_installed_tasks(monkeypatch, tmp_path_factory):
    """
    This is a manual test, call with >>pytest --check_tasks -k "installed"<<.
    It checks whether the currently present tasks meta information match the currently coded meta information (imagine
    updating coded task information without adapting installed one).
    """
    # here we will store coded meta
    coded_meta = {}

    # this dummy creator will be mocked and used instead of the actual task creator, it stores passed kwargs
    class DummyTaskCreator:
        def __init__(self, **kwargs):
            nonlocal coded_meta
            kwargs.pop("dset_path", None)
            coded_meta[kwargs["name"]] = kwargs
            raise RuntimeError("Intended interruption.")

    # this dummy iterator will be necessary in cases of dict-like task creators
    def dummy_get_iterator(**kwargs):
        return None, None

    # mock globally, this requires no previous import of the task creators
    # more precisely: the mml.create_tasks.task_preparation module, otherwise the TaskCreator will already be linked!
    # also see https://docs.python.org/dev/library/unittest.mock.html#where-to-patch
    monkeypatch.setattr(target=task_creator_module, name="TaskCreator", value=DummyTaskCreator)
    monkeypatch.setattr(
        target=prep_utils_module, name="get_iterator_and_mapping_from_image_dataset", value=dummy_get_iterator
    )

    # import all the task creators
    import mml.api

    mml.api.load_env()
    mml.api.load_mml_plugins()
    import mml.core.data_preparation.registry

    assert len(mml.core.data_preparation.registry._TASKCREATORS) > 0
    assert len(mml.core.data_preparation.registry._DATASET_CREATORS) > 0
    # iterate over registered coded task creators
    for task_alias, task_creator in mml.core.data_preparation.registry._TASKCREATORS.items():
        if task_alias == "mml_fake_task":
            continue
        dummy_dset_path = Path("/dummy")
        with pytest.raises(RuntimeError, match="Intended interruption."):
            _ = task_creator(dset_path=dummy_dset_path)

    # now load installed tasks (this is why we mark with pytest.mark.env, to use the actual underlying data)
    assignments_backup = MMLFileManager._path_assignments.copy()
    log_path = tmp_path_factory.mktemp(basename="logging")
    monkeypatch.chdir(log_path)
    manager = MMLFileManager(
        data_path=Path(os.getenv("MML_DATA_PATH")),
        proj_path=tmp_path_factory.mktemp(basename="project"),
        log_path=log_path,
    )
    # iterate over installed tasks
    for alias in manager.task_index:
        if alias in coded_meta:
            # load installed meta
            written_meta = MMLFileManager.load_task_description_header(
                manager.data_path / manager.task_index[alias]["none"]
            )
            for attr in coded_meta[alias]:
                kwarg_mapper = {"desc": "description", "ref": "reference", "instr": "download", "lic": "license"}
                if attr in kwarg_mapper:
                    ref = kwarg_mapper[attr]
                else:
                    ref = attr
                assert (
                    getattr(written_meta, ref) == coded_meta[alias][attr]
                ), f"{alias=}, {attr=}, {ref=}, {getattr(written_meta, ref)=}, {coded_meta[alias][attr]=}"
    # clearance
    manager.clear_instance()
    MMLFileManager._path_assignments = assignments_backup
