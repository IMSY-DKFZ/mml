# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from mml.core.scripts.model_storage import ModelStorage


def test_restoring(dummy_model_storage, dummy_task_struct):
    path = dummy_model_storage.store(task_struct=dummy_task_struct)
    loaded = ModelStorage.from_json(path=path)
    print(loaded)
    print(dummy_model_storage)
    assert dummy_model_storage == loaded


def test_updating(dummy_model_storage, dummy_task_struct):
    path = dummy_model_storage.store(task_struct=dummy_task_struct)
    dummy_model_storage.training_time += 1.0
    updated_path = dummy_model_storage.store(path=path)
    assert updated_path == path
