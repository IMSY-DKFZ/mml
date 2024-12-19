# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

# TODO
import pytest

from mml.core.data_loading.file_manager import MMLFileManager


@pytest.mark.benchmark(group="file_manager_init")
def test_file_manager_init_speed(tmp_path_factory, monkeypatch, benchmark):
    # store class attributes
    assignments_backup = MMLFileManager._path_assignments.copy()
    log_path = tmp_path_factory.mktemp(basename="logging")
    monkeypatch.chdir(log_path)

    def init_fm():
        _ = MMLFileManager(
            data_path=tmp_path_factory.mktemp(basename="data"),
            proj_path=tmp_path_factory.mktemp(basename="project"),
            log_path=log_path,
        )
        MMLFileManager.clear_instance()

    benchmark(init_fm)
    # correct teardown
    MMLFileManager._path_assignments = assignments_backup
