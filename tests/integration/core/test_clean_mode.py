# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import shutil

from mml.core.data_loading.file_manager import MMLFileManager


def test_clean_scheduler_console(script_runner, file_manager: MMLFileManager, fake_task, no_plugins):
    # create download path
    download_path = file_manager.get_download_path("mml_fake_dataset")
    # create temp.json
    json_path = file_manager.data_path / file_manager.task_index["mml_fake_task"]["none"]
    dst_path = json_path.parent / "temp.json"
    shutil.copy(src=json_path, dst=dst_path)
    ret = script_runner.run(["mml", "clean", "task_list=[mml_fake_task]", "mode.force=True"], print_result=True)
    assert ret.success
    assert "MML run time was" in ret.stdout
    assert not dst_path.exists()
    assert json_path.exists()
    assert not download_path.exists()
