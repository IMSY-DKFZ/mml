# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#


def test_create_scheduler_console(script_runner, file_manager):
    ret = script_runner.run(["mml", "create", "task_list=[mml_fake_task]"], print_result=True)
    assert ret.success
    assert "MML run time was" in ret.stdout
