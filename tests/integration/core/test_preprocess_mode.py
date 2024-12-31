# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#


def test_preprocess_scheduler_console(script_runner, fake_task, no_plugins):
    ret = script_runner.run(["mml", "pp", "task_list=[mml_fake_task]", "num_workers=1"], print_result=True)
    assert ret.success
    assert "MML run time was" in ret.stdout
