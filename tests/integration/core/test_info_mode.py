# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#


def test_info_scheduler_console(script_runner, no_plugins):
    ret = script_runner.run(["mml", "info", "tasks=none"], print_result=False)
    assert ret.success
    assert "MML run time was" in ret.stdout
    assert ret.stderr == ""
