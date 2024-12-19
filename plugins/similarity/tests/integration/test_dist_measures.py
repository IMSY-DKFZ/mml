# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("dist_measure", ["kld", "emd", "fed", "fid", "mmd", "semantic"])
def test_dist_measures_console(script_runner, fake_task, no_plugins, dist_measure):
    ret = script_runner.run(
        [
            "mml",
            "similarity",
            "tasks=fake",
            f"distance={dist_measure}",
        ],
        print_result=True,
    )
    assert ret.success
