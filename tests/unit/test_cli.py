# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import mml


def test_version_tag(script_runner):
    ret = script_runner.run(["mml", "--version"])
    assert ret.success
    assert f"mml-core {mml.__version__}" in ret.stdout
    assert "Hydra" not in ret.stdout
