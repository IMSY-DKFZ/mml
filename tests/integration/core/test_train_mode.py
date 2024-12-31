# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import pytest


@pytest.mark.gpu
def test_train_predict_console(script_runner, fake_task, no_plugins):
    ret = script_runner.run(
        [
            "mml",
            "train",
            "tasks=fake",
            "mode.subroutines=[train,predict]",
            "trainer.max_epochs=1",
            "tune.lr=false",
            "mode.cv=false",
            "mode.nested=false",
        ],
        print_result=True,
    )
    assert ret.success
    assert "MML run time was" in ret.stdout


@pytest.mark.gpu
def test_train_test_console(script_runner, fake_task, no_plugins):
    ret = script_runner.run(
        [
            "mml",
            "train",
            "tasks=fake",
            "mode.subroutines=[train,test]",
            "trainer.max_epochs=1",
            "tune.lr=false",
            "mode.cv=false",
        ],
        print_result=True,
    )
    assert ret.success
    assert "MML run time was" in ret.stdout
