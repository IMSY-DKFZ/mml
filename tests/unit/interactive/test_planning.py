# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import os.path

import pandas as pd
import pytest

from mml.core.data_loading.task_attributes import Keyword, TaskType
from mml.interactive import _check_init
from mml.interactive.planning import AllTasksInfos, DefaultRequirements, MMLJobDescription, write_out_commands


def test_valid_rendering(script_runner):
    job = MMLJobDescription(
        prefix_req=DefaultRequirements(), mode="info", config_options={"tasks": "none"}, multirun=False
    )
    ret = script_runner.run(job.render().split(" "), print_result=False)
    assert ret.success
    assert "MML run time was" in ret.stdout
    assert ret.stderr == ""


def test_write_out(monkeypatch, tmp_path):
    def mocked_path(*args, **kwargs):
        return tmp_path

    monkeypatch.setattr(target=os.path, name="abspath", value=mocked_path)

    cmds = [
        MMLJobDescription(prefix_req=DefaultRequirements(), mode="dummy", config_options={"optimizer.lr": 0.01}),
        MMLJobDescription(
            prefix_req=DefaultRequirements(),
            mode="dummy",
            config_options={"optimizer.lr": [0.01, 0.001]},
            multirun=True,
        ),
    ]
    assert tmp_path.exists()
    assert len(list(tmp_path.iterdir())) == 0
    write_out_commands(cmd_list=cmds, name="bazz", seperator="", max_cmds=1)
    assert len(list(tmp_path.iterdir())) == 2
    assert all(["bazz" in p.stem for p in tmp_path.iterdir()])


def test_task_info_write_read(tmp_path):
    tasks = ["a", "b", "c"]
    infos = AllTasksInfos(
        num_classes={t: idx for idx, t in enumerate(tasks)},
        num_samples={t: idx * 100 for idx, t in enumerate(tasks)},
        imbalance_ratios={t: idx * 1.2345 for idx, t in enumerate(tasks)},
        datasets={t: "data_" + t for t in tasks},
        keywords={t: set(Keyword(_kw) for _kw in Keyword.list()[: (idx + 1) * 2]) for idx, t in enumerate(tasks)},
        task_types={t: TaskType(TaskType.list()[idx]) for idx, t in enumerate(tasks)},
        domains={t: Keyword(Keyword.list()[idx]) for idx, t in enumerate(tasks)},
        dimensions={t: idx * 13 for idx, t in enumerate(tasks)},
        max_resolution={t: idx * 1000000 for idx, t in enumerate(tasks)},
        min_resolution={t: idx * 1000 for idx, t in enumerate(tasks)},
        small_tasks=[tasks[0]],
        medium_tasks=[tasks[1]],
        large_tasks=[tasks[2]],
    )
    infos.store_csv(tmp_path / "infos.csv")
    df = pd.read_csv(tmp_path / "infos.csv")
    infos2 = AllTasksInfos.from_csv(tmp_path / "infos.csv")
    infos2.store_csv(tmp_path / "infos2.csv")
    df2 = pd.read_csv(tmp_path / "infos2.csv")
    pd.testing.assert_frame_equal(df.set_index("name"), df2.set_index("name")[df.set_index("name").columns])


@pytest.mark.env
def test_init_check():
    with pytest.raises(RuntimeError):
        _check_init()
