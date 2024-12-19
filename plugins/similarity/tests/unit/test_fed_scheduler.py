# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
from mml_similarity.scripts.fed_scheduler import FEDScheduler


@pytest.mark.gpu
def test_train_probe_model(test_task_monkeypatch, fed_config) -> None:
    fed_config.mode.subroutines = ["tune"]
    scheduler = FEDScheduler(cfg=fed_config)
    scheduler.train_probe_model(task_name="test_task_a")
    assert "fc_tuned" in test_task_monkeypatch["test_task_a"].paths


@pytest.mark.gpu
def test_compute_fim(test_task_monkeypatch, fed_config, tmp_path) -> None:
    fed_config.mode.subroutines = ["fim"]
    scheduler = FEDScheduler(cfg=fed_config)
    # simulate saved tuned params
    model = scheduler.create_model([test_task_monkeypatch["test_task_a"]]).model
    save_path = tmp_path / "tmp.pth"
    model.save_checkpoint(param_path=save_path)
    test_task_monkeypatch["test_task_a"].paths["fc_tuned"] = save_path
    scheduler.compute_fim(task_name="test_task_a")
    assert "fim" in test_task_monkeypatch["test_task_a"].paths


def test_compute_distance(test_task_monkeypatch, fed_config, tmp_path) -> None:
    fed_config.mode.subroutines = ["distance"]
    fed_config.task_list = ["test_task_a", "test_task_b"]
    scheduler = FEDScheduler(cfg=fed_config)
    scheduler.after_preparation_hook()
    # simulate saved fim
    fim_a = {
        "mod0": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        "mod1": torch.tensor([3.1415]),
        "mod2": torch.tensor(
            [
                [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ]
        ),
    }
    fim_b = {
        "mod0": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]) + 1,
        "mod1": torch.tensor([3.1415]) * 2,
        "mod2": torch.tensor(
            [
                [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ]
        )
        ** 2,
    }
    save_path_a = tmp_path / "fim_a.pkl"
    save_path_b = tmp_path / "fim_b.pkl"
    torch.save(fim_a, save_path_a)
    torch.save(fim_b, save_path_b)
    test_task_monkeypatch["test_task_a"].paths["fim"] = save_path_a
    test_task_monkeypatch["test_task_b"].paths["fim"] = save_path_b
    scheduler._compute_distance(source="test_task_a", target="test_task_b")
    assert pytest.approx(0.0502121448516845) == scheduler.load_distances().at["test_task_a", "test_task_b"]
