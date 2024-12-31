# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import pytest
from omegaconf import open_dict


# by default the fixture below overrides the no_plugins fixture from mml.testing.fixtures to allow plugin testing
# you may remove this deactivation, put it into another submodule conftest.py and/or use the "plugin" marker to
# deactivate plugin loading solely for specific tests
@pytest.fixture(autouse=True)
def no_plugins(monkeypatch, request):
    yield


@pytest.fixture
def fed_config(mml_config):
    with open_dict(mml_config):
        mml_config.distance = {}
        mml_config.distance.name = "fed"
        mml_config.distance.metric = "cosine"
        mml_config.distance.prefix = ""
        mml_config.distance.fim = {
            "samples": 10,
            "empirical": False,
            "ignore_bias": True,
            "ignore_bn": False,
            "ignore_downsample": True,
            "average_filters": True,
            "final_fraction": 0.6,
            "nngeom": True,
        }
    yield mml_config
