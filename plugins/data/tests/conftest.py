# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import pytest


# by default the fixture below overrides the no_plugins fixture from mml.testing.fixtures to allow plugin testing
# you may remove this deactivation, put it into another submodule conftest.py and/or use the "plugin" marker to
# deactivate plugin loading solely for specific tests
@pytest.fixture(autouse=True)
def no_plugins(monkeypatch, request):
    yield
