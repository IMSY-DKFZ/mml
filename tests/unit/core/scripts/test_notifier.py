# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest

from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.notifier import BaseNotifier, DummyNotifier


def test_is_master(monkeypatch):
    assert BaseNotifier.is_master()
    monkeypatch.setenv(name="RANK", value="0")
    assert BaseNotifier.is_master()
    monkeypatch.setenv(name="RANK", value="42")
    assert not BaseNotifier.is_master()


@pytest.mark.parametrize(argnames="exception", argvalues=[OSError(), MMLMisconfigurationException(), RuntimeError()])
def test_create_failure_message(exception):
    notifier = DummyNotifier(on_failure=True)
    with pytest.warns():
        notifier.notify_on_failure(error=exception)
