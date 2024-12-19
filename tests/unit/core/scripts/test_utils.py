# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import time

import pytest

from mml.core.scripts.decorators import beta
from mml.core.scripts.utils import Singleton, StrEnum, catch_time, throttle_logging


def test_catch_time():
    with catch_time() as timer:
        time.sleep(1.0)
    assert timer.seconds > 0
    assert len(timer.pretty_time) > 0


def test_throttle_logging():
    # dummy handler that raises an error if a message is emmitted
    class DummyHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            raise RuntimeError("Should not be emitted!")

    logger = logging.getLogger("mml.test.dummy")
    logger.addHandler(hdlr=DummyHandler())
    logger.setLevel(level=logging.DEBUG)
    # test 1: no message if logging is throttled above
    with throttle_logging(logging.INFO):
        logger.info("something")
    # test 2: message if logging is throttled below
    with pytest.raises(RuntimeError):
        with throttle_logging(logging.DEBUG):
            logger.info("something")
    # test 3: message if logging is throttled on a different logger
    with pytest.raises(RuntimeError):
        with throttle_logging(logging.INFO, package="mml.test.other"):
            logger.info("something")
    # test 4: no message if logging is throttled on that exact logger
    with throttle_logging(logging.INFO, package="mml.test.dummy"):
        logger.info("something")
    # test 5: reset of level
    assert logger.level == logging.DEBUG


def test_singleton_class():
    class Dummy(Singleton):
        def __init__(self, x):
            self.x = x

    instance = Dummy(3)
    assert Dummy.instance() is instance
    assert Dummy.instance(5).x == 3
    assert Dummy.exists()
    Dummy.clear_instance()
    assert Dummy.instance(5).x == 5
    assert Dummy.instance() is not instance


def test_strenum():
    class Dummy(StrEnum):
        A = "a"
        B = "bb"
        C = "ccc"

    assert Dummy.C is Dummy.from_str("c")
    assert Dummy.B == "bb"
    with pytest.raises(ValueError):
        Dummy.from_str("aaa")


def test_beta_decorator(monkeypatch):
    @beta("some message")
    class DummyClass:
        pass

    # raises first time
    with pytest.warns(UserWarning, match="some message"):
        DummyClass()
    # no raise
    DummyClass()

    @beta("some message")
    def func():
        pass

    # raises again, since new caller
    with pytest.warns(UserWarning, match="some message"):
        func()
