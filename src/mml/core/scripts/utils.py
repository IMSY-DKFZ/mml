# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar

import p_tqdm.p_tqdm as p_tqdm
from dotenv import load_dotenv
from pathos.multiprocessing import ProcessPool
from pathos.threading import ThreadPool

import mml
from mml.core.scripts.decorators import timeout
from mml.core.scripts.exceptions import MMLMisconfigurationException

T = TypeVar("T")
TSingleton = TypeVar("TSingleton", bound="Singleton")

__all__ = [
    "catch_time",
    "throttle_logging",
    "load_env",
    "multi_threaded_p_tqdm",
    "Singleton",
    "LearningPhase",
    "load_mml_plugins",
    "TAG_SEP",
    "ARG_SEP",
    "ask_confirmation",
]

logger = logging.getLogger(__name__)

TAG_SEP = "+"
ARG_SEP = "?"
# provides information on loaded plugins
MML_PLUGINS_LOADED = {}


class catch_time:
    """
    Timing utility context manager. Usage:

    .. code-block: python

        with catch_time() as timer:
            # some code

    access time via `timer.pretty_time` afterward, e.g. for logging.
    """

    def __enter__(self) -> "catch_time":
        self.elapsed = time.monotonic()
        # self.time = datetime.datetime.now().replace(microsecond=0)
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]
    ) -> None:
        self.elapsed = time.monotonic() - self.elapsed
        self.hours, rest = divmod(self.elapsed, 3600)
        self.minutes, self.seconds = divmod(rest, 60)
        self.pretty_time = f"{self.hours}h {self.minutes}m {self.seconds:5.2f}s"


class throttle_logging:
    """
    Logging utility context manager. Usage:

    .. code-block: python

        with throttle_logging(logging.SOME_LEVEL, (optional) package):
            # some code that will only propagate logging above (excluding) specified level (of package if given)

    afterwards logging continues as before. The context manager checks if the root logger is in DEBUG mode and prevents
    throttling in that case.
    """

    def __init__(self, level: int = logging.INFO, package: Optional[str] = None):
        self.level = level
        self.logger = logging.getLogger(package) if package else None
        self.stored_level = self.logger.level if self.logger else None

    def __enter__(self) -> "throttle_logging":
        if logging.root.level > logging.DEBUG:
            if self.logger:
                self.logger.setLevel(self.level + 1)
            else:
                logging.disable(self.level)
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]
    ) -> None:
        if self.logger:
            self.logger.setLevel(self.stored_level)  # type: ignore
        else:
            logging.disable(logging.NOTSET)


def load_env() -> None:
    """
    Loads the `mml.env` variables. Make sure to have renamed and adapted `example.env` beforehand. If an
    environment variable `MML_ENV_PATH` is given this file is preferred, else the default path inside the `mml`
    package is used.

    :return: None
    """
    if os.getenv("MML_ENV_PATH", None):
        dotenv_path = Path(os.getenv("MML_ENV_PATH"))
        logger.debug(f"MML_ENV_PATH provided, will try to load env variables from {dotenv_path}.")
    else:
        dotenv_path = Path(mml.__file__).parent / "mml.env"
        logger.debug(f"No MML_ENV_PATH provided, will try to load env variables from default path ({dotenv_path}).")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        if not Path(os.getenv("MML_DATA_PATH")).exists():
            raise MMLMisconfigurationException("Invalid MML_DATA_PATH, have you modified the mml.env entry?")
        if not Path(os.getenv("MML_RESULTS_PATH")).exists():
            raise MMLMisconfigurationException("Invalid MML_RESULTS_PATH, have you modified the mml.env entry?")
        try:
            _ = int(os.getenv("MML_LOCAL_WORKERS"))
        except ValueError:
            raise MMLMisconfigurationException("Invalid MML_LOCAL_WORKERS, have you modified the mml.env entry?")
        if not Path(os.getenv("MML_DATA_PATH")).exists():
            raise MMLMisconfigurationException("Invalid MML_DATA_PATH, have you modified the mml.env entry?")
    else:
        raise MMLMisconfigurationException(
            f".env file not found at {dotenv_path}! Please follow the documentation instructions to setup MML."
        )


class multi_threaded_p_tqdm:
    """
    Switches the internally used pool type of p_tqdm package from ProcessPool to ThreadPool.
    """

    def __enter__(self) -> None:
        p_tqdm.Pool = ThreadPool

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]
    ) -> None:
        p_tqdm.Pool = ProcessPool


class _SingletonMeta(type):
    """
    This is a helper Metaclass to implement the Singleton class.
    """

    _instances: ClassVar[Dict[Type[T], T]] = {}

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in cls._instances:
            cls._instances[cls] = super(_SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(metaclass=_SingletonMeta):
    """
    The actual Singleton class to inherit from. Make sure to have Singleton as the leftmost base class.
    """

    @classmethod
    def clear_instance(cls) -> None:
        """
        Clears the cached instance of the singleton. Be aware, that this does not affect references to the "old"
        instance, but any further call of "instance" or Class() will create a new instance, that will be returned from
        any further call.

        :return:
        """
        cls._instances.pop(cls)

    @classmethod
    def instance(cls: Type[TSingleton], *args: Any, **kwargs: Any) -> TSingleton:
        """
        Convenience function that does the same as Class(), but makes the singleton property more readable in code.

        :param args: any init args, be aware that these are ignored if there already exists an instance
        :param kwargs: any init kwargs, be aware that these are ignored if there already exists an instance
        :return: either a new instance (first call) or a reference to the already existing instance
        """
        return cls.__call__(*args, **kwargs)

    @classmethod
    def exists(cls) -> bool:
        return cls in cls._instances


class StrEnum(str, Enum):
    """
    Type of any enumerator with allowed comparison to string invariant to cases.

    Adopted from :class:`~pytorch_lightning.utilities.enums.LightningEnum`.
    """

    @classmethod
    def from_str(cls, value: str) -> Optional["StrEnum"]:
        for enum_key, enum_val in cls.__members__.items():
            if enum_val.lower() == value.lower() or enum_key.lower() == value.lower():
                return cls[enum_key]
        raise ValueError(f"No match found for value {value} in enum {cls.__name__}")

    def __str__(self) -> str:
        return self.value.lower()

    def __eq__(self, other: object) -> bool:
        other = other.value if isinstance(other, Enum) else str(other)
        return self.value.lower() == other.lower()

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        return hash(self.value.lower())

    @classmethod
    def list(cls) -> List[str]:
        """
        Lists all members of a StrEnum class.
        """
        return list(map(lambda c: c.value, cls))


class LearningPhase(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @staticmethod
    def all_phases() -> List["LearningPhase"]:
        return [LearningPhase.TRAIN, LearningPhase.VAL, LearningPhase.TEST]


def load_mml_plugins() -> None:
    """
    This function allows to load mml plugins. These are other installed packages that provide a 'mml.plugins' entry
    point. See https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata
    for details on this mechanism.
    """
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points, version
    else:
        from importlib.metadata import entry_points, version
    # load registered plugins
    discovered_plugins = entry_points(group="mml.plugins")
    if len(discovered_plugins) > 0:
        # usually logger is not yet configured by hydra, so plugins are also stored in global variable later on
        logger.info(f"Discovered plugins: {[p.name for p in discovered_plugins]}!")
        for plugin in discovered_plugins:
            logger.debug(f"Loading plugin {plugin.name}.")
            _ = plugin.load()
            logger.debug(f"Successfully loaded plugin {plugin.name}.")
    global MML_PLUGINS_LOADED
    MML_PLUGINS_LOADED.update({p.name: version(p.module.split(".")[0]) for p in discovered_plugins})


@timeout(seconds=60)
def ask_confirmation(message: str = "") -> bool:
    """
    Lets user confirm a message.

    :param message: The message to be confirmed
    :return: bool indicating whether the message has been confirmed
    :rtype: bool
    """
    print(message)
    response = input(">>")
    if response.lower().strip() != "y":
        return False
    else:
        return True
