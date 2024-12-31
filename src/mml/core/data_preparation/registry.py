# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Any, Callable, Dict, Union

_DATASET_CREATORS: Dict[str, Callable[[], Path]] = {}
_TASKCREATORS: Dict[str, Callable[[Path], None]] = {}
_TASK_TO_DSET: Dict[str, str] = {}


def register_dsetcreator(dset_name: str) -> Callable[[Callable[[], Path]], Callable[[], Path]]:
    """
    Registers a dataset creator.

    :param dset_name: the dataset to be linked with
    :return: identical creator, but has been linked
    """

    def decorator(dset_creation_func: Callable[[], Path]) -> Callable[[], Path]:
        _DATASET_CREATORS[dset_name] = dset_creation_func
        return dset_creation_func

    return decorator


def register_taskcreator(
    task_name: str, dset_name: str
) -> Callable[[Callable[[Path], None]], Callable[[Path], Union[None, Path]]]:
    """
    Registers a task creator.

    :param task_name: the task name to be linked with
    :param dset_name: the dataset to be linked with
    :return:
    """

    def decorator(task_creation_func: Callable[[Path], None]) -> Callable[[Path], Union[None, Path]]:
        _TASKCREATORS[task_name] = task_creation_func
        _TASK_TO_DSET[task_name] = dset_name
        return task_creation_func

    return decorator


def create_creator_func(create_func: Callable[..., None], **kwargs: Any) -> Callable[[Path], None]:
    """
    Convenience function if multiple task creation function shall be created dynamically.

    :param create_func: the actual creation function that takes more kwargs than solely the dset_path
    :param kwargs: the kwarg values
    :return: a static function, independent of any globals for generation at runtime
    """

    def func(dset_path: Path) -> None:
        create_func(dset_path=dset_path, **kwargs)

    return func


def get_dset_for_task(task_name: str) -> str:
    """
    Appropriate way to receive the dataset from a task.

    :param task_name: name of the task
    :return: name of the dataset
    """
    if task_name not in _TASK_TO_DSET:
        raise KeyError(f"Task {task_name} has no registered link to a dataset.")
    return _TASK_TO_DSET[task_name]


def get_task_creator(task_name: str) -> Callable[[Path], None]:
    """
    Appropriate way to receive the creator for a task.

    :param task_name: name of the task
    :return: task creator function
    """
    if task_name not in _TASKCREATORS:
        raise KeyError(f"Task {task_name} has no registered link to a task creator.")
    return _TASKCREATORS[task_name]


def get_dset_creator(dset_name: str) -> Callable[[], Path]:
    """
    Appropriate way to receive the creator for a dataset

    :param dset_name: dataset name
    :return: dataset create function
    """
    if dset_name not in _DATASET_CREATORS:
        raise KeyError(f"Dataset {dset_name} has no registered link to a dataset creator.")
    return _DATASET_CREATORS[dset_name]
