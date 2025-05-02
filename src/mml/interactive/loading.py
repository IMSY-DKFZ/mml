# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import contextlib
import os
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Union

import omegaconf
from omegaconf import DictConfig

from mml.core.data_loading.file_manager import MMLFileManager, ReuseConfig
from mml.core.data_loading.task_struct import TaskStruct, TaskStructFactory
from mml.core.scripts.model_storage import ModelStorage
from mml.interactive import _check_init


def load_project_models(project: str) -> Dict[str, List[ModelStorage]]:
    """
    Loading utility to get all models of a given project.

    :param str project: name of the project, what has been inserted with 'proj=...'
    :return: dict with task name keys and a list of all corresponding ModelStorages
    """
    _check_init()
    proj_path = Path(f"{os.getenv('MML_RESULTS_PATH')}/{project}")
    tmp_log_path = proj_path / "tmp_log"
    tmp_log_path.mkdir(exist_ok=True)
    r_conf = ReuseConfig(models=f"{project}")
    fm = MMLFileManager(
        data_path=Path(f"{os.getenv('MML_DATA_PATH')}"), proj_path=proj_path, log_path=tmp_log_path, reuse_cfg=r_conf
    )
    all_reusables = fm.reusables
    MMLFileManager.clear_instance()
    return {task_name: task_reusables["models"] for task_name, task_reusables in all_reusables.items()}


def merge_project_models(project_models_list: Iterable[Dict[str, List[ModelStorage]]]) -> Dict[str, List[ModelStorage]]:
    """
    Merges models loaded from multiple projects.

    :param project_models_list: list of dicts, as returned by multiple calls from func::load_project_models
    :return: merged list, as if all models were trained in one single project
    """
    out = {}
    for project_models in project_models_list:
        for task, model_list in project_models.items():
            if task in out:
                out[task].extend(model_list)
            else:
                out[task] = model_list
    return out


@contextlib.contextmanager
def default_file_manager(
    reuse_config: Optional[Union[DictConfig, ReuseConfig]] = None,
) -> Generator[MMLFileManager, None, None]:
    """
    Convenience method to get a MMLFileManager instance. To be used in a with statement:

    .. code-block:: python

        with default_file_manager() as fm:
            fm.do_something (e.g. extract information)
            ...

    continue code with extracted information (without fm)

    :return:
    """
    _check_init()
    proj_path = Path(f"{os.getenv('MML_RESULTS_PATH')}/default")
    proj_path.mkdir(exist_ok=True)
    tmp_log_path = proj_path / "tmp_log"
    tmp_log_path.mkdir(exist_ok=True)
    if reuse_config is None:
        reuse_config = ReuseConfig()
    try:
        yield MMLFileManager(
            data_path=Path(f"{os.getenv('MML_DATA_PATH')}"),
            proj_path=proj_path,
            log_path=tmp_log_path,
            reuse_cfg=reuse_config,
        )
    finally:
        MMLFileManager.clear_instance()


def get_task_structs(tasks: Union[str, Sequence[str]], preprocessing: str = "default") -> List[TaskStruct]:
    """
    Create a task struct on the fly.

    :param str tasks: task name or sequence of task names
    :param str preprocessing: the preprocessing id of the task (default: 'default')
    :return: the corresponding task struct
    :rtype: TaskStruct
    """
    _check_init()
    cfg = omegaconf.OmegaConf.create({"preprocessing": {"id": preprocessing}})
    if isinstance(tasks, str):
        tasks = [tasks]
    structs = []
    with default_file_manager():
        factory = TaskStructFactory(cfg=cfg)
        for task in tasks:
            structs.append(factory.create_task_struct(name=task, return_ref=True))
    return structs
