# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
"""
The "mml.interactive" module contains helpers for using mml within interactive sessions, such as the REPL or a jupyter
notebook.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

from mml.core.scripts.utils import load_env, load_mml_plugins
from mml.core.visualization.logo import show_logo

_MML_INTERACTIVE_INITIALIZED = False


def init(env_path: Optional[Path] = None):
    """
    The init function loads environment variables and mml plugins. It is recommended as first function call after
    imports within a jupyter notebook or any other interactive session to plan, process or analyze any mml experiments.

    :param Optional[Path] env_path: as jupyter sometimes struggles to load `MML_ENV_PATH` it may be provided here
    :return:
    """
    global _MML_INTERACTIVE_INITIALIZED
    if not _MML_INTERACTIVE_INITIALIZED:
        # export MML_ENV_PATH if provided
        if env_path:
            if not env_path.exists():
                raise ValueError("Provided env_path does not exist, please provide existing mml.env path!")
            else:
                os.environ["MML_ENV_PATH"] = str(env_path)
        else:
            if "MML_ENV_PATH" not in os.environ:
                warnings.warn(
                    'Did not provide a "env_path", neither found set "MML_ENV_PATH" variable, '
                    'you might need to provide "env_path" to "init" in order to use "mml" '
                    "interactively in a jupyter/ipython setting."
                )
        # try to load everything
        load_env()
        load_mml_plugins()
        show_logo()
        _MML_INTERACTIVE_INITIALIZED = True
        print("Interactive MML API initialized.")
    else:
        print("MML API already initialized.")


def _check_init() -> None:
    """
    This function is intended to warn users if they approach functionality that requires initialization, but they missed
    to do so.

    :raises: RuntimeError - in case no initialization took place
    """
    # to capture the case that mml.interactive is used with already loaded env (e.g. in tests) we check for one entry
    if not _MML_INTERACTIVE_INITIALIZED and not os.getenv("MML_DATA_PATH"):
        raise RuntimeError("To use mml.interactive you need to call mml.interactive.init first.")


from mml.interactive.loading import (  # noqa: E402
    default_file_manager,
    get_task_structs,
    load_project_models,
    merge_project_models,
)
from mml.interactive.planning import (  # noqa: E402
    AllTasksInfos,
    DefaultRequirements,
    EmbeddedJobRunner,
    JobPrefixRequirements,
    JobRunner,
    MMLJobDescription,
    SubprocessJobRunner,
    get_task_infos,
    write_out_commands,
)

__all__ = [
    "load_project_models",
    "AllTasksInfos",
    "DefaultRequirements",
    "JobPrefixRequirements",
    "get_task_infos",
    "MMLJobDescription",
    "init",
    "write_out_commands",
    "merge_project_models",
    "default_file_manager",
    "get_task_structs",
    "JobRunner",
    "EmbeddedJobRunner",
    "SubprocessJobRunner",
]
