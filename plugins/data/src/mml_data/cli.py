# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from prettytable.colortable import ColorTable, Themes

import mml.core.data_preparation.fake_task
import mml.core.data_preparation.task_creator as task_creator_module
from mml.core.data_loading.task_attributes import Keyword, TaskType
from mml.core.data_preparation.registry import _DATASET_CREATORS, _TASK_TO_DSET, _TASKCREATORS
from mml.core.scripts.utils import TAG_SEP, load_env, load_mml_plugins
from mml.interactive import default_file_manager


def get_all_avail_meta_data() -> dict:
    """
    Reads in meta-data of available task creators without actually installing those tasks.
    """
    # here we will store coded meta
    coded_meta = {}

    # this dummy creator will be mocked and used instead of the actual task creator, it stores passed kwargs
    class DummyTaskCreator:
        def __init__(self, **kwargs):
            nonlocal coded_meta
            kwargs.pop("dset_path", None)
            coded_meta[kwargs["name"]] = kwargs
            raise RuntimeError("Intended interruption.")

    # do the mocking
    creator_backup = task_creator_module.TaskCreator
    task_creator_module.TaskCreator = DummyTaskCreator

    # import all the task creators and iterate
    load_mml_plugins()
    dummy_dset_path = Path("/dummy")
    for task_alias, task_creator in _TASKCREATORS.items():
        # we cannot prevent to import this creator before mocking, on the other hand it is not required anyway
        if task_alias == mml.core.data_preparation.fake_task.task_name:
            continue
        try:
            # this reads in all kwargs into coded_meta variable
            task_creator(dummy_dset_path)
        except RuntimeError:
            pass

    # undo mocking
    task_creator_module.TaskCreator = creator_backup

    return coded_meta


def get_all_installed_tasks() -> Dict[str, List[str]]:
    """
    Gathers all installed tasks and clusters by variants thereof.
    """
    task_variants = {}
    load_env()
    with default_file_manager() as fm:
        # all installed tasks
        all_tasks = list(fm.task_index.keys())
        base_tasks = [t for t in all_tasks if (" " not in t and TAG_SEP not in t)]
        for task in base_tasks:
            task_variants[task] = [
                t[len(task) + 1 :] for t in all_tasks if (t.startswith(task + " ") or t.startswith(t + TAG_SEP))
            ]
    return task_variants


def list_tasks(
    task_types: Optional[List[TaskType]] = None,
    keywords: Optional[List[Keyword]] = None,
    installed: bool = False,
    variants: bool = False,
):
    """
    Lists all available tasks with filters applied.

    :param Optional[List[TaskType]] task_types: only show tasks of any given type
    :param Optional[List[Keyword]] keywords: only show tasks that are marked with all provided keywords
    :param bool installed: only show tasks that are installed (via create mode)
    :param bool variants: instead of the default columns, show installed variants of tasks
    :return: prints a table with columns ['name', 'dataset', 'type', 'installed', 'license'] and filtered
        available tasks, if variants is True, the columns will be ['name', 'installed', 'variants']
    """
    # get task via definition
    coded_meta = get_all_avail_meta_data()
    # get installed tasks
    task_variants = get_all_installed_tasks()
    # will be displayed to the terminal
    table = ColorTable(theme=Themes.OCEAN)
    # counts variants of tasks during the loop
    var_counter = 0
    # define table columns
    if variants:
        table.field_names = ["name", "installed", "variants"]
    else:
        table.field_names = ["name", "dataset", "type", "installed", "license"]
    # loop over task definitions
    for task, meta in coded_meta.items():
        # omit non-installed tasks if required
        if installed and task not in task_variants:
            continue
        # filter by task type
        if task_types and meta["task_type"] not in task_types:
            continue
        # filter by keywords
        if keywords and not all([t in meta["keywords"] for t in keywords]):
            continue
        # count variants if installed
        if task in task_variants:
            var_counter += len(task_variants[task])
        # add information to table
        if variants:
            table.add_row([task, task in task_variants, task_variants[task] if task in task_variants else "-"])
        else:
            table.add_row([task, _TASK_TO_DSET[task], meta["task_type"].value, task in task_variants, meta["lic"].name])
    # print table and summary
    print(table)
    print(
        f"Overall {len(_TASKCREATORS)} raw tasks available, from {len(_DATASET_CREATORS)} datasets. Filter matches "
        f"{len(table.rows)} raw tasks with additional {var_counter} variants derived thereof."
    )


def cli():
    """
    Parse arguments to mml-data and call the list_tasks function to print task information.
    """
    parser = argparse.ArgumentParser(
        prog="mml-data", description="Print available mml tasks with certain filters applied."
    )
    parser.add_argument("-i", "--installed", action="store_true", help="limit to installed tasks only")
    parser.add_argument(
        "-k",
        "--keyword",
        action="extend",
        nargs="*",
        type=Keyword,
        help=f"limit to a mandatory combination of keywords, options: {Keyword.list()}",
    )
    parser.add_argument(
        "-t",
        "--type",
        action="extend",
        nargs="*",
        type=TaskType,
        help=f"limit to a variety of task types, options: {TaskType.list()}",
    )
    parser.add_argument(
        "-v", "--variants", action="store_true", help="display all variants of the given task that are installed"
    )
    args = parser.parse_args()
    list_tasks(task_types=args.type, keywords=args.keyword, installed=args.installed, variants=args.variants)
