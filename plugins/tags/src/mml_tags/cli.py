# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from mml_tags import __version__

from mml.core.data_preparation.task_creator import TASK_CREATOR_TAG_MAP, TaskCreator
from mml.core.scripts.utils import ARG_SEP, TAG_SEP, load_mml_plugins


def list_available_tags():
    load_mml_plugins()
    print(f"mml_tags plugin, version {__version__}")
    print(f"Available tags (total {len(TASK_CREATOR_TAG_MAP)}):")
    for tag_id, tag_fct in TASK_CREATOR_TAG_MAP.items():
        print(f"  * {tag_id}\n{getattr(TaskCreator, tag_fct).__doc__}\n")
    print(
        f"\nApply tags with with tag seperator {TAG_SEP} and argument seperator {ARG_SEP}. See configs/tasks for "
        f"more details."
    )
