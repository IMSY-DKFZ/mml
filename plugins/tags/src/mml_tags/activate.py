# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from mml_tags.tags.confuse import confuse_task
from mml_tags.tags.redistribute_splits import redistribute_splits
from mml_tags.tags.shrink_train import shrink_train
from mml_tags.tags.subclass import subclass_task
from mml_tags.tags.subset import subset_task

from mml.core.data_preparation.task_creator import TASK_CREATOR_TAG_MAP, TaskCreator

TaskCreator.shrink_train = shrink_train
TaskCreator.confuse_task = confuse_task
TaskCreator.redistribute_splits = redistribute_splits
TaskCreator.subclass_task = subclass_task
TaskCreator.subset_task = subset_task

TASK_CREATOR_TAG_MAP["shrink_train"] = "shrink_train"
TASK_CREATOR_TAG_MAP["redistribute_splits"] = "redistribute_splits"
TASK_CREATOR_TAG_MAP["confuse"] = "confuse_task"
TASK_CREATOR_TAG_MAP["subset"] = "subset_task"
TASK_CREATOR_TAG_MAP["subclass"] = "subclass_task"
