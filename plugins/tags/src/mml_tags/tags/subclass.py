# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from typing import List

from mml.core.data_preparation.task_creator import TaskCreator, implements_action
from mml.core.data_preparation.utils import TaskCreatorActions


@implements_action(TaskCreatorActions.NONE)
def subclass_task(self: TaskCreator, *classes_to_keep: List[str]) -> None:
    """
    Restricts the task to the classes given. This may be either ...
    :param classes_to_keep:
    :return:
    """
    # TODO implement functionality
    # requires remapping in idx_to_class!
    # requires new class_occ
    raise NotImplementedError()
