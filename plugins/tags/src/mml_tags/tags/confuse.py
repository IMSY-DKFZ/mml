# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from itertools import chain

import numpy as np

from mml.core.data_loading.task_attributes import Modality, TaskType
from mml.core.data_preparation.task_creator import TaskCreator, implements_action
from mml.core.data_preparation.utils import TaskCreatorActions

logger = logging.getLogger(__name__)


@implements_action(TaskCreatorActions.NONE)
def confuse_task(self: TaskCreator, val_string: str) -> None:
    """
    Simple version of making a task confusing by setting a fraction of labels as random.

    :param val_string: string of a float within (0, 1), replace comma by underscore (e.g. 0_01 for 1%)
    :return: None
    """
    if self.current_meta.task_type != TaskType.CLASSIFICATION:
        raise RuntimeError("confuse tag only valid for classification task")
    percent = float(val_string.replace("_", ".")) * 100
    assert 0 < percent < 100, f"was given tag confuse with {val_string}, must be in range (0, 1)"
    logger.info(f"Confusing {percent}% of training data of task {self.current_meta.name}.")
    self.protocol(f"Confused with value {val_string}")
    # ensure reproducibility
    np.random.seed(42)
    all_ids = list(chain(*self.current_meta.train_folds))
    confuse_ids = np.random.choice(a=all_ids, size=int(percent * len(all_ids) / 100), replace=False).tolist()
    assert abs((len(confuse_ids) * 100 / len(all_ids)) - percent) <= 1, "confusing was not successful"
    num_classes = len(set(self.current_meta.idx_to_class.values()))
    for confuse_id in confuse_ids:
        new_val = np.random.randint(num_classes)
        old_val = self.current_meta.train_samples[confuse_id][Modality.CLASS.value]
        self.current_meta.train_samples[confuse_id][Modality.CLASS.value] = new_val
        self.current_meta.class_occ[self.current_meta.idx_to_class[old_val]] -= 1
        self.current_meta.class_occ[self.current_meta.idx_to_class[new_val]] += 1
    logger.debug(
        f"Confused {len(confuse_ids)} out of {len(all_ids)} training_tuples equalling roughly "
        f"{len(confuse_ids) * 100 / len(all_ids)}%."
    )
