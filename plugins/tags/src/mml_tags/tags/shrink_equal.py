# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2025 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from collections import Counter

import numpy as np
import numpy.random

from itertools import chain

from mml.core.data_loading.task_attributes import Modality, TaskType, DataSplit
from mml.core.data_preparation.task_creator import TaskCreator, implements_action
from mml.core.data_preparation.utils import TaskCreatorActions

logger = logging.getLogger(__name__)



@implements_action(TaskCreatorActions.SET_STATS)
def shrink_equal(self: TaskCreator, val_string: str, seed: str = "42") -> None:
    """
    Alternative version of sub-setting a task. Will pick val_string samples from each class equally and
    re-assign them to the training folds from scratch. This means that all folds will be balanced. Does not change
    the test set.

    :param val_string: int within [1, min(num_samples_per_class)]
    :return: None, influences current_meta attribute
    """
    rng = numpy.random.default_rng(int(seed))
    if self.current_meta.task_type != TaskType.CLASSIFICATION:
        raise NotImplementedError(f"shrink_train is not implemented for task type {self.current_meta.task_type}")
    samples_per_class = int(val_string)
    if samples_per_class < 1:
        raise ValueError(f"For equal shrink the value must be at least 1, but was {samples_per_class}.")
    logger.info(
        f"Shrinking training samples of task {self.current_meta.name} to {samples_per_class} samples per class. "
        f"Seed is: {seed}."
    )
    self.protocol(f"Balancing training data to {samples_per_class} samples per class with seed {seed}.")
    for cls, occ in self.current_meta.class_occ.items():
        if occ < samples_per_class:
            raise ValueError(f"Class {cls} has only {occ} samples, but {samples_per_class} were requested.")
    selected_samples = set()
    for cls in self.current_meta.idx_to_class.values():
        cls_samples = [ s_id for s_id in chain(*self.current_meta.train_folds)
            if self.current_meta.idx_to_class[self.current_meta.train_samples[s_id][Modality.CLASS.value]] == cls
        ]
        # import IPython; IPython.embed()
        selected_samples.update(rng.choice(a=cls_samples, size=samples_per_class, replace=False).tolist())

    # run self.split_folds
    self.data = {
        DataSplit.FULL_TRAIN: {s_id: self.current_meta.train_samples[s_id] for s_id in selected_samples},
        DataSplit.TEST: self.current_meta.test_samples,
    }
    old_folds_n = len(self.current_meta.train_folds)
    self.split_folds(n_folds=old_folds_n, ensure_balancing=True, ignore_state=True)

    # update class occurrences
    self.current_meta.class_occ = Counter(
        [self.current_meta.idx_to_class[self.current_meta.train_samples[s_id][Modality.CLASS]] for s_id in selected_samples]
    )
