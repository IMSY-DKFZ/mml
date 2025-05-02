# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging

import numpy as np
import numpy.random

from mml.core.data_loading.task_attributes import Modality, TaskType
from mml.core.data_preparation.task_creator import TaskCreator, implements_action
from mml.core.data_preparation.utils import TaskCreatorActions

logger = logging.getLogger(__name__)

MIN_OCC_PER_CLASS = 5  # constant representing the minimum occ per class


@implements_action(TaskCreatorActions.SET_STATS)
def shrink_train(self: TaskCreator, val_string: str, seed: str = "42") -> None:
    """
    Alternative version of sub-setting a task. Every fold but fold 0 (which will be used for evaluation)
    is deterministically shrunk to sum all remaining training samples to total val_string.

    MIN_OCC_PER_CLASS ensures (if sufficient samples of a class are available) a minimum number of samples of a
    class.

    :param val_string: string of an int within [(num_folds - 1)*5*num_classes, (num_folds - 1) * num_samples / num_folds]
    :return: None, influences current_meta attribute
    """
    if self.current_meta.task_type != TaskType.CLASSIFICATION:
        raise NotImplementedError(f"shrink_train is not implemented for task type {self.current_meta.task_type}")
    new_total_class_occ = {val: 0 for val in set(self.current_meta.idx_to_class.values())}
    num_folds = len(self.current_meta.train_folds)
    max_samples = sum([len(fold) for fold in self.current_meta.train_folds[1:]])
    min_samples = (num_folds - 1) * MIN_OCC_PER_CLASS * len(new_total_class_occ)
    remaining = int(val_string)
    assert min_samples <= remaining <= max_samples, (
        f"was given tag subset with {val_string=}, must be in range [{min_samples}, {max_samples}]"
    )
    logger.info(
        f"Shrinking training folds of task {self.current_meta.name} to {remaining} of training "
        f"data (ensuring folds). Seed is: {seed}."
    )
    self.protocol(f"Shrink training with value {val_string} and seed {seed}")
    for fold_ix, fold in enumerate(self.current_meta.train_folds):
        # skip first fold, this will be the validation / test split
        if fold_ix == 0:
            continue
        # set random seed to ensure reproducibility
        rng = numpy.random.default_rng(int(seed) + fold_ix * 42)
        # calculate remaining size for this fold
        size = remaining // (num_folds - 1)
        if remaining % (num_folds - 1) > fold_ix:
            size += 1
        assert 1 <= size <= len(fold), f"Size requirement for fold {fold_ix} not met, {size=} but {len(fold)=}."
        # assert all classes have enough occurrences
        pre_selection = set()
        for cls in new_total_class_occ:
            cls_samples = [
                sample_id
                for sample_id in fold
                if self.current_meta.idx_to_class[self.current_meta.train_samples[sample_id][Modality.CLASS.value]]
                == cls
            ]
            if len(cls_samples) < MIN_OCC_PER_CLASS:
                logger.warning(f"Only {len(cls_samples)} samples for class {cls} (requires {MIN_OCC_PER_CLASS}).")
                pre_selection.update(cls_samples)
            else:
                pre_selection.update(rng.choice(a=cls_samples, size=MIN_OCC_PER_CLASS, replace=False).tolist())
        # now fill the remaining gap
        if size - len(pre_selection) <= 0:
            raise RuntimeError("Too many classes to ensure sufficient samples per class in each fold.")
        non_selected = [sample_id for sample_id in fold if sample_id not in pre_selection]
        post_selection = rng.choice(a=non_selected, size=size - len(pre_selection), replace=False).tolist()
        new_fold = np.random.permutation(list(pre_selection) + post_selection).tolist()
        new_fold_class_occ = {val: 0 for val in set(self.current_meta.idx_to_class.values())}
        for sample_id in new_fold:
            new_fold_class_occ[
                self.current_meta.idx_to_class[self.current_meta.train_samples[sample_id][Modality.CLASS]]
            ] += 1
        if not all([occ >= MIN_OCC_PER_CLASS for occ in new_fold_class_occ.values()]):
            logger.warning(
                f"not sufficient samples to guarantee {MIN_OCC_PER_CLASS} occ per class in "
                f"fold {fold_ix}, resulting occ was {new_fold_class_occ}."
            )
        else:
            # successful shrinkage
            logger.debug(
                f"was able to guarantee {MIN_OCC_PER_CLASS} occ per class in fold {fold_ix}, "
                f"resulting occ was {new_fold_class_occ}."
            )
        logger.debug(
            f"Shrunk fold {fold_ix} from {len(fold)} to {len(new_fold)} equalling roughly "
            f"{len(new_fold) * 100 / len(fold):.2f}%."
        )
        # Remove unused samples
        unused = [sample_id for sample_id in self.current_meta.train_folds[fold_ix] if sample_id not in set(new_fold)]
        for sample_id in unused:
            self.current_meta.train_samples.pop(sample_id)
        # update total class_occ
        for k, v in new_fold_class_occ.items():
            new_total_class_occ[k] += v
        # finally set the fold
        self.current_meta.train_folds[fold_ix] = new_fold
    # set new total class occ, this one ignores the validation split!
    self.current_meta.class_occ = new_total_class_occ
