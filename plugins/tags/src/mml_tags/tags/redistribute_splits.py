# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from collections import Counter

from mml.core.data_loading.task_attributes import DataSplit, Modality, TaskType
from mml.core.data_preparation.task_creator import TaskCreator, implements_action
from mml.core.data_preparation.utils import TaskCreatorActions

logger = logging.getLogger(__name__)


@implements_action(TaskCreatorActions.SET_STATS)
def redistribute_splits(self: TaskCreator, *new_distribution: str) -> None:
    """
    Allows for redistribution of the train-validation-test splits of a task.

    Receives either 1, 2 or 3 arguments that are interpreted as train(:val) or train:val:test percentages.
    Values must be floats (given as e.g. '0_25'), in case of 3 arguments they must sum 1, otherwise raises a
    :exc:ValueError.

    If one value is provided the test split is left untouched, all train tuples are redistributed into folds,
    whereas conserving the number of folds, but fold 1 ... fold n receive together the fraction of samples given
    in the argument while fold 0 (which is interpreted as validation fold) receives the rest. Be aware that for
    classification tasks balanced classes are tried to be ensured.

    If two values are provided the original test split is dropped, if the two values do not sum up to one, the
    remaining fraction is (randomly) selected as new test set (from the original training split). For the rest the
    method behaves as in the case with one argument - first value fraction will be distributed across folds 1, ...,
    n and the second value fraction become fold 0.

    For the variant with three arguments ALL original samples (train and test) are put into a pool and test split
    is resampled from this pool (based on the third value fraction). Afterward with the remaining samples the
    procedure is identical to the other argument variants, except as the folds are built from the remaining pool.

    :param new_distribution: either one, two or three strings that represent floats for distributing samples
    :return: None
    """
    # log
    logger.info(f"Redistributing data of task {self.current_meta.name} with {new_distribution=}.")
    self.protocol(f"Redistributed with value(s) {new_distribution}.")
    # check args
    if len(new_distribution) < 1 or len(new_distribution) > 3:
        raise ValueError("Redistribute tag must be provided with either 1, 2 or 3 arguments (split by ' ')")
    new_distribution = [float(elem.replace("_", ".")) for elem in new_distribution]
    drop_original_test = len(new_distribution) == 2
    if drop_original_test:
        test_frac = 1 - sum(new_distribution)
        if test_frac < 0 or test_frac >= 1:
            raise ValueError("Redistribute tag with two arguments must sum within (0, 1].")
        new_distribution.append(test_frac)
    modify_test = len(new_distribution) == 3
    if modify_test and sum(new_distribution) != 1:
        raise ValueError("Redistribute tag with three arguments must sum up to one.")
    if not modify_test and (sum(new_distribution) <= 0 or sum(new_distribution) >= 1):
        raise ValueError("Redistribute tag with one argument must be within (0, 1) excluding borders.")
    if modify_test and self.current_meta.task_type != TaskType.CLASSIFICATION:
        raise RuntimeError(
            "Since class occurrences are changed if train/test set changes, this is currently "
            "only available for classification tasks."
        )
    if modify_test and self.fm.preprocessed_data in self.dset_path.parents:
        raise RuntimeError(
            "Since redistribute tag modifies the test data, this can only be run on a raw (none "
            "preprocessed) version of a task. If this task already has been preprocessed with "
            "some preprocessing id X, consider creating the task with some dummy preprocessing id "
            "e.g. mml info (task settings as before) preprocessing=example and run your "
            "original command afterwards."
        )
    # set self.data['train'] and self.data['test'], recalculate class_occ if necessary
    old_folds_n = len(self.current_meta.train_folds)
    if modify_test:
        all_ids = list(self.current_meta.train_samples.keys())
        all_samples = self.current_meta.train_samples
        if not drop_original_test:
            all_ids.extend(list(self.current_meta.test_samples.keys()))
            all_samples.update(self.current_meta.test_samples)
        # use split folds mechanism to extract the new test split balanced
        self.data = {DataSplit.FULL_TRAIN: all_samples}
        self.split_folds(n_folds=2, ensure_balancing=True, fold_0_fraction=new_distribution[2], ignore_state=True)
        train_ids = self.current_meta.train_folds[1]
        test_ids = self.current_meta.train_folds[0]
        self.data = {
            DataSplit.FULL_TRAIN: {s_id: all_samples[s_id] for s_id in train_ids},
            DataSplit.TEST: {s_id: all_samples[s_id] for s_id in test_ids},
        }
        # update class occurrences
        self.current_meta.class_occ = Counter(
            [self.current_meta.idx_to_class[all_samples[s_id][Modality.CLASS]] for s_id in train_ids]
        )
    else:
        self.data = {
            DataSplit.FULL_TRAIN: self.current_meta.train_samples,
            DataSplit.TEST: self.current_meta.test_samplestest_tuples,
        }
    # calculate validation split and run self.split_folds
    if modify_test:
        fraction = new_distribution[1] / (new_distribution[0] + new_distribution[1])
    else:
        fraction = 1 - new_distribution[0]
    self.split_folds(n_folds=old_folds_n, ensure_balancing=True, fold_0_fraction=fraction, ignore_state=True)
