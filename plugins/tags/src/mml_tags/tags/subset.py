# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from collections import Counter

import numpy as np

from mml.core.data_loading.task_attributes import Modality, TaskType
from mml.core.data_preparation.task_creator import TaskCreator, implements_action
from mml.core.data_preparation.utils import TaskCreatorActions

logger = logging.getLogger(__name__)


@implements_action(TaskCreatorActions.SET_FOLDING)
def subset_task(self: TaskCreator, val_string: str) -> None:
    """
    Simple version of subsetting a task. Every fold will be deterministically shrunk to roughly val_string
    * 100 percent of samples. This process will not take class balance into consideration.

    :param val_string: string of a float within (0, 1), replace comma by underscore (e.g. 0_01 for 1%)
    :return: False, since by sub-setting heavy stats shifts may occur
    """
    percent = float(val_string.replace("_", ".")) * 100
    assert 0 < percent < 100, f"was given tag subset with {val_string=}, must be in range (0, 1)"
    logger.info(f"Subsetting task {self.current_meta.name} to {percent}% of training data (ensuring folds).")
    self.protocol(f"Subset with value {val_string}")
    # set random seed to ensure reproducibility
    np.random.seed(42)
    for fold_ix, fold in enumerate(self.current_meta.train_folds):
        # version 1
        size = int(percent * len(fold) / 100)
        if size == 0:
            msg = f"Subsetting to {percent}% would result in 0 samples within fold {fold_ix}"
            logger.critical(msg)
            raise RuntimeError(msg)
        new_fold = np.random.choice(a=fold, size=size, replace=False).tolist()
        assert abs((len(new_fold) * 100 / len(fold)) - percent) <= 1, "subsetting was not successful"
        logger.debug(
            f"Shrunk fold {fold_ix} from {len(fold)} to {len(new_fold)} equalling roughly "
            f"{len(new_fold) * 100 / len(fold):.2f}%."
        )
        # Remove unused samples
        unused = [sample_id for sample_id in self.current_meta.train_folds[fold_ix] if sample_id not in set(new_fold)]
        for sample_id in unused:
            self.current_meta.train_samples.pop(sample_id)
        # finally set the fold
        self.current_meta.train_folds[fold_ix] = new_fold
    # re-compute class occurrences
    if self.current_meta.task_type == TaskType.CLASSIFICATION:
        self.current_meta.class_occ = Counter(
            [
                self.current_meta.idx_to_class[sample[Modality.CLASS]]
                for sample in self.current_meta.train_samples.values()
            ]
        )
        # we might have lost some classes due to subsetting
        for cls in self.current_meta.idx_to_class.values():
            if cls not in self.current_meta.class_occ:
                self.current_meta.class_occ[cls] = 0
