# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

logger = logging.getLogger(__name__)

REFERENCE = """
@misc{jr2ngb,
title={Cataract dataset},
url={https://www.kaggle.com/datasets/jr2ngb/cataractdataset},
publisher={Kaggle}
}
"""

dset_name = "cataract"
task_name = "eye_condition_classification"
classes = ["1_normal", "2_cataract", "2_glaucoma", "3_retina_disease"]


@register_dsetcreator(dset_name=dset_name)
def create_dset() -> Path:
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.kaggle_download(dataset="jr2ngb/cataractdataset", data_kind=DataKind.TRAINING_DATA)
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="The dataset contains 4 categories of eye condition i.e. normal, cataract, glaucoma, retina disease",
        ref=REFERENCE,
        url="https://www.kaggle.com/datasets/jr2ngb/cataractdataset",
        instr="download via kaggle (https://www.kaggle.com/datasets/jr2ngb/cataractdataset)",
        lic=License.UNKNOWN,
        release="Unknown",
        keywords=[Keyword.MEDICAL, Keyword.EYE, Keyword.CATARACT_SURGERY],
    )
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "dataset", classes=classes
    )
    task.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
