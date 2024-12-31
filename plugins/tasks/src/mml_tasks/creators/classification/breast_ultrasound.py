# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

logger = logging.getLogger(__name__)

REFERENCE = """
@article{article,
author = {Al-Dhabyani, Walid and Gomaa, Mohammed and Khaled, Hussien and Fahmy, Aly},
year = {2019},
month = {11},
pages = {104863},
title = {Dataset of Breast Ultrasound Images},
volume = {28},
journal = {Data in Brief},
doi = {10.1016/j.dib.2019.104863}
}
"""

dset_name = "breast_ultrasound"
task_name = "breast_cancer_classification_v2"
classes = ["benign", "malignant", "normal"]


@register_dsetcreator(dset_name=dset_name)
def create_dset() -> Path:
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.kaggle_download(
        dataset="aryashah2k/breast-ultrasound-images-dataset", data_kind=DataKind.TRAINING_DATA
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images",
        ref=REFERENCE,
        url="https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset",
        instr="download via kaggle (https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)",
        lic=License.CC_0_1_0,
        release="2019",
        keywords=[Keyword.MEDICAL, Keyword.BREAST, Keyword.ULTRASOUND],
    )
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "Dataset_BUSI_with_GT", classes=classes
    )
    cleaned_iterator = [d for d in data_iterator if "mask" not in d[Modality.SAMPLE_ID]]
    logger.info(f"Removed {len(data_iterator) - len(cleaned_iterator)} mask items.")
    task.find_data(train_iterator=cleaned_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
