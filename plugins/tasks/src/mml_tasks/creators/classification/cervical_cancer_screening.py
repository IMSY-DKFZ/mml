# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path
from typing import List

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

logger = logging.getLogger(__name__)

REFERENCE = """
@misc{
title={intel-mobileodt-cervical-cancer-screening}, 
url={https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data}, 
publisher={Kaggle}
}
"""  # noqa W291

dset_name = "cervical_screening"
task_name = "cervix_type_classification"
classes = ["Type_1", "Type_2", "Type_3"]


@register_dsetcreator(dset_name=dset_name)
def create_dset() -> Path:
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.kaggle_download(
        competition="intel-mobileodt-cervical-cancer-screening", data_kind=DataKind.TRAINING_DATA
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


def get_sequences(dset_path: Path) -> List[Path]:
    """
    Returns all sequence folder paths.

    :param dset_path:
    :return:
    """
    releases = ["train/train", "additional_Type_1_v2", "additional_Type_2_v2", "additional_Type_3_v2"]
    folder_paths = []
    for rel in releases:
        p = dset_path / DataKind.TRAINING_DATA / rel
        folder_paths.append(p)
    return folder_paths


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="The dataset contains three different types of cervix which is helpful in identifying the "
        "transformation zones",
        ref=REFERENCE,
        url="https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data",
        instr="download via kaggle (https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data)",
        lic=License.UNKNOWN,
        release="Unknown",
        keywords=[Keyword.MEDICAL, Keyword.GYNECOLOGY],
    )
    train_iterator = []
    for seq in get_sequences(dset_path):
        for class_folder in seq.iterdir():
            for img_path in class_folder.iterdir():
                train_iterator.append(
                    {
                        Modality.SAMPLE_ID: seq.name + img_path.stem,
                        Modality.IMAGE: img_path,
                        Modality.CLASS: classes.index(class_folder.name),
                    }
                )
    idx_to_class = {classes.index(cl): cl for cl in classes}
    task.find_data(train_iterator=train_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
