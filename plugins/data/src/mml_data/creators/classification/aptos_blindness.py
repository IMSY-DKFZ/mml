# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import pandas as pd

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@misc{
url = {https://www.kaggle.com/c/aptos2019-blindness-detection/data},
title = {APTOS 2019 Blindness Detection challenge}
}
"""

dset_name = "aptos_blindness"


@register_dsetcreator(dset_name=dset_name)
def create_dset() -> Path:
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.kaggle_download(competition="aptos2019-blindness-detection", data_kind=DataKind.TRAINING_DATA)
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="aptos19_blindness_detection", dset_name="aptos_blindness")
def create_aptos(dset_path: Path):
    diabetic_retinopathy = TaskCreator(
        dset_path=dset_path,
        name="aptos19_blindness_detection",
        task_type=TaskType.CLASSIFICATION,
        desc="A large set of retina images taken using fundus photography under a "
        "variety of imaging conditions. Each image is rated for the severity of "
        "diabetic retinopathy on a scale of 0 to 4",
        ref=REFERENCE,
        url="https://www.kaggle.com/c/aptos2019-blindness-detection/data",
        instr="download via https://www.kaggle.com/c/aptos2019-blindness-detection/data",
        lic=License.UNKNOWN,
        release="2018",
        keywords=[Keyword.MEDICAL, Keyword.FUNDUS_PHOTOGRAPHY, Keyword.EYE],
    )

    idx_to_class = {
        0: "No Diabetic Retinopathy",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative Diabetic Retinopathy",
    }
    train_iterator = []
    train_matrix = pd.read_csv(dset_path / DataKind.TRAINING_DATA / "train.csv")
    for _, row in train_matrix.iterrows():
        train_iterator.append(
            {
                Modality.SAMPLE_ID: row["id_code"],
                Modality.IMAGE: dset_path / DataKind.TRAINING_DATA / "train_images" / f"{row['id_code']}.png",
                Modality.CLASS: row["diagnosis"],
            }
        )
    unlabled_iterator = []
    unlabeled_matrix = pd.read_csv(dset_path / DataKind.TRAINING_DATA / "test.csv")
    for _, row in unlabeled_matrix.iterrows():
        unlabled_iterator.append(
            {
                Modality.SAMPLE_ID: row["id_code"],
                Modality.IMAGE: dset_path / DataKind.TRAINING_DATA / "test_images" / f"{row['id_code']}.png",
            }
        )
    diabetic_retinopathy.find_data(
        train_iterator=train_iterator, unlabeled_iterator=unlabled_iterator, idx_to_class=idx_to_class
    )
    diabetic_retinopathy.split_folds(n_folds=5, ensure_balancing=True)
    diabetic_retinopathy.infer_stats()
    diabetic_retinopathy.push_and_test()
