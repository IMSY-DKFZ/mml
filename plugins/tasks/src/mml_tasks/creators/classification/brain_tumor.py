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

# TODO refactor as automatic kaggle download
REFERENCE = """
@misc{jakesh bohaju_2020, 
title={Brain Tumor}, 
url={https://www.kaggle.com/dsv/1370629}, 
DOI={10.34740/KAGGLE/DSV/1370629}, 
publisher={Kaggle}, 
author={Jakesh Bohaju}, 
year={2020} 
}
"""  # noqa W291

dset_name = "brain_tumor"


@register_dsetcreator(dset_name=dset_name)
def create_dset() -> Path:
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.kaggle_download(dataset="jakeshbohaju/brain-tumor", data_kind=DataKind.TRAINING_DATA)
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="brain_tumor_classification", dset_name="brain_tumor")
def create_bt_classification(dset_path: Path):
    bt_classification = TaskCreator(
        dset_path=dset_path,
        name="brain_tumor_classification",
        task_type=TaskType.CLASSIFICATION,
        desc="Brain Tumor dataset containing scans with brain tumor and no brain tumor",
        ref=REFERENCE,
        url="https://www.kaggle.com/jakeshbohaju/brain-tumor",
        instr="download via https://www.kaggle.com/jakeshbohaju/brain-tumor",
        lic=License.CC_BY_NC_SA_4_0,
        release="2020",
        keywords=[Keyword.MEDICAL, Keyword.MRI_SCAN, Keyword.BRAIN, Keyword.TISSUE_PATHOLOGY],
    )
    data_iterator = []
    idx_to_class = {0: "No Tumor", 1: "Tumor"}
    data_matrix = pd.read_csv(dset_path / DataKind.TRAINING_DATA / "Brain Tumor.csv")
    for _, row in data_matrix.iterrows():
        data_iterator.append(
            {
                Modality.SAMPLE_ID: row["Image"],
                Modality.IMAGE: (
                    dset_path / DataKind.TRAINING_DATA / "Brain Tumor" / "Brain Tumor" / f"{row['Image']}.jpg"
                ),
                Modality.CLASS: row["Class"],
            }
        )
    bt_classification.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    bt_classification.split_folds(n_folds=5, ensure_balancing=True)
    bt_classification.infer_stats()
    bt_classification.push_and_test()
