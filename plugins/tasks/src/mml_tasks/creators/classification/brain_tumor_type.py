# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@misc{cheng_2017, 
title={brain tumor dataset}, 
url={https://figshare.com/articles/dataset/brain_tumor_dataset/1512427/5}, 
DOI={10.6084/m9.figshare.1512427.v5}, 
abstractNote={This brain tumor dataset contains 3064 T1-weighted contrast-inhanced images with three kinds of brain 
tumor. Detailed information of the dataset can be found in readme file.}, 
publisher={figshare}, 
author={Cheng, Jun}, 
year={2017},
month={Apr} 
} 
"""  # noqa W291

classes = ["1", "2", "3"]
dset_name = "brain_tumor_type"


@register_dsetcreator(dset_name=dset_name)
def create_dset() -> Path:
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.kaggle_download(dataset="denizkavi1/brain-tumor", data_kind=DataKind.TRAINING_DATA)
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="brain_tumor_type_classification", dset_name="brain_tumor_type")
def create_bt_classification(dset_path: Path):
    bt_classification = TaskCreator(
        dset_path=dset_path,
        name="brain_tumor_type_classification",
        task_type=TaskType.CLASSIFICATION,
        desc="The dataset containing  samples of meningioma(1), glioma(2), pituitary tumor(3) brain tumor types",
        ref=REFERENCE,
        url="https://www.kaggle.com/denizkavi1/brain-tumor",
        instr="download via https://www.kaggle.com/denizkavi1/brain-tumor",
        lic=License.CC_BY_4_0,
        release="2017",
        keywords=[Keyword.MEDICAL, Keyword.MRI_SCAN, Keyword.BRAIN, Keyword.TISSUE_PATHOLOGY],
    )
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=classes
    )
    bt_classification.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    bt_classification.split_folds(n_folds=5, ensure_balancing=True)
    bt_classification.infer_stats()
    bt_classification.push_and_test()
