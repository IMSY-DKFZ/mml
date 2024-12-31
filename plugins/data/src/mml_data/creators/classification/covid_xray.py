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
@misc{HematejaAluru2021,
  title = {Covid19 X-Ray classification dataset on kaggle},
  howpublished = {\\url{https://www.kaggle.com/ahemateja19bec1025/covid-xray-dataset}},
  note = {Accessed: 2022-01-13}
}
"""

dset_name = "covid_xray"
task_name = "covid_xray_classification"


@register_dsetcreator(dset_name=dset_name)
def create_dset() -> Path:
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.kaggle_download(dataset="ahemateja19bec1025/covid-xray-dataset", data_kind=DataKind.TRAINING_DATA)
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="This is the dataset that Aluru V N M Hemateja gathered from different sources for Covid "
        "19 Detection. There are around 3000 images in those two files and there is a handling "
        'code in the Notebook named "Official" at the end. Please use the dataset wisely and only '
        "for good purposes.",
        ref=REFERENCE,
        url="https://www.kaggle.com/ahemateja19bec1025/covid-xray-dataset",
        instr="download via kaggle (https://www.kaggle.com/ahemateja19bec1025/covid-xray-dataset)",
        lic=License.CC_0_1_0,
        release="2021 version 2",
        keywords=[Keyword.MEDICAL, Keyword.X_RAY, Keyword.CHEST, Keyword.TISSUE_PATHOLOGY],
    )
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "DATA" / "DATA", classes=None
    )
    class_name_remapper = {"0": "Without Covid", "1": "With Covid"}
    idx_to_class = {k: class_name_remapper[v] for k, v in idx_to_class.items()}
    task.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
