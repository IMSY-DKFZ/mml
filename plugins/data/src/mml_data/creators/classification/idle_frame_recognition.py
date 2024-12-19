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
@inproceedings{inproceedings,
author = {Ghamsarian, Negin and Amirpour, Hadi and Timmerer, Christian and Taschwer, Mario and Schoeffmann, Klaus},
year = {2020},
month = {10},
pages = {},
title = {Relevance-Based Compression of Cataract Surgery Videos Using Convolutional Neural Networks},
doi = {10.1145/3394171.3413658}
}
"""


@register_dsetcreator(dset_name="idle_action")
def create_frame_recognition():
    dset_creator = DSetCreator(dset_name="idle_action")
    dset_creator.download(
        url="ftp.itec.aau.at/datasets/ovid/CatRelevanceCompression/downloads/Idle_frame_recognition.zip",
        file_name="Idle_frame_recognition.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="idle_action_recognition", dset_name="idle_action")
def create_idle_action_recognition(dset_path: Path):
    frame_recognition = TaskCreator(
        dset_path=dset_path,
        name="idle_action_recognition",
        task_type=TaskType.CLASSIFICATION,
        desc="This dataset contains manual annotations of idle and action frames in "
        "cataract surgery videos for idle-frame-recognition networks",
        ref=REFERENCE,
        url="http://ftp.itec.aau.at/datasets/ovid/CatRelevanceCompression/",
        instr="download Idle_frame_recognition dataset via " "ftp.itec.aau.at/datasets/ovid/CatRelevanceCompression/",
        lic=License.CC_BY_NC_4_0,
        release="2020",
        keywords=[Keyword.MEDICAL, Keyword.CATARACT_SURGERY, Keyword.EYE],
    )
    classes = ["Action", "Idle"]
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "Idle_frame_recognition" / "Train", dup_id_flag=True, classes=classes
    )
    frame_recognition.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    frame_recognition.split_folds(n_folds=5, ensure_balancing=True)
    frame_recognition.infer_stats()
    frame_recognition.push_and_test()
