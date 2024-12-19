# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
url = {https://aidasub-clebarrett.grand-challenge.org/home/},
title = {Analysis of Images to Detect Abnormalities in Endoscopy (AIDA-E) challenge}, 
published during the 2016 edition of The IEEE International Symposium on Biomedical Imaging (ISBI)
"""  # noqa W291


@register_dsetcreator(dset_name="barretts_esophagus")
def create_dset():
    instructions = """
     Download link is not available. Please download the dataset by clicking on the download button manually via
     https://www.dropbox.com/sh/m4xkmwjv3dm5j4b/AADICJ7IsBFZfSfQUM5TeClma/CLE_barrett.zip
     Once the download is complete place the downloaded folder 'CLE_barrett.zip' in
     <MML_DATA_ROOT>\\DOWNLOADS\\barretts_esophagus
    """
    dset_creator = DSetCreator(dset_name="barretts_esophagus")
    dset_creator.verify_pre_download(
        file_name="CLE_barrett.zip", instructions=instructions, data_kind=DataKind.TRAINING_DATA
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="barretts_esophagus_diagnosis", dset_name="barretts_esophagus")
def create_barretts(dset_path: Path):
    barretts_diagnosis = TaskCreator(
        dset_path=dset_path,
        name="barretts_esophagus_diagnosis",
        task_type=TaskType.CLASSIFICATION,
        desc="Barretts esophagus dataset showing gastric metaplasia (GMP), "
        "intestinal metaplasia or proper Barrett's esophagus (BAR), or neplasia (NPL)",
        ref=REFERENCE,
        url="https://aidasub-clebarrett.grand-challenge.org/home/",
        instr="download via dropbox "
        "https://www.dropbox.com/sh/m4xkmwjv3dm5j4b/AADICJ7IsBFZfSfQUM5TeClma/CLE_barrett.zip",
        lic=License.UNKNOWN,
        release="2016",
        keywords=[Keyword.MEDICAL, Keyword.ENDOSCOPY, Keyword.CLE, Keyword.TISSUE_PATHOLOGY],
    )
    data_iterator = []
    classes = ["GMP", "BAR", "NPL"]
    idx_to_class = {ix: cls for ix, cls in enumerate(classes)}
    for img_path in (dset_path / DataKind.TRAINING_DATA).iterdir():
        data_iterator.append(
            {
                Modality.SAMPLE_ID: img_path.name,
                Modality.IMAGE: img_path,
                Modality.CLASS: classes.index(img_path.name[:3]),
            }
        )
    barretts_diagnosis.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    barretts_diagnosis.split_folds(n_folds=5, ensure_balancing=True)
    barretts_diagnosis.infer_stats()
    barretts_diagnosis.push_and_test()
