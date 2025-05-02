# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import os
from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@dataset{sara_moccia_2018_1162784,
  author       = {Sara Moccia and
                  Gabriele Omodeo Vanone and
                  Elena De Momi and
                  Leonardo S. Mattos},
  title        = {NBI-InfFrames},
  month        = jan,
  year         = 2018,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.1162784},
  url          = {https://doi.org/10.5281/zenodo.1162784}
}
"""


@register_dsetcreator(dset_name="nbi_infframes")
def create_nbi_infframes():
    dset_creator = DSetCreator(dset_name="nbi_infframes")
    dset_creator.download(
        url="https://zenodo.org/record/1162784/files/FRAMES.zip?download=1",
        file_name="FRAMES.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="identify_nbi_infframes", dset_name="nbi_infframes")
def create_identify_nbi_infframes(dset_path: Path):
    nbi_infframes = TaskCreator(
        dset_path=dset_path,
        name="identify_nbi_infframes",
        task_type=TaskType.CLASSIFICATION,
        desc="NBI-InfFrames dataset for the identification of informative endoscopic video frames.",
        ref=REFERENCE,
        url="https://nearlab.polimi.it/medical/dataset/",
        instr="download via zenodo.org/record/1162784/files/FRAMES.zip?download=1",
        lic=License.CC_BY_NC_4_0,
        release="Version 1",
        keywords=[Keyword.MEDICAL, Keyword.LARYNGOSCOPY, Keyword.IMAGE_ARTEFACTS, Keyword.ENDOSCOPY],
    )
    classes = ["B", "I", "S", "U"]
    folds = ["Fold1", "Fold2", "Fold3"]
    data_iterator = []
    for fold in folds:
        root = dset_path / DataKind.TRAINING_DATA / "FRAMES" / f"{fold}"
        folders = [p.name for p in root.iterdir() if p.is_dir()]
        assert all([cl in folders for cl in classes]), "some class folder is not existent"
        for class_folder in root.iterdir():
            if class_folder.name != ".DS_Store":
                assert class_folder.is_dir()
            if class_folder.name not in classes:
                continue
            for img_path in class_folder.iterdir():
                head, tail = os.path.split(img_path)
                if tail != ".DS_Store":
                    data_iterator.append(
                        {
                            Modality.SAMPLE_ID: img_path.stem,
                            Modality.IMAGE: img_path,
                            Modality.CLASS: classes.index(class_folder.name),
                        }
                    )
    idx_to_class = {classes.index(cl): cl for cl in classes}
    nbi_infframes.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    nbi_infframes.split_folds(n_folds=5, ensure_balancing=True)
    nbi_infframes.infer_stats()
    nbi_infframes.push_and_test()


@register_taskcreator(task_name="identify_nbi_infframes_original_folds", dset_name="nbi_infframes")
def create_identify_nbi_infframes_original_folds(dset_path: Path):
    nbi_infframes_original_folds = TaskCreator(
        dset_path=dset_path,
        name="identify_nbi_infframes_original_folds",
        task_type=TaskType.CLASSIFICATION,
        desc="NBI-InfFrames dataset for the identification of informative endoscopic video frames.",
        ref=REFERENCE,
        url="https://nearlab.polimi.it/medical/dataset/",
        instr="download via zenodo.org/record/1162784/files/FRAMES.zip?download=1",
        lic=License.CC_BY_NC_4_0,
        release="Version 1",
        keywords=[Keyword.MEDICAL, Keyword.LARYNGOSCOPY, Keyword.IMAGE_ARTEFACTS],
    )
    classes = ["B", "I", "S", "U"]
    folds = ["Fold1", "Fold2", "Fold3"]
    data_iterator = []
    fold_definition = []
    for fold in folds:
        root = dset_path / DataKind.TRAINING_DATA / "FRAMES" / f"{fold}"
        folders = [p.name for p in root.iterdir() if p.is_dir()]
        assert all([cl in folders for cl in classes]), "some class folder is not existent"
        ids = []
        for class_folder in root.iterdir():
            if class_folder.name != ".DS_Store":
                assert class_folder.is_dir()
            if class_folder.name not in classes:
                continue
            for img_path in class_folder.iterdir():
                head, tail = os.path.split(img_path)
                if tail != ".DS_Store":
                    ids.append(img_path.stem)
                    data_iterator.append(
                        {
                            Modality.SAMPLE_ID: img_path.stem,
                            Modality.IMAGE: img_path,
                            Modality.CLASS: classes.index(class_folder.name),
                        }
                    )
        fold_definition.append(ids)
    idx_to_class = {classes.index(cl): cl for cl in classes}
    nbi_infframes_original_folds.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    nbi_infframes_original_folds.use_existing_folds(fold_definition=fold_definition)
    nbi_infframes_original_folds.infer_stats()
    nbi_infframes_original_folds.push_and_test()
