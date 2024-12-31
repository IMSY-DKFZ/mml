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
@dataset{sara_moccia_2017_1003200,
  author       = {Sara Moccia and
                  Elena De Momi and
                  Leonardo S. Mattos},
  title        = {Laryngeal dataset},
  month        = oct,
  year         = 2017,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.1003200},
  url          = {https://doi.org/10.5281/zenodo.1003200}
}
"""


@register_dsetcreator(dset_name="laryngeal")
def create_laryngeal():
    dset_creator = DSetCreator(dset_name="laryngeal")
    dset_creator.download(
        url="https://zenodo.org/record/1003200/files/laryngeal%20dataset.tar?download=1",
        file_name="laryngeal dataset.tar",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="laryngeal_tissues", dset_name="laryngeal")
def create_laryngeal_tissue(dset_path: Path):
    laryngeal_tissue = TaskCreator(
        dset_path=dset_path,
        name="laryngeal_tissues",
        task_type=TaskType.CLASSIFICATION,
        desc="Laryngeal dataset for patches of healthy and early-stage cancerous laryngeal tissues",
        ref=REFERENCE,
        url="https://nearlab.polimi.it/medical/dataset/",
        instr="download via zenodo.org/record/1003200/files/laryngeal%20dataset.tar?download=1",
        lic=License.CC_BY_NC_4_0,
        release="2017",
        keywords=[Keyword.MEDICAL, Keyword.LARYNGOSCOPY, Keyword.TISSUE_PATHOLOGY, Keyword.ENDOSCOPY],
    )
    classes = ["Hbv", "He", "IPCL", "Le"]
    folds = ["FOLD 1", "FOLD 2", "FOLD 3"]
    data_iterator = []
    for fold in folds:
        root = dset_path / DataKind.TRAINING_DATA / "laryngeal dataset" / f"{fold}"
        folders = [p.name for p in root.iterdir() if p.is_dir()]
        assert all([cl in folders for cl in classes]), "some class folder is not existent"
        for class_folder in root.iterdir():
            assert class_folder.is_dir()
            if class_folder.name not in classes:
                continue
            for img_path in class_folder.iterdir():
                data_iterator.append(
                    {
                        Modality.SAMPLE_ID: img_path.stem,
                        Modality.IMAGE: img_path,
                        Modality.CLASS: classes.index(class_folder.name),
                    }
                )
    idx_to_class = {classes.index(cl): cl for cl in classes}
    laryngeal_tissue.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    laryngeal_tissue.split_folds(n_folds=5, ensure_balancing=True)
    laryngeal_tissue.infer_stats()
    laryngeal_tissue.push_and_test()


@register_taskcreator(task_name="laryngeal_tissues_original_folds", dset_name="laryngeal")
def create_laryngeal_tissue_original_folds(dset_path: Path):
    laryngeal_tissue_original_folds = TaskCreator(
        dset_path=dset_path,
        name="laryngeal_tissues_original_folds",
        task_type=TaskType.CLASSIFICATION,
        desc="Laryngeal dataset for patches of healthy and early-stage cancerous laryngeal tissues",
        ref=REFERENCE,
        url="https://nearlab.polimi.it/medical/dataset/",
        instr="download via zenodo.org/record/1003200/files/laryngeal%20dataset.tar?download=1",
        lic=License.CC_BY_NC_4_0,
        release="2017",
        keywords=[Keyword.MEDICAL, Keyword.LARYNGOSCOPY, Keyword.TISSUE_PATHOLOGY, Keyword.ENDOSCOPY],
    )
    classes = ["Hbv", "He", "IPCL", "Le"]
    FOLDS = ["FOLD 1", "FOLD 2", "FOLD 3"]
    data_iterator = []
    fold_definition = []
    for fold in FOLDS:
        root = dset_path / DataKind.TRAINING_DATA / "laryngeal dataset" / f"{fold}"
        folders = [p.name for p in root.iterdir() if p.is_dir()]
        assert all([cl in folders for cl in classes]), "some class folder is not existent"
        ids = []
        for class_folder in root.iterdir():
            assert class_folder.is_dir()
            if class_folder.name not in classes:
                continue
            for img_path in class_folder.iterdir():
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
    laryngeal_tissue_original_folds.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    laryngeal_tissue_original_folds.use_existing_folds(fold_definition=fold_definition)
    laryngeal_tissue_original_folds.infer_stats()
    laryngeal_tissue_original_folds.push_and_test()
