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
@ONLINE {beansdata,
    author="Makerere AI Lab",
    title="Bean disease dataset",
    month="January",
    year="2020",
    url="https://github.com/AI-Lab-Makerere/ibean/"
}
"""

classes = ["healthy", "bean_rust", "angular_leaf_spot"]


@register_dsetcreator(dset_name="ibean")
def create_dset():
    dset_creator = DSetCreator(dset_name="ibean")
    dset_creator.download(
        url="https://storage.googleapis.com/ibeans/train.zip", file_name="train.zip", data_kind=DataKind.TRAINING_DATA
    )
    dset_creator.download(
        url="https://storage.googleapis.com/ibeans/validation.zip",
        file_name="validation.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_creator.download(
        url="https://storage.googleapis.com/ibeans/test.zip", file_name="test.zip", data_kind=DataKind.TESTING_DATA
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="bean_plant_disease_classification", dset_name="ibean")
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name="bean_plant_disease_classification",
        task_type=TaskType.CLASSIFICATION,
        desc="ibean dataset for bean plant disease classification. The data is of leaf"
        "images representing 3 classes: the healthy class of images, and two "
        "disease classes including Angular Leaf Spot and Bean Rust diseases.",
        ref=REFERENCE,
        url="https://github.com/AI-Lab-Makerere/ibean/",
        instr="download individual train, validation and test dataset from github.com/AI-Lab-Makerere/ibean/",
        lic=License.MIT,
        release="2020",
        keywords=[Keyword.NATURAL_OBJECTS],
    )
    train_iterator = []
    test_iterator = []
    train_path = [dset_path / DataKind.TRAINING_DATA / "train", dset_path / DataKind.TRAINING_DATA / "validation"]
    test_path = dset_path / DataKind.TESTING_DATA / "test"
    for seq in train_path:
        for class_folder in seq.iterdir():
            for img_path in class_folder.iterdir():
                if img_path.name == "healthy_train.120tore":  # corrupted image
                    continue
                train_iterator.append(
                    {
                        Modality.SAMPLE_ID: img_path.name,
                        Modality.IMAGE: img_path,
                        Modality.CLASS: classes.index(class_folder.name),
                    }
                )
    for class_folder in test_path.iterdir():
        for img_path in class_folder.iterdir():
            test_iterator.append(
                {
                    Modality.SAMPLE_ID: img_path.name,
                    Modality.IMAGE: img_path,
                    Modality.CLASS: classes.index(class_folder.name),
                }
            )
    idx_to_class = {classes.index(cl): cl for cl in classes}
    task.find_data(train_iterator=train_iterator, test_iterator=test_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
