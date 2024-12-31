# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from torchvision.datasets import Caltech256

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@inproceedings{Griffin2007Caltech256OC,
  title={Caltech-256 Object Category Dataset},
  author={Gregory Griffin and Alex Holub and Pietro Perona},
  year={2007}
}
"""

# TODO The caltech256 download url has changed. extract_from_pytorch_dataset is not working anymore!

dset_name = "caltech256"
task_name = "caltech256_object_classification"


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dset_name)
    vision_dset = Caltech256(root=dset_creator.download_path, download=True)
    artefact = dset_creator.download_path / "caltech256" / "256_ObjectCategories" / "056.dog" / "greg" / "vision309"
    # there seems to be an empty folder that causes errors, so to be save we will remove it and reload the dataset
    if artefact.exists():
        artefact.rmdir()
        artefact.parent.rmdir()
        vision_dset = Caltech256(root=dset_creator.download_path, download=False)
    # Furthermore there is a bash script, that causes the same error
    artefact = dset_creator.download_path / "caltech256" / "256_ObjectCategories" / "198.spider" / "RENAME2"
    if artefact.exists():
        artefact.unlink()
        vision_dset = Caltech256(root=dset_creator.download_path, download=False)
    dset_path = dset_creator.extract_from_pytorch_datasets(
        datasets={"training": vision_dset}, task_type=TaskType.CLASSIFICATION, class_names=vision_dset.categories
    )
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="Collection of 30607 images from 256 categories",
        ref=REFERENCE,
        url="http://www.vision.caltech.edu/Image_Datasets/Caltech256/",
        instr="downloaded via torchvision dataset "
        "(https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.Caltech256)",
        lic=License.UNKNOWN,
        release="2007",
        keywords=[Keyword.NATURAL_OBJECTS],
    )
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=None
    )
    task.find_data(train_iterator=train_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
