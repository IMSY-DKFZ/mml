# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from torchvision.datasets import Caltech101

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@inproceedings{FeiFei2004LearningGV,
  title={Learning Generative Visual Models from Few Training Examples: An Incremental Bayesian Approach Tested on 101 Object Categories},
  author={Li Fei-Fei and Rob Fergus and Pietro Perona},
  booktitle={CVPR Workshops},
  year={2004}
}
"""

dset_name = "caltech101"
task_name = "caltech101_object_classification"


# TODO The caltech101 download url has changed. extract_from_pytorch_dataset is not working anymore!


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dset_name)
    vision_dset = Caltech101(root=dset_creator.download_path, target_type="category", download=True)
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
        desc="Pictures of objects belonging to 101 categories. About 40 to 800 images per category. "
        "Most categories have about 50 images. Collected in September 2003 by Fei-Fei Li, Marco "
        "Andreetto, and Marc 'Aurelio Ranzato.  The size of each image is roughly 300 x 200 "
        "pixels.",
        ref=REFERENCE,
        url="http://www.vision.caltech.edu/Image_Datasets/Caltech101/",
        instr="downloaded via torchvision dataset "
        "(https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.Caltech101)",
        lic=License.UNKNOWN,
        release="2004",
        keywords=[Keyword.NATURAL_OBJECTS],
    )
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=None
    )
    task.find_data(train_iterator=train_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
