# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@inproceedings{Krizhevsky2009LearningML,
  title={Learning Multiple Layers of Features from Tiny Images},
  author={Alex Krizhevsky},
  year={2009}
}
"""

dset_names = ["cifar10", "cifar100"]
task_names = ["cifar10_object_classification", "cifar100_object_classification"]


@register_dsetcreator(dset_name=dset_names[0])
def create_cifar10():
    dset_creator = DSetCreator(dset_name=dset_names[0])
    train_dset = CIFAR10(root=dset_creator.download_path, download=True, train=True)
    test_dset = CIFAR10(root=dset_creator.download_path, download=True, train=False)
    dset_path = dset_creator.extract_from_pytorch_datasets(
        datasets={"training": train_dset, "testing": test_dset},
        task_type=TaskType.CLASSIFICATION,
        class_names=train_dset.classes,
    )
    return dset_path


@register_taskcreator(task_name=task_names[0], dset_name=dset_names[0])
def create_cifar_10_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_names[0],
        task_type=TaskType.CLASSIFICATION,
        desc="The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 "
        "images per class. There are 50000 training images and 10000 test images. ",
        ref=REFERENCE,
        url="https://www.cs.toronto.edu/~kriz/cifar.html",
        instr="downloaded via torchvision dataset "
        "(https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10)",
        lic=License.UNKNOWN,
        release="2009",
        keywords=[Keyword.NATURAL_OBJECTS],
    )
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=None
    )
    task.find_data(train_iterator=train_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()


@register_dsetcreator(dset_name=dset_names[1])
def create_cifar100():
    dset_creator = DSetCreator(dset_name=dset_names[1])
    train_dset = CIFAR100(root=dset_creator.download_path, download=True, train=True)
    test_dset = CIFAR100(root=dset_creator.download_path, download=True, train=False)
    dset_path = dset_creator.extract_from_pytorch_datasets(
        datasets={"training": train_dset, "testing": test_dset},
        task_type=TaskType.CLASSIFICATION,
        class_names=train_dset.classes,
    )
    return dset_path


@register_taskcreator(task_name=task_names[1], dset_name=dset_names[1])
def create_cifar_100_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_names[1],
        task_type=TaskType.CLASSIFICATION,
        desc="This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images "
        "each. There are 500 training images and 100 testing images per class. The 100 classes in "
        "the CIFAR-100 are grouped into 20 superclasses. Each image comes with a 'fine' label (the "
        "class to which it belongs) and a 'coarse' label (the superclass to which it belongs).",
        ref=REFERENCE,
        url="https://www.cs.toronto.edu/~kriz/cifar.html",
        instr="downloaded via torchvision dataset "
        "(https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR100)",
        lic=License.UNKNOWN,
        release="2009",
        keywords=[Keyword.NATURAL_OBJECTS],
    )
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=None
    )
    task.find_data(train_iterator=train_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
