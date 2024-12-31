# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from torchvision.datasets import MNIST

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@inproceedings{LeCun1998GradientbasedLA,
  title={Gradient-based learning applied to document recognition},
  author={Yann LeCun and L{\'e}on Bottou and Yoshua Bengio and Patrick Haffner},
  year={1998}
}
"""

dset_name = "mnist"
task_name = "mnist_digit_classification"


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dset_name)
    train_dset = MNIST(root=dset_creator.download_path, train=True, download=True)
    test_dset = MNIST(root=dset_creator.download_path, train=False, download=True)
    dset_path = dset_creator.extract_from_pytorch_datasets(
        datasets={"training": train_dset, "testing": test_dset},
        task_type=TaskType.CLASSIFICATION,
        class_names=[str(ix) for ix in range(10)],
    )
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="The MNIST database of handwritten digits, available from this page, has a training set "
        "of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set "
        "available from NIST. The digits have been size-normalized and centered in a fixed-size "
        "image. ",
        ref=REFERENCE,
        url="http://yann.lecun.com/exdb/mnist/",
        instr="downloaded via torchvision dataset "
        "(https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST)",
        lic=License.UNKNOWN,
        release="1998",
        keywords=[Keyword.CHARS_DIGITS, Keyword.HANDWRITINGS],
    )
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=None
    )
    task.find_data(train_iterator=train_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
