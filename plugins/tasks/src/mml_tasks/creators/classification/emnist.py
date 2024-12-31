# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from torchvision.datasets import EMNIST

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

dset_name = "emnist"
task_name = "emnist_digit_classification"


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dset_name)
    train_dset = EMNIST(root=dset_creator.download_path, train=True, download=True, split="balanced")
    test_dset = EMNIST(root=dset_creator.download_path, train=False, download=True, split="balanced")
    dset_path = dset_creator.extract_from_pytorch_datasets(
        datasets={"training": train_dset, "testing": test_dset},
        task_type=TaskType.CLASSIFICATION,
        class_names=train_dset.classes,
    )
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="A variant of the full NIST dataset, which was called Extended MNIST (EMNIST), "
        "and follows the same conversion paradigm used to create the MNIST dataset. ",
        ref=REFERENCE,
        url="https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist",
        instr="downloaded via torchvision dataset "
        "(https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.EMNIST)",
        lic=License.UNKNOWN,
        release="2017",
        keywords=[Keyword.CHARS_DIGITS, Keyword.HANDWRITINGS],
    )
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=None
    )
    task.find_data(train_iterator=train_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
