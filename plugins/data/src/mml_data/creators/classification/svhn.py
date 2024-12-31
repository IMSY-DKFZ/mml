# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from torchvision.datasets import SVHN

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@inproceedings{Netzer2011ReadingDI,
  title={Reading Digits in Natural Images with Unsupervised Feature Learning},
  author={Yuval Netzer and Tiejie Wang and Adam Coates and A. Bissacco and Bo Wu and A. Ng},
  year={2011}
}
"""

dset_name = "svhn"
classes = [str(ix) for ix in range(10)]


@register_dsetcreator(dset_name=dset_name)
def create_street_view_house_numbers():
    dset_creator = DSetCreator(dset_name=dset_name)
    train = SVHN(root=dset_creator.download_path, split="train", download=True)
    test = SVHN(root=dset_creator.download_path, split="test", download=True)
    # TODO see if extra may be added as unlabeled additional data
    # extra = SVHN(root=dset_creator.download_path, split='extra', download=True)
    dset_path = dset_creator.extract_from_pytorch_datasets(
        datasets={"training": train, "testing": test}, task_type=TaskType.CLASSIFICATION, class_names=classes
    )
    return dset_path


task_name = "svhn"


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_svhn(dset_path: Path):
    svhn = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="Street view house numbers classification",
        ref=REFERENCE,
        url="http://ufldl.stanford.edu/housenumbers/",
        instr="downloaded via torchvision dataset (https://pytorch.org/vision/stable/datasets.html#svhn)",
        lic=License.UNKNOWN,
        release="2011",
        keywords=[Keyword.CHARS_DIGITS, Keyword.NATURAL_OBJECTS],
    )
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=classes
    )
    test_iterator, idx_to_class_2 = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TESTING_DATA, classes=classes
    )
    assert all([a == b for a, b in zip(idx_to_class, idx_to_class_2)])
    svhn.find_data(train_iterator=train_iterator, test_iterator=test_iterator, idx_to_class=idx_to_class)
    svhn.split_folds(n_folds=5, ensure_balancing=True)
    svhn.infer_stats()
    svhn.push_and_test()
