# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

"""
Provides a fake task for testing purposes.
"""

import string
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import FakeData

from mml.core.data_loading.task_attributes import Keyword, License, RGBInfo, Sizes, TaskType
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import (
    get_iterator_and_mapping_from_image_dataset,
    get_iterator_from_unlabeled_dataset,
)

dset_name = "mml_fake_dataset"
task_name = "mml_fake_task"
num_classes = 10
classes = [char for char in string.ascii_uppercase[:num_classes]]


@register_dsetcreator(dset_name=dset_name)
def create_fake_dset() -> Path:
    dset_creator = DSetCreator(dset_name=dset_name)
    fake_train = FakeData(size=1000, num_classes=num_classes)
    fake_test = FakeData(size=500, num_classes=num_classes)

    class UnlabeledFakeData(Dataset):
        def __init__(self, size: int):
            self.internal = FakeData(size=size, num_classes=2)

        def __len__(self):
            return len(self.internal)

        def __getitem__(self, idx: int):
            return self.internal[idx][0]

    fake_unlabeled = UnlabeledFakeData(size=300)
    dset_path = dset_creator.extract_from_pytorch_datasets(
        datasets={"training": fake_train, "testing": fake_test, "unlabeled": fake_unlabeled},
        task_type=TaskType.CLASSIFICATION,
        class_names=classes,
    )
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_fake_task(dset_path: Path) -> Path:
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="Fake task, based on torchvision.datasets.FakeData.",
        ref="No reference.",
        url="No url.",
        instr="No instr.",
        lic=License.UNKNOWN,
        release="2004",
        keywords=[Keyword.ARTIFICIAL],
    )
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / "training_data", classes=classes
    )
    test_iterator, _ = get_iterator_and_mapping_from_image_dataset(root=dset_path / "testing_data", classes=classes)
    unlabeled_iterator = get_iterator_from_unlabeled_dataset(root=dset_path / "unlabeled_data")
    task.find_data(
        train_iterator=train_iterator,
        test_iterator=test_iterator,
        unlabeled_iterator=unlabeled_iterator,
        idx_to_class=idx_to_class,
    )
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.set_stats(means=RGBInfo(0.5, 0.5, 0.5), stds=RGBInfo(0.29, 0.29, 0.29), sizes=Sizes(224, 224, 224, 224))
    return task.push_and_test()
