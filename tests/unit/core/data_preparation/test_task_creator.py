# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import copy
import shutil
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from mml.core.data_loading.task_attributes import DataSplit, Modality, RGBInfo, Sizes, TaskType
from mml.core.data_loading.task_dataset import TaskDataset
from mml.core.data_preparation.task_creator import TaskCreator, implements_action
from mml.core.data_preparation.utils import TaskCreatorActions, TaskCreatorState
from mml.core.scripts.exceptions import InvalidTransitionError
from mml.core.scripts.utils import ARG_SEP, TAG_SEP


@pytest.fixture()
def dummy_task_creator(tmp_path, file_manager):
    yield TaskCreator(dset_path=tmp_path)


@pytest.fixture()
def patch_loading(image, monkeypatch):
    def dummy_loading(self, index):
        return {"image": image, "class": np.random.randint(low=0, high=3)}

    def dummy_pillow(*args, **kwargs):
        return Image.fromarray(image)

    monkeypatch.setattr(target=TaskDataset, name="load_sample", value=dummy_loading)
    monkeypatch.setattr(target=Image, name="open", value=dummy_pillow)
    yield


def test_task_creator_protocol(dummy_task_creator):
    old_protocol = copy.copy(dummy_task_creator.current_meta.creation_protocol)
    dummy_task_creator.protocol("Random important message!")
    assert dummy_task_creator.current_meta.creation_protocol.startswith(old_protocol)
    assert len(dummy_task_creator.current_meta.creation_protocol) > len(old_protocol)
    assert "Random important message!" in dummy_task_creator.current_meta.creation_protocol


def test_loading(dummy_meta_class_path, file_manager):
    creator = TaskCreator(dset_path=dummy_meta_class_path.parent)
    creator.load_existent(dummy_meta_class_path)
    assert "Copied" in creator.current_meta.creation_protocol


@pytest.mark.parametrize(
    argnames=["class_list", "n_folds", "balancing", "fraction", "expected_split"],
    argvalues=[
        # simplest case
        ([0, 1, 1, 1, 0, 1, 1, 1, 0, 0], 5, False, None, [[2, 6], [8, 1], [5, 4], [0, 9], [7, 3]]),
        # turning on balancing changes output
        ([0, 1, 0, 1, 0, 1, 1, 1, 0, 0], 5, True, None, [[4, 5], [7, 9], [6, 8], [0, 1], [3, 2]]),
        # as long as ensure balancing is enabled, changing the class list also changes output
        ([0, 0, 1, 0, 1, 1, 1, 0, 0, 1], 5, True, None, [[3, 5], [9, 8], [6, 7], [0, 2], [4, 1]]),
        # if pool sizes are not multiples of n_folds, it may happen that fold_0 is the largest
        ([0, 1, 1, 1, 0, 1, 1, 1, 1, 0], 2, True, None, [[5, 0, 2, 8, 9, 7], [1, 6, 3, 4]]),
        # it may also happen that fold_1 is the largest
        ([0, 1, 1, 1, 0, 1, 1, 1, 1, 0], 3, True, None, [[0, 8, 2], [7, 5, 6, 9], [1, 4, 3]]),
        # if pool sizes are multiples of n_folds, folds should be of equal size
        ([0, 1, 1, 1, 0, 1, 1, 0, 1], 3, True, None, [[0, 5, 1], [6, 3, 7], [2, 4, 8]]),
        # increasing the fraction of fold 0
        ([0, 1, 1, 1, 0, 1, 1, 1, 0, 0], 5, False, 0.6, [[4, 2, 1, 8, 6, 5], [0], [9], [3], [7]]),
        # decreasing the fraction of fold 0
        ([0, 1, 1, 1, 0, 1, 1, 1, 0, 0], 5, False, 0.1, [[2], [8, 1, 6], [5, 4], [0, 9], [7, 3]]),
        # increasing the fraction of fold 0 in balanced mode
        ([0, 1, 1, 1, 0, 1, 1, 1, 0, 0], 2, True, 0.6, [[3, 8, 1, 5, 9, 6], [2, 4, 7, 0]]),
    ],
)
def test_fold_splitting_success(dummy_task_creator, class_list, n_folds, balancing, fraction, expected_split):
    dummy_task_creator.data = {
        DataSplit.FULL_TRAIN: {str(ix): {Modality.CLASS: cls_id} for ix, cls_id in enumerate(class_list)}
    }
    dummy_task_creator.current_meta.class_occ = Counter(class_list)
    dummy_task_creator.current_meta.idx_to_class = {ix: str(ix) for ix in set(class_list)}
    if balancing:
        dummy_task_creator.current_meta.task_type = TaskType.CLASSIFICATION
    dummy_task_creator.split_folds(
        n_folds=n_folds, ensure_balancing=balancing, fold_0_fraction=fraction, ignore_state=True
    )
    assert dummy_task_creator.current_meta.train_folds == [[str(s_id) for s_id in fold] for fold in expected_split]


@pytest.mark.parametrize(
    argnames=["class_list", "n_folds", "balancing", "fraction"],
    argvalues=[
        # too many folds
        ([0, 1, 1, 1, 0, 1, 1, 1, 1, 0], 11, False, None),
        # too few folds
        ([0, 1, 1, 1, 0, 1, 1, 1, 1, 0], 1, False, None),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5, True, None),
        # balancing not possible with the number of folds
        ([0, 1, 1, 1, 0, 1, 1, 1, 1, 0], 4, True, None),
        ([0, 1, 1, 1, 0, 1, 1, 1, 1], 2, True, 2 / 3),
        # first fold fraction too small
        ([0, 1, 0, 1, 0, 0, 1, 0, 1], 5, True, 0.1),
    ],
)
def test_fold_splitting_failure(dummy_task_creator, class_list, n_folds, balancing, fraction):
    dummy_task_creator.data = {
        DataSplit.FULL_TRAIN: {str(ix): {Modality.CLASS: cls_id} for ix, cls_id in enumerate(class_list)}
    }
    dummy_task_creator.current_meta.class_occ = Counter(class_list)
    dummy_task_creator.current_meta.idx_to_class = {ix: str(ix) for ix in set(class_list)}
    if balancing:
        dummy_task_creator.current_meta.task_type = TaskType.CLASSIFICATION
    with pytest.raises(ValueError):
        dummy_task_creator.split_folds(
            n_folds=n_folds, ensure_balancing=balancing, fold_0_fraction=fraction, ignore_state=True
        )


def test_invalid_traversal(dummy_meta_class_path, file_manager, monkeypatch):
    @implements_action(TaskCreatorActions.SET_FOLDING)
    def dummy_traversal(self: TaskCreator) -> None:
        pass

    creator = TaskCreator(dset_path=dummy_meta_class_path.parent)
    creator.load_existent(dummy_meta_class_path)

    dummy_traversal(creator)

    with pytest.raises(InvalidTransitionError):
        creator.push_and_test()


def test_valid_traversals(dummy_meta_class_path, file_manager, patch_loading):
    creator = TaskCreator(dset_path=dummy_meta_class_path.parent)
    creator.load_existent(dummy_meta_class_path)
    creator.identity()
    creator.set_stats(means=RGBInfo(0.5, 0.5, 0.5), stds=RGBInfo(0.29, 0.29, 0.29), sizes=Sizes(224, 224, 224, 224))


def test_auto_complete_after_modify(dummy_meta_class_path, file_manager, patch_loading):
    p = file_manager.raw_data / "dummy_dset"
    p.mkdir()
    shutil.copy(src=dummy_meta_class_path, dst=p)
    creator = TaskCreator(dset_path=p)
    creator.load_existent(p / dummy_meta_class_path.name)
    creator.identity()
    creator.data = {
        DataSplit.FULL_TRAIN: {
            f"ID_{ix}": {Modality.CLASS: ix % 3, Modality.IMAGE: f"dummy_{ix}.bmp"} for ix in range(100)
        }
    }
    creator.current_meta.class_occ = Counter(
        [
            creator.current_meta.idx_to_class[elem[Modality.CLASS]]
            for elem in creator.data[DataSplit.FULL_TRAIN].values()
        ]
    )
    creator._state = TaskCreatorState.DATA_FOUND
    creator.auto_complete(device=torch.device("cpu"))


def test_identity_tag(file_manager, dummy_meta_class_path, patch_loading):
    p = file_manager.raw_data / "dummy_dset"
    p.mkdir()
    shutil.copy(src=dummy_meta_class_path, dst=p)
    file_manager.add_to_task_index(path=p / dummy_meta_class_path.name)
    new_path = TaskCreator.auto_create_tagged(full_alias=f"dummy_task{TAG_SEP}identity")
    assert file_manager.load_task_description(new_path).name == f"dummy_task{TAG_SEP}identity"


@pytest.mark.parametrize(
    argnames=["fold_str", "new_folds_str", "valid"],
    argvalues=[
        # too few folds
        ("3", "1", False),
        # fold number invalid
        ("7", "4", False),
        ("-1", "3", False),
        ("nine", "3", False),
        ("", "3", False),
        # valid cases, unfortunately higher new_folds leads to impossible balancing
        ("4", "2", True),
        ("0", "2", True),
    ],
)
def test_nested_validation_tag(file_manager, dummy_meta_class_path, patch_loading, fold_str, new_folds_str, valid):
    p = file_manager.raw_data / "dummy_dset"
    p.mkdir()
    shutil.copy(src=dummy_meta_class_path, dst=p)
    file_manager.add_to_task_index(path=p / dummy_meta_class_path.name)
    old_meta = file_manager.load_task_description(p / dummy_meta_class_path.name)
    if not valid:
        with pytest.raises(ValueError):
            _ = TaskCreator.auto_create_tagged(
                full_alias=f"dummy_task{TAG_SEP}nested{ARG_SEP}{fold_str}{ARG_SEP}{new_folds_str}"
            )
    else:
        new_path = TaskCreator.auto_create_tagged(
            full_alias=f"dummy_task{TAG_SEP}nested{ARG_SEP}{fold_str}{ARG_SEP}{new_folds_str}"
        )
        new_meta = file_manager.load_task_description(new_path)
        # correct number of folds in new task
        assert len(new_meta.train_folds) == int(new_folds_str)
        # old validation moved to test samples
        assert all([s_id in new_meta.test_samples for s_id in old_meta.train_folds[int(fold_str)]])
        # fold one is never moved to test samples
        assert all([s_id not in new_meta.test_samples for s_id in old_meta.train_folds[1]])
