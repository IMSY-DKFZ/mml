# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import copy

import cv2
import numpy as np
import pytest
import torch
from omegaconf import ListConfig

from mml.core.data_loading.augmentations.albumentations import AlbumentationsAugmentationModule
from mml.core.data_loading.task_attributes import EMPTY_MASK_TOKEN, IMAGENET_MEAN, IMAGENET_STD, DataSplit, Modality
from mml.core.data_loading.task_dataset import TaskDataset


@pytest.mark.parametrize("idx_dict", [{0: "background", 1: "interest"}, {3: "test", 4: "test", 5: "other"}])
def test_get_classes_from_idx_dict_success(idx_dict):
    classes = TaskDataset.get_classes_from_idx_dict(idx_to_class=idx_dict)
    # each class shall be unique
    assert len(classes) == len(set(classes))
    # classes shall be independent of dict order
    reversed_dict = {k: v for k, v in reversed(idx_dict.items())}
    assert classes == TaskDataset.get_classes_from_idx_dict(reversed_dict)


@pytest.fixture
def class_task_dataset_instance(dummy_meta_class_path):
    yield TaskDataset(root=dummy_meta_class_path, split=DataSplit.TRAIN, fold=0, transform=None)


@pytest.fixture
def seg_task_dataset_instance(dummy_meta_seg_path):
    yield TaskDataset(root=dummy_meta_seg_path, split=DataSplit.TRAIN, fold=0, transform=None)


def test_task_sample_selection(class_task_dataset_instance):
    sample_backup = copy.deepcopy(class_task_dataset_instance.samples)
    class_task_dataset_instance.select_samples(split=DataSplit.TRAIN, fold=1)
    assert len(class_task_dataset_instance.samples) == len(sample_backup)
    assert class_task_dataset_instance.samples != sample_backup
    class_task_dataset_instance.select_samples(split=DataSplit.VAL, fold=0)
    assert all([val_sample not in sample_backup for val_sample in class_task_dataset_instance.samples])
    assert len(class_task_dataset_instance.samples) + len(sample_backup) == len(
        class_task_dataset_instance.task_description.train_samples
    )
    class_task_dataset_instance.select_samples(split=DataSplit.TRAIN, fold=0)
    assert class_task_dataset_instance.samples == sample_backup


def test_task_sample_loading(class_task_dataset_instance, seg_task_dataset_instance, monkeypatch, image):
    def dummy_imread(*args):
        return image.copy()

    monkeypatch.setattr(target=cv2, name="imread", value=dummy_imread)
    for task_dataset_instance in [seg_task_dataset_instance, class_task_dataset_instance]:
        sample = task_dataset_instance[0]
        assert isinstance(sample, dict)
        assert (Modality.CLASS.value in sample or Modality.MASK.value in sample) and Modality.IMAGE.value in sample
        if Modality.CLASS.value in sample:
            assert isinstance(sample[Modality.CLASS.value], torch.Tensor)
        if Modality.MASK.value in sample:
            assert isinstance(sample[Modality.MASK.value], np.ndarray)
        sample_duplicate = task_dataset_instance[0]
        assert np.array_equal(sample[Modality.IMAGE.value], sample_duplicate[Modality.IMAGE.value])
        assert sample is not sample_duplicate
        task_dataset_instance.transform = AlbumentationsAugmentationModule(
            device="cpu", cfg=ListConfig([]), is_first=True, is_last=True, means=IMAGENET_MEAN, stds=IMAGENET_STD
        )
        sample_trans = task_dataset_instance[0]
        assert isinstance(sample_trans[Modality.IMAGE.value], torch.Tensor)
        assert not np.array_equal(sample[Modality.IMAGE.value], sample_trans[Modality.IMAGE.value].numpy())


@pytest.mark.parametrize("size", [(100, 100), (150, 100)])
def test_empty_mask_token_loading(seg_task_dataset_instance, monkeypatch, size):
    all_empties = [
        ix
        for ix, sample_dict in enumerate(seg_task_dataset_instance.samples)
        if sample_dict[Modality.MASK.value] == EMPTY_MASK_TOKEN
    ]
    assert len(all_empties) > 0

    def dummy_imread(*args):
        return np.ones((*size, 3), dtype=np.uint8)

    monkeypatch.setattr(target=cv2, name="imread", value=dummy_imread)
    loaded_sample = seg_task_dataset_instance[all_empties[0]]

    assert Modality.MASK.value in loaded_sample
    assert np.unique(loaded_sample[Modality.MASK.value]).size == 1
    assert 0 in np.unique(loaded_sample[Modality.MASK.value])
    assert loaded_sample[Modality.MASK.value].shape[:2] == loaded_sample[Modality.IMAGE.value].shape[:2]
    assert loaded_sample[Modality.MASK.value].ndim == 2


def test_caching(dummy_meta_class_path, monkeypatch, image):
    def dummy_imread(*args):
        return image.copy()

    monkeypatch.setattr(target=cv2, name="imread", value=dummy_imread)
    # create dataset and cache
    ds = TaskDataset(root=dummy_meta_class_path, split=DataSplit.TRAIN, fold=0, transform=None, caching_limit=100)
    ds.fill_cache(num_workers=2)

    # use cache and avoid reading images

    def imread_error(*args):
        raise RuntimeError

    monkeypatch.setattr(target=cv2, name="imread", value=imread_error)
    for _ in ds:
        pass
