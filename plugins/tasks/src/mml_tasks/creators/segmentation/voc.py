# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from torchvision.datasets import VOCSegmentation

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_from_segmentation_dataset

REFERENCE = """
@misc{pascal-voc-2012,
  author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
  title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults",
  howpublished = "http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"}
"""

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

VOC_COLORMAP = [
    tuple(ix)
    for ix in [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
]

dset_name = "VOC12"


@register_dsetcreator(dset_name=dset_name)
def create_voc12_dataset():
    dset_creator = DSetCreator(dset_name=dset_name)
    train = VOCSegmentation(root=dset_creator.download_path, year="2012", image_set="train", download=True)
    test = VOCSegmentation(root=dset_creator.download_path, year="2012", image_set="val", download=True)
    dset_path = dset_creator.extract_from_pytorch_datasets(
        datasets={"training": train, "testing": test}, task_type=TaskType.SEMANTIC_SEGMENTATION
    )
    train_iterator = get_iterator_from_segmentation_dataset(
        images_root=dset_path / DataKind.TRAINING_DATA, masks_root=dset_path / DataKind.TRAINING_LABELS
    )
    test_iterator = get_iterator_from_segmentation_dataset(
        images_root=dset_path / DataKind.TESTING_DATA, masks_root=dset_path / DataKind.TESTING_LABELS
    )
    train_masks = [item[Modality.MASK] for item in train_iterator]
    dset_creator.transform_masks(
        masks=train_masks,
        transform={pix_val: idx for idx, pix_val in enumerate(VOC_COLORMAP)},
        load="rgb",
        train=True,
        ignore=[(224, 224, 192)],
    )
    test_masks = [item[Modality.MASK] for item in test_iterator]
    dset_creator.transform_masks(
        masks=test_masks,
        transform={pix_val: idx for idx, pix_val in enumerate(VOC_COLORMAP)},
        load="rgb",
        train=False,
        ignore=[(224, 224, 192)],
    )
    return dset_path


task_name = "pascal_voc_challenge_2012"


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_voc12_task(dset_path: Path):
    task_creator = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="PASCAL VOC 2012 segmentation challenge images",
        ref=REFERENCE,
        url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html",
        instr="downloaded via torchvision dataset (https://pytorch.org/vision/stable/datasets.html#voc)",
        lic=License.UNKNOWN,
        release="2012",
        keywords=[Keyword.NATURAL_OBJECTS],
    )

    train_iterator = get_iterator_from_segmentation_dataset(
        images_root=dset_path / DataKind.TRAINING_DATA,
        masks_root=dset_path / DataKind.TRAINING_LABELS / "transformed_masks",
    )
    test_iterator = get_iterator_from_segmentation_dataset(
        images_root=dset_path / DataKind.TESTING_DATA,
        masks_root=dset_path / DataKind.TESTING_LABELS / "transformed_masks",
    )
    idx_to_class = {ix: val for ix, val in enumerate(VOC_CLASSES)}
    task_creator.find_data(train_iterator=train_iterator, test_iterator=test_iterator, idx_to_class=idx_to_class)
    task_creator.split_folds(n_folds=5)
    task_creator.infer_stats()
    task_creator.push_and_test()
