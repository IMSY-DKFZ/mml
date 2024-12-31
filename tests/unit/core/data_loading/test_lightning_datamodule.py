# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest
import torch.utils.data
from omegaconf import OmegaConf

from mml.core.data_loading.lightning_datamodule import MultiTaskDataModule
from mml.core.data_loading.task_attributes import Modality
from mml.core.data_loading.task_dataset import TaskDataset
from mml.core.scripts.utils import LearningPhase


def test_datamodule_setup(datamodule):
    datamodule.setup(stage="fit")
    tasks = list(datamodule.task_datasets.keys())
    assert len(tasks) == 1  # one task
    assert len(datamodule.task_datasets[tasks[0]]) == 2  # train and val sets
    assert all([isinstance(ds, TaskDataset) for ds in datamodule.task_datasets[tasks[0]].values()])
    assert all([len(ds) > 0 for ds in datamodule.task_datasets[tasks[0]].values()])


@pytest.mark.parametrize("fold", [ix for ix in range(5)])
def test_get_balancing_weights(fold, dummy_meta_class_path):
    ds = TaskDataset(root=dummy_meta_class_path, fold=fold)
    weights = MultiTaskDataModule.get_dataset_balancing_weights(ds)
    assert weights.size() == torch.Size([len(ds)])
    # weights and class frequency must cancel out each other
    all_mapped_classes = [ds.loaders[Modality.CLASS].load(entry=s[Modality.CLASS]) for s in ds.samples]
    assert all(
        [
            pytest.approx(1.0) == weights[i].item() * ds.class_occ[ds.classes[all_mapped_classes[i]]]
            for i in range(len(ds))
        ]
    )


@pytest.mark.parametrize(
    "cfg_overrides",
    [
        {
            "cpu": {
                "backend": "albumentations",
                "pipeline": [
                    {"name": "RandomCrop", "height": 10, "width": 10},
                    {
                        "name": "Affine",
                        "scale": 0.05,
                        "translate_percent": 0.05,
                        "rotate": 15,
                        "p": 0.5,
                    },
                ],
            }
        },
        {
            "cpu": {
                "backend": "torchvision",
                "pipeline": [
                    {"name": "RandomResizedCrop", "size": [10, 10], "antialias": True},
                    {"name": "RandomHorizontalFlip", "p": 0.5},
                ],
            }
        },
        {
            "gpu": {
                "backend": "torchvision",
                "pipeline": [
                    {"name": "RandomResizedCrop", "size": [10, 10], "antialias": True},
                    {"name": "RandomHorizontalFlip", "p": 0.5},
                ],
            }
        },
        {
            "gpu": {
                "backend": "kornia",
                "pipeline": [
                    {"name": "RandomCrop", "size": [10, 10]},
                    {"name": "RandomAffine", "translate": [0.05, 0.05], "scale": [0.05, 0.05], "degrees": 15, "p": 0.5},
                ],
            }
        },
    ],
)
def test_get_common_train_transforms(datamodule, image, mask, dummy_task_struct, cfg_overrides):
    OmegaConf.update(datamodule.cfg, key="augmentations", value=cfg_overrides, force_add=True)
    if len(datamodule.cfg.augmentations.gpu) > 1:
        datamodule.has_gpu_augs = True
        datamodule.setup(stage="fit")

        # print(datamodule.gpu_train_augs.pipeline({'image': Image(torch.rand(size=(1, 3, 100, 100)))}).size())
        class DummyTrainer:
            training: bool = False

        datamodule.trainer = DummyTrainer

    for phase in [LearningPhase.TRAIN, LearningPhase.VAL, LearningPhase.TEST]:
        if len(datamodule.cfg.augmentations.gpu) > 1:
            datamodule.trainer.training = phase == LearningPhase.TRAIN
        cpu_transforms = datamodule.get_cpu_transforms(phase=phase, struct=dummy_task_struct)
        cpu_transformed = cpu_transforms(image=image, mask=mask)
        batch = torch.utils.data.default_collate([cpu_transformed])
        if phase == LearningPhase.TRAIN:
            # combined loader treats val and test sequentially
            batch = {"dummy_task": batch}
        transformed = datamodule.on_after_batch_transfer(batch=batch, dataloader_idx=0)
        if phase == LearningPhase.TRAIN:
            transformed = transformed["dummy_task"]
        assert isinstance(transformed["image"], torch.Tensor) and isinstance(transformed["mask"], torch.Tensor)
        assert transformed["image"].size()[2:] == transformed["mask"].size()[1:]  # omit batch and channel dims
        if phase == LearningPhase.TRAIN:
            assert transformed["mask"].size()[1:] == torch.Size([10, 10])
        else:
            assert transformed["mask"].size()[1:] != torch.Size([10, 10])
        assert transformed["image"].dtype == torch.float32
