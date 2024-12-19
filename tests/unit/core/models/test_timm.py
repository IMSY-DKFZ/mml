# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest
import torch

from mml.core.data_loading.task_attributes import Modality, RGBInfo, Sizes, TaskType
from mml.core.data_loading.task_struct import TaskStruct
from mml.core.models.timm import TimmGenericModel


@pytest.mark.parametrize("model_name", ["resnet18", "vit_tiny_r_s16_p8_224"])
def test_timm_freeze_unfreeze(model_name, dummy_task_struct):
    model = TimmGenericModel(name=model_name, pretrained=True, drop_rate=0.5)
    model.add_head(task_struct=dummy_task_struct)
    model.freeze_backbone()
    assert not next(model.backbone.parameters()).requires_grad
    assert next(model.heads[dummy_task_struct.name].parameters()).requires_grad
    model.unfreeze_backbone()
    assert next(model.backbone.parameters()).requires_grad


def test_timm_multi_heads():
    model = TimmGenericModel(name="resnet18", pretrained=True, drop_rate=0.5)
    struct_one = TaskStruct(
        name="dummy",
        task_type=TaskType.CLASSIFICATION,
        means=RGBInfo(*[0.1, 0.2, 0.3]),
        stds=RGBInfo(*[0.4, 0.5, 0.6]),
        sizes=Sizes(*[1, 2, 3, 4]),
        relative_root="dummy_root",
        class_occ={"one": 1, "two": 2},
        preprocessed="none",
        keywords=[],
        idx_to_class={0: "one", 1: "two"},
        modalities={Modality.CLASS: "stuff"},
    )
    struct_two = TaskStruct(
        name="dummy_two",
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        means=RGBInfo(*[0.1, 0.2, 0.3]),
        stds=RGBInfo(*[0.4, 0.5, 0.6]),
        sizes=Sizes(*[1, 2, 3, 4]),
        relative_root="dummy_root",
        class_occ={"one": 1, "two": 2, "three": 3},
        preprocessed="none",
        keywords=[],
        idx_to_class={0: "one", 1: "two", 3: "three"},
        modalities={Modality.SOFT_CLASSES: "stuff"},
    )
    struct_three = TaskStruct(
        name="dummy_three",
        task_type=TaskType.REGRESSION,
        means=RGBInfo(*[0.1, 0.2, 0.3]),
        stds=RGBInfo(*[0.4, 0.5, 0.6]),
        sizes=Sizes(*[1, 2, 3, 4]),
        relative_root="dummy_root",
        class_occ={"one": 1, "two": 2},
        preprocessed="none",
        keywords=[],
        idx_to_class={0: "one", 1: "two"},
        modalities={Modality.VALUE: "stuff"},
    )
    model.add_head(struct_one)
    model.add_head(struct_two)
    model.add_head(struct_three)
    result = model(torch.rand((7, 3, 224, 224)))
    assert all(struct.name in result for struct in [struct_one, struct_two, struct_three])
    assert all(isinstance(result[struct.name], torch.Tensor) for struct in [struct_one, struct_two, struct_three])


@pytest.mark.parametrize("model_name", ["resnet18", "vit_tiny_r_s16_p8_224"])
def test_timm_count_parameters(model_name, dummy_task_struct):
    model = TimmGenericModel(name=model_name, pretrained=True, drop_rate=0.5)
    bb_pars = model.count_parameters()["backbone"]
    model.add_head(task_struct=dummy_task_struct)
    _ = model.count_parameters()[dummy_task_struct.name]
    assert bb_pars == model.count_parameters()["backbone"]  # backbone pars do not change with another head
    model.freeze_backbone()
    assert model.count_parameters()["backbone"] == 0
    assert model.count_parameters(only_trainable=False)["backbone"] > 0
