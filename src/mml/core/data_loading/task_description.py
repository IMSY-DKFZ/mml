# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mml.core.data_loading.task_attributes import Keyword, License, Modality, ModalityEntry, RGBInfo, Sizes, TaskType

logger = logging.getLogger(__name__)

# entries required to construct a TaskStruct
STRUCT_REQ_HEADER_KEYS = [
    "name",
    "task_type",
    "keywords",
    "modalities",
    "means",
    "stds",
    "sizes",
    "idx_to_class",
    "class_occ",
]
# plus additional entries that are not yet part of the data
ALL_HEADER_KEYS = STRUCT_REQ_HEADER_KEYS + [
    "description",
    "creation_protocol",
    "reference",
    "url",
    "download",
    "license",
    "release",
]
# now these are all entries
ALL_TASK_DESCRIPTION_KEYS = ALL_HEADER_KEYS + ["unlabeled_samples", "train_folds", "train_samples", "test_samples"]

SampleDescription = Dict[Modality, ModalityEntry]


@dataclass
class TaskDescription:
    """
    A task description holding the meta information on task background as well as the actual links to samples.
    """

    # provided
    name: Optional[str] = None  # renamed from alias
    description: str = ""
    creation_protocol: str = ""
    reference: str = ""
    url: str = ""
    download: str = ""
    license: License = License.UNKNOWN
    release: str = ""
    task_type: TaskType = TaskType.UNKNOWN
    keywords: List[Keyword] = field(default_factory=list)  # renamed from tags
    # inferred
    means: RGBInfo = field(default_factory=RGBInfo)
    stds: RGBInfo = field(default_factory=RGBInfo)
    sizes: Sizes = field(default_factory=Sizes)
    modalities: Dict[Modality, str] = field(default_factory=dict)
    idx_to_class: Dict[int, str] = field(default_factory=dict)
    class_occ: Dict[str, int] = field(default_factory=dict)
    # created
    unlabeled_samples: Dict[str, SampleDescription] = field(default_factory=dict)  # renamed
    train_folds: List[List[str]] = field(default_factory=list)
    train_samples: Dict[str, SampleDescription] = field(default_factory=dict)  # renamed
    test_samples: Dict[str, SampleDescription] = field(default_factory=dict)  # renamed

    def to_json(self) -> Dict[str, Any]:
        """
        Helper to transform a TaskDescription into a json compatible dict.

        :return: (dic) A dictionary without any custom classes as values to be saved in json format.
        """
        json_data = {}
        for key in ALL_TASK_DESCRIPTION_KEYS:
            value = getattr(self, key)
            # transform plain StrEnum to str
            if key in ["task_type", "license"]:
                value = value.value
            # transform listed StrEnum to str
            elif key in ["keywords"]:
                value = [elem.value for elem in value]
            # transform dict StrEnum to str
            elif key in ["modalities"]:
                value = {k.value: v for k, v in value.items()}
            # transform RGBInfo and Sizes
            elif key in ["means", "stds", "sizes"]:
                value = value.to_list()
            # transform nested StrEnum to str
            elif key in ["unlabeled_samples", "train_samples", "test_samples"]:
                value = {
                    top_k: {low_k.value: low_v for low_k, low_v in top_v.items()} for top_k, top_v in value.items()
                }
            # json only allows str keys
            elif key in ["idx_to_class"]:
                value = {str(k): v for k, v in value.items()}
            json_data[key] = value
        return json_data

    @classmethod
    def from_json(cls, data_dict: Dict[str, Any]) -> "TaskDescription":
        """
        Counterpart for the to_json function: Replaces enum values with their entities and creates a TaskDescription.

        :param Dict[str, Any] data_dict: a dictionary without any custom classes as values to be saved in json format.
        :return: a TaskDescription with entries as encoded in the data_dict
        """
        data_dict = deepcopy(data_dict)
        cls_kwargs = {}
        for key in data_dict:
            if key not in ALL_TASK_DESCRIPTION_KEYS:
                raise KeyError(f"Key {key} not part of a TaskDescription!")
            value = data_dict[key]
            # transform plain StrEnum
            if key in ["task_type", "license"]:
                C = {"task_type": TaskType, "license": License}[key]
                value = C(value)
            # transform listed StrEnum
            elif key in ["keywords"]:
                value = [Keyword(elem) for elem in value]
            # transform dict StrEnum
            elif key in ["modalities"]:
                value = {Modality(k): v for k, v in value.items()}
            # transform RGBInfo and Sizes
            elif key in ["means", "stds", "sizes"]:
                C = {"means": RGBInfo, "stds": RGBInfo, "sizes": Sizes}[key]
                value = C(*value)
            # transform nested StrEnum to str
            elif key in ["unlabeled_samples", "train_samples", "test_samples"]:
                value = {
                    top_k: {Modality(low_k): low_v for low_k, low_v in top_v.items()} for top_k, top_v in value.items()
                }
            elif key in ["idx_to_class"]:
                value = {int(k): v for k, v in value.items()}
            cls_kwargs[key] = value
        return cls(**cls_kwargs)

    @property
    def num_samples(self) -> int:
        """
        The number or training (or unlabeled) samples.
        """
        # list lookup is faster than dict length computation
        train_cases = sum(map(len, self.train_folds))
        if train_cases > 0:
            return train_cases
        # we might have an unlabeled task if no train folds are defined
        elif len(self.unlabeled_samples) > 0:
            return len(self.unlabeled_samples)
        # we return zero if neither are found, but log error
        logger.error(f"Asked for number of samples of task {self.name}, but did not find any samples")
        return 0
