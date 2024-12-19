# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from dataclasses import dataclass
from typing import List, Union

from mml.core.scripts.utils import StrEnum

# these are the kind of values assigned to modalities in the task description samples
ModalityEntry = Union[int, List[int], List[float], str]

# these are the acceptable keywords for kornia augmentations
EMPTY_MASK_TOKEN = "EMPTY_MASK_TOKEN"


class Modality(StrEnum):
    """
    The modalities represent the possible keys of a loaded sample from a dataset. E.g. {'image': 'some/path/file.png',
    'class': 3}. Note that while TaskDescription stores the Modality as enum in its samples, the loaded batch will
    contain the str representations!
    """

    # supported types
    IMAGE = "image"  # (str) default RGB image
    MASK = "mask"  # (str) grayscale mask, used e.g. in instance or semantic segmentation
    CLASS = "class"  # (int) multi or binary whole image classification label
    CLASSES = "classes"  # (List[int]) multi-label classification labels
    SOFT_CLASSES = "soft_classes"  # (List[float]) allows soft labels [0, 1] for a multi-class / multi-label setup
    VALUE = "value"  # (float) regression target value
    # future possibilities, not supported yet
    BBOX = "bbox"
    KEYPOINTS = "keypoints"
    VIDEO_CLIP = "video_clip"
    THREE_D_IMAGE = "three_d_image"
    # meta information, not necessarily given
    TASK = "task"  # (str) the name of the task the sample is loaded from
    SAMPLE_ID = "sample_id"  # (str) the id of the sample


class TaskType(StrEnum):
    """
    Defines the type of task. Different task types usually require completely different architectures and/or training
    procedures. This is aligned with the torchvision.models split at
    https://pytorch.org/docs/stable/torchvision/models.html
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    DETECTION = "detection"
    VIDEO_CLASS = "video_classification"
    NO_TASK = "no_task"  # for completely unlabeled datasets
    UNKNOWN = "unknown"
    DEFAULT = "unknown"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"

    def requires(self) -> List[List[Modality]]:
        """
        Returns the necessary modalitie(s) for this kind of task. First list level is OR and second level is AND. So
        if returns [[A, B], [C, D]] either [A and B] or [C and D] are required.
        """
        assignment = {
            TaskType.CLASSIFICATION: [[Modality.CLASS, Modality.IMAGE]],
            TaskType.SEMANTIC_SEGMENTATION: [[Modality.MASK, Modality.IMAGE]],
            TaskType.MULTILABEL_CLASSIFICATION: [
                [Modality.CLASSES, Modality.IMAGE],
                [Modality.SOFT_CLASSES, Modality.IMAGE],
            ],
            TaskType.NO_TASK: [[Modality.IMAGE]],
            TaskType.REGRESSION: [[Modality.IMAGE, Modality.VALUE]],
        }
        if self in assignment:
            return assignment[self]
        else:
            return [[]]


class Keyword(StrEnum):
    """
    Keyword labels of a task. Refers e.g. to the shown entities within the images.
    """

    # domains
    MEDICAL = "medical"
    ANIMALS = "animals"
    BUILDINGS = "buildings"
    ARTIFICIAL = "artificial"
    NATURAL_OBJECTS = "natural_objects"
    HANDWRITINGS = "handwritings"
    SCENES = "scenes"
    FACES = "faces"
    DRIVING = "driving"
    DERMATOSCOPY = "dermatoscopy"
    CATARACT_SURGERY = "cataract_surgery"
    LARYNGOSCOPY = "laryngoscopy"
    LAPAROSCOPY = "laparoscopy"
    GASTROSCOPY_COLONOSCOPY = "gastroscopy_colonoscopy"
    ENDOSCOPY = "endoscopy"
    NEPHRECTOMY = "Nephrectomy"
    FUNDUS_PHOTOGRAPHY = "fundus_photography"
    ULTRASOUND = "ultrasound"
    MRI_SCAN = "mri_scan"
    X_RAY = "x_ray"
    CT_SCAN = "ct_scan"
    CLE = "confocal laser endomicroscopy"
    CAPSULE_ENDOSCOPY = "capsule endoscopy"
    COLPOSCOPY = "colposcopy"
    # task type
    ENDOSCOPIC_INSTRUMENTS = "endoscopic instruments"
    INSTRUMENT_COUNT = "counting endoscopic instruments"
    ANATOMICAL_STRUCTURES = "anatomical structures"
    TISSUE_PATHOLOGY = "tissue_pathology"
    IMAGE_ARTEFACTS = "image_artefacts"
    CHARS_DIGITS = "chars_or_digits"
    # body locations
    CHEST = "chest"
    BRAIN = "brain"
    EYE = "eye"
    BREAST = "breast"
    BONE = "bone"
    GYNECOLOGY = "Gynecology"


class License(StrEnum):
    """
    License for distribution of a task (data).
    """

    # there is no such thing as no license -> strict limitations apply here!
    UNKNOWN = "unknown"
    # Use when license is specific to data set
    CUSTOM = "License defined in TaskCreator description"
    # https://creativecommons.org/licenses/by-nc/4.0/
    CC_BY_NC_4_0 = "Creative Commons Attribution-NonCommercial 4.0 International"
    # https://creativecommons.org/licenses/by/4.0/
    CC_BY_4_0 = "Creative Commons Attribution 4.0 International"
    # https://creativecommons.org/licenses/by-nc-sa/4.0/
    CC_BY_NC_SA_4_0 = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"
    # https://opendatacommons.org/licenses/dbcl/1-0/
    DATABASE_CONTENTS_LICENSE_1_0 = "Open Data Commons DbCL v1.0"
    # https://creativecommons.org/publicdomain/zero/1.0/
    CC_0_1_0 = "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"
    # https://choosealicense.com/licenses/mit/
    MIT = "Massachusetts Institute of Technology"


@dataclass
class Sizes:
    """
    Small dataclass storing information about the dimensionality of a set of images.
    """

    min_height: int = 0
    max_height: int = 0
    min_width: int = 0
    max_width: int = 0

    def to_list(self) -> List[int]:
        return [self.min_height, self.max_height, self.min_width, self.max_width]


@dataclass
class RGBInfo:
    """
    Small dataclass storing information about image channels (mostly mean and std).
    """

    r: float = 0.0
    g: float = 0.0
    b: float = 0.0

    def get_rgb(self) -> List[float]:
        return [self.r, self.g, self.b]

    def to_list(self) -> List[float]:
        return self.get_rgb()


IMAGENET_MEAN = RGBInfo(0.485, 0.456, 0.406)
IMAGENET_STD = RGBInfo(0.229, 0.224, 0.225)


class DataSplit(StrEnum):
    """
    Represents parts of a dataset that are loaded together. May be selected joint by a fold number to determine the
    exact samples that are available for iteration over a :class:`~mml.core.data_loading.task_dataset.TaskDataset`.
    """

    TRAIN = "TRAIN"
    FULL_TRAIN = "FULL_TRAIN"
    VAL = "VAL"
    TEST = "TEST"
    UNLABELLED = "UNLABELLED"
