# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import cv2
import numpy as np
import skimage.io
import torch
import torch.nn.functional
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.tv_tensors import Image as TVImage

from mml.core.data_loading.task_attributes import Modality, ModalityEntry, TaskType
from mml.core.scripts.decorators import beta

if TYPE_CHECKING:
    from mml.core.data_loading.task_dataset import TaskDataset

logger = logging.getLogger(__name__)


class ModalityLoader(ABC):
    """
    A modality loader provides the implementation to load entries of the sample dicts for a specific modality.
    """

    def __init__(self, modality: Modality, suffixes: Optional[List[str]], entry_type: Optional[type]):
        """
        This init stores supported file suffixes and the modality. They are used for the default matches implementation.

        :param Modality modality: the modality this loader supports
        :param Optional[List[str]] suffixes: a list of supported file suffixes, used for loader matching. Provide None
            to support "all" suffixes.
        :param Optional[type] entry_type: if given, entries must be of this type to match
        """
        self.modality = modality
        self.suffixes = suffixes
        self.entry_type = entry_type

    @abstractmethod
    def setup(self, task_dataset: "TaskDataset") -> None:
        pass

    @abstractmethod
    def load(self, entry: ModalityEntry) -> Any:
        """
        The load function is the main routine that will be called with the corresponding entry of a samples modality
        within.
        """
        pass

    def matches(self, entry: ModalityEntry) -> bool:
        """
        This method may be used to find the correct loader for a modality. It is given the entry and
        returns True if those can be handled or False if the loader does not support the provided kinds.

        :param ModalityEntry entry: the entry in the sample description corresponding to the modality
        :return: whether the loader accepts or rejects this
        """
        if self.entry_type and not isinstance(entry, self.entry_type):
            return False
        if isinstance(entry, str) and self.suffixes:
            return any(entry.lower().endswith(suf) for suf in self.suffixes)
        return True


class OpenCVImageLoader(ModalityLoader):
    def __init__(self):
        """Default loader for images."""
        self.base_path: Optional[Path] = None  # will be setup later
        super().__init__(modality=Modality.IMAGE, suffixes=[".bmp", ".jpeg", ".jpg", ".png"], entry_type=str)
        # for supported file types see
        # https://docs.opencv.org/4.8.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56

    def load(self, entry: ModalityEntry) -> Any:
        path = self.base_path / entry
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    def setup(self, task_dataset: "TaskDataset") -> None:
        self.base_path = task_dataset.root.parent


class PillowImageLoader(ModalityLoader):
    def __init__(self):
        """Another loader for images."""
        self.base_path: Optional[Path] = None  # will be setup later
        super().__init__(modality=Modality.IMAGE, suffixes=[".bmp", ".jpeg", ".jpg", ".png"], entry_type=str)
        # for supported file types see
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

    def load(self, entry: ModalityEntry) -> Any:
        path = self.base_path / entry
        return self.transform(Image.open(path).convert("RGB"))

    def setup(self, task_dataset: "TaskDataset") -> None:
        self.base_path = task_dataset.root.parent

    @staticmethod
    def transform(image: Image.Image) -> np.ndarray:
        return np.asarray(image)


class AcceleratedPillowImageLoader(PillowImageLoader):
    """
    Extended Pillow ImageLoader based on the article "Fast import of Pillow images to NumPy / OpenCV arrays"
    by Alex Karpinsky, see https://uploadcare.com/blog/fast-import-of-pillow-images-to-numpy-opencv-arrays/
    """

    @staticmethod
    def transform(image: Image.Image) -> np.ndarray:
        image.load()
        # unpack data
        e = Image._getencoder(image.mode, "raw", image.mode)
        e.setimage(image.im)

        # NumPy buffer for the result
        shape, typestr = Image._conv_type_shape(image)
        data = np.empty(shape, dtype=np.dtype(typestr))
        mem = data.data.cast("B", (data.data.nbytes,))

        bufsize, s, offset = 65536, 0, 0
        while not s:
            _, s, d = e.encode(bufsize)
            mem[offset : offset + len(d)] = d
            offset += len(d)
        if s < 0:
            raise RuntimeError("encoder error %d in tobytes" % s)
        return data


class PureTorchvisionImageLoader(ModalityLoader):
    def __init__(self):
        """Another loader for images based purely on torchvision (no pillow)."""
        self.base_path: Optional[Path] = None  # will be setup later
        super().__init__(modality=Modality.IMAGE, suffixes=[".jpeg", ".jpg", ".png"], entry_type=str)
        # for supported file types see
        # https://pytorch.org/vision/stable/generated/torchvision.io.read_image.html#torchvision.io.read_image

    def load(self, entry: ModalityEntry) -> Any:
        path = self.base_path / entry
        return TVImage(read_image(path=str(path), mode=ImageReadMode.RGB), requires_grad=False)

    def setup(self, task_dataset: "TaskDataset") -> None:
        self.base_path = task_dataset.root.parent


class NumpyArrayImageLoader(ModalityLoader):
    def __init__(self):
        """Loads images stored as numpy arrays."""
        self.base_path: Optional[Path] = None  # will be setup later
        super().__init__(modality=Modality.IMAGE, suffixes=[".npy"], entry_type=str)

    @beta("Numpy loading is in beta stage. Not tested yet.")
    def load(self, entry: ModalityEntry) -> Any:
        path = self.base_path / entry
        return np.load(path)

    def setup(self, task_dataset: "TaskDataset") -> None:
        self.base_path = task_dataset.root.parent


class ScikitImageLoader(ModalityLoader):
    def __init__(self):
        """Scikit-Image loader for images."""
        self.base_path: Optional[Path] = None  # will be setup later
        super().__init__(modality=Modality.IMAGE, suffixes=None, entry_type=str)
        # supported file types depend on the plugins available
        # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread

    def load(self, entry: ModalityEntry) -> Any:
        path = self.base_path / entry
        return skimage.io.imread(path)

    def setup(self, task_dataset: "TaskDataset") -> None:
        self.base_path = task_dataset.root.parent


class OpenCVMaskLoader(ModalityLoader):
    """Default loader for segmentation masks. Adds greyscale interpretation and class mapping on top of image
    loading."""

    def __init__(self):
        self.base_path: Optional[Path] = None  # will be setup later
        self.array_mapper = None  # will be setup later
        super().__init__(modality=Modality.MASK, suffixes=[".bmp", ".jpeg", ".jpg", ".png"], entry_type=str)
        # for supported file types see
        # https://docs.opencv.org/4.8.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56

    def load(self, entry: ModalityEntry) -> Any:
        path = self.base_path / entry
        return self.array_mapper[np.asarray(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))]

    def setup(self, task_dataset: "TaskDataset") -> None:
        if task_dataset.task_type != TaskType.SEMANTIC_SEGMENTATION:
            # prevent misuse
            logger.debug(
                f"Will skip index mapper setup for OpenCVMaskLoader since given task type " f"{task_dataset.task_type}."
            )
            return
        self.base_path = task_dataset.root.parent
        dict_mapper = {raw: task_dataset.classes.index(cls) for raw, cls in task_dataset.raw_idx_to_class.items()}
        k = np.array(list(dict_mapper.keys()))
        v = np.array(list(dict_mapper.values()))
        self.array_mapper = np.zeros(256, dtype=v.dtype)
        self.array_mapper[k] = v
        # the 255 integer represents ignored pixels, generally unsure (e.g. boundaries in the VOC dataset)
        self.array_mapper[255] = 255


class NonMappingOpenCVMaskLoader(ModalityLoader):
    """Special loader of masks, that does not implement a class mapping. Used during preprocessing."""

    def __init__(self):
        self.base_path: Optional[Path] = None  # will be setup later
        super().__init__(modality=Modality.MASK, suffixes=[".bmp", ".jpeg", ".jpg", ".png"], entry_type=str)
        # for supported file types see
        # https://docs.opencv.org/4.8.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56

    def load(self, entry: ModalityEntry) -> Any:
        path = self.base_path / entry
        return np.asarray(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))

    def setup(self, task_dataset: "TaskDataset") -> None:
        if task_dataset.task_type != TaskType.SEMANTIC_SEGMENTATION:
            # prevent misuse
            logger.debug(
                f"Will skip index mapper setup for OpenCVMaskLoader since given task type " f"{task_dataset.task_type}."
            )
            return
        self.base_path = task_dataset.root.parent


class ClassLoader(ModalityLoader):
    """Loads a simple class entry."""

    def __init__(self):
        self.index_mapper = None  # will be setup later
        super().__init__(modality=Modality.CLASS, suffixes=None, entry_type=int)

    def load(self, entry: ModalityEntry) -> Any:
        return torch.tensor(self.index_mapper[entry])

    def setup(self, task_dataset: "TaskDataset") -> None:
        if task_dataset.task_type != TaskType.CLASSIFICATION:
            # prevent misuse
            logger.debug(
                f"Will skip index mapper setup for ClassLoader since given task type " f"{task_dataset.task_type}."
            )
            return
        self.index_mapper = {raw: task_dataset.classes.index(cls) for raw, cls in task_dataset.raw_idx_to_class.items()}


class MultiLabelClassLoader(ModalityLoader):
    """Loads multi-label classification entries and one-hot encodes them."""

    def __init__(self):
        self.array_mapper = None  # will be setup later
        self.num_classes = None  # will be setup later
        super().__init__(modality=Modality.CLASSES, suffixes=None, entry_type=list)

    def load(self, entry: ModalityEntry) -> Any:
        labels = torch.tensor(self.array_mapper[entry])
        if labels.numel() == 0:
            # no label present
            return torch.zeros(len(self.num_classes), dtype=torch.float16)
        else:
            return torch.nn.functional.one_hot(labels, num_classes=len(self.num_classes)).sum(
                dim=0, dtype=torch.float16
            )

    def setup(self, task_dataset: "TaskDataset") -> None:
        if task_dataset.task_type != TaskType.MULTILABEL_CLASSIFICATION:
            # prevent misuse
            logger.debug(
                f"Will skip index mapper setup for MultiLabelClassLoader since given task type "
                f"{task_dataset.task_type}."
            )
            return
        self.num_classes = len(task_dataset.classes)
        dict_mapper = {raw: task_dataset.classes.index(cls) for raw, cls in task_dataset.raw_idx_to_class.items()}
        k = np.array(list(dict_mapper.keys()))
        v = np.array(list(dict_mapper.values()))
        self.array_mapper = np.zeros(self.num_classes, dtype=v.dtype)
        self.array_mapper[k] = v


class SoftLabelClassLoader(ModalityLoader):
    """Loads soft-label classification entries."""

    def __init__(self):
        self.matrix_mapper = None  # will be setup later
        super().__init__(modality=Modality.SOFT_CLASSES, suffixes=None, entry_type=list)

    def load(self, entry: ModalityEntry) -> Any:
        return torch.tensor(self.matrix_mapper.dot(np.asarray(entry)))

    def setup(self, task_dataset: "TaskDataset") -> None:
        if task_dataset.task_type != TaskType.MULTILABEL_CLASSIFICATION:
            # prevent misuse
            logger.debug(
                f"Will skip index mapper setup for SoftLabelClassLoader since given task type "
                f"{task_dataset.task_type}."
            )
            return
        num_classes = len(task_dataset.classes)
        dict_mapper = {raw: task_dataset.classes.index(cls) for raw, cls in task_dataset.raw_idx_to_class.items()}
        k = np.array(list(dict_mapper.keys()))
        v = np.array(list(dict_mapper.values()))
        array_mapper = np.zeros(num_classes, dtype=v.dtype)
        array_mapper[k] = v
        # for soft labels we generate a mapping matrix
        self.matrix_mapper = np.zeros((num_classes, len(array_mapper)))
        self.matrix_mapper[array_mapper, np.arange(array_mapper.size)] = 1


class ValueLoader(ModalityLoader):
    """Loads a simple value entry."""

    def __init__(self):
        super().__init__(modality=Modality.VALUE, suffixes=None, entry_type=float)

    def load(self, entry: ModalityEntry) -> Any:
        return torch.tensor(entry).unsqueeze(0)

    def setup(self, task_dataset: "TaskDataset") -> None:
        pass


class CombinedModalityLoader(ModalityLoader):
    """Combines multiple modality loaders to support diverse data setups."""

    def __init__(self, loaders: List[ModalityLoader]):
        if len(loaders) == 0:
            raise ValueError("Must provide at least one loader.")
        _mod = loaders[0].modality
        if any(loader.modality != _mod for loader in loaders):
            raise RuntimeError("Combined modality loaders only supports consistent modality.")
        self._loaders = loaders
        super().__init__(modality=_mod, suffixes=None, entry_type=None)  # implements its own .matches() method

    def load(self, entry: ModalityEntry) -> Any:
        for loader in self._loaders:
            if loader.matches(entry=entry):
                return loader.load(entry=entry)
        raise RuntimeError(f"Unable to find a suitable loader for {entry=}.")

    def setup(self, task_dataset: "TaskDataset") -> None:
        for loader in self._loaders:
            loader.setup(task_dataset=task_dataset)

    def matches(self, entry: ModalityEntry) -> bool:
        return any(loader.matches(entry=entry) for loader in self._loaders)


# these are the backward compatible defaults that will be used in task dataset if no loaders are provided
DEFAULT_MODALITY_LOADERS = {
    Modality.IMAGE: OpenCVImageLoader,
    Modality.MASK: OpenCVMaskLoader,
    Modality.CLASS: ClassLoader,
    Modality.CLASSES: MultiLabelClassLoader,
    Modality.SOFT_CLASSES: SoftLabelClassLoader,
    Modality.VALUE: ValueLoader,
}
