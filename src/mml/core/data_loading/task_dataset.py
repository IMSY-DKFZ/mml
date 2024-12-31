# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import ctypes
import logging
import multiprocessing as mp
import warnings
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from mml.core.data_loading.augmentations.augmentation_module import AugmentationModule, AugmentationModuleContainer
from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.data_loading.modality_loaders import DEFAULT_MODALITY_LOADERS, ModalityLoader
from mml.core.data_loading.task_attributes import EMPTY_MASK_TOKEN, DataSplit, Modality, TaskType
from mml.core.data_loading.task_description import SampleDescription

logger = logging.getLogger(__name__)
_ERROR_RETRY = 0  # may be set higher to allow for corrupted samples


class TaskDataset(Dataset):
    """
    The TaskDataset class represents a loadable dataset, handling folds, data loading, different modalities of a task
    as well as non-batched transforms. After initialization, it may be directly given to some (multithreaded)
    dataloader.
    """

    def __init__(
        self,
        root: Union[Path, str],
        split: DataSplit = DataSplit.TRAIN,
        fold: int = 0,
        transform: Optional[Union[AugmentationModule, AugmentationModuleContainer]] = None,
        caching_limit: int = 0,
        loaders: Optional[Dict[Modality, ModalityLoader]] = None,
    ):
        """
        The TaskDataset initialization loads all meta information on the task and selects active split + fold. This
        choice can later be changed by the 'select_samples' method.

        :param Path root: Path to TASKXXX_name.json file of task to load.
        :param DataSplit split: one of 'train', 'val', 'full_train' and 'test'
        :param int fold: irrelevant if 'test' split, inactive fold in 'train' split and only active fold in 'val' split
        :param Optional[A.Compose] transform: :mod:albumentation compose transform to be applied on samples
        :param int caching_limit: this corresponds to the number of max images cached
        :param Optional[Dict[Modality, ModalityLoader]]: a dict of ModalityLoaders for this task, if None are given a
            default set of loaders is used
        """
        # class basics
        self.root = Path(root)
        # load and parse meta information
        self.task_type = None
        self.raw_idx_to_class = None
        self.classes = None
        self.modalities: Optional[Dict[Modality, str]] = None
        self.class_occ = None
        self.samples: List[SampleDescription] = []
        self._sample_ids: List[str] = []
        self.task_description = MMLFileManager.load_task_description(self.root)
        self._parse_meta()
        # caching option variables init
        self.caching_limit = caching_limit
        self.allow_caching = caching_limit > 0
        self.shared_array = None
        self._use_cache: bool = False  # after cache has been created and filled enable via self.enable_cache
        # prepare data loading
        self.transform = transform
        self._consecutive_errors: int = 0
        # setup loaders
        if loaders is None:
            # select default loaders for backward compatibility
            loaders = {mod: DEFAULT_MODALITY_LOADERS[mod]() for mod in self.modalities}
        self.loaders = loaders
        if any(mod not in self.loaders for mod in self.modalities):
            raise ValueError("No loader found for some modality.")
        for loader in self.loaders.values():
            loader.setup(self)
        # select samples (done last, triggers caching if activated)
        self.active_fold: Optional[Tuple[DataSplit, int]] = None
        self.select_samples(split, fold)  # needs to be done after cache variables are set

    def _create_cache(self) -> None:
        """
        Creates the array in memory to store images. Is called after samples have been selected.
        """
        # gather dimensions, therefore loading a sample with temporally disabled transforms
        tmp_bkp = self.transform
        self.transform = None
        # temporary dict, to allow loading without an underlying array
        self.shared_array = {}
        sample_image = self[0][Modality.IMAGE.value]
        h, w, c = sample_image.shape
        # create array
        shared_array_base = mp.Array(ctypes.c_ubyte, len(self) * c * h * w)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        self.shared_array = shared_array.reshape(len(self), h, w, c)
        # array cannot be used (it is still empty!)
        self._use_cache = False
        # resetting transforms
        self.transform = tmp_bkp
        logger.debug(f"Created cache array for {self.task_description.name}.")

    def enable_cache(self) -> None:
        """
        After cache has been created and filled, enable caching to speed up training.
        :return:
        """
        if self.allow_caching:
            logger.info(f"Caching activated for {self.task_description.name}.")
            self._use_cache = True
        else:
            logger.error("Requested cache enabling without allowing cache during dataset init!")

    def disable_cache(self) -> None:
        """
        Deactivates the usage of the internal image cache.
        :return:
        """
        self._use_cache = False
        logger.info(f"Caching DE-activated for {self.task_description.name}.")

    def fill_cache(self, num_workers: int = 0) -> None:
        """

        :return:
        """
        if self.allow_caching:
            self._use_cache = False
            # disable transforms
            tmp_bkp = self.transform
            self.transform = None
            # create a simple dataloader that needs no sampler nor batching / collating /memory pinning / ...
            dl = DataLoader(
                self,
                batch_size=None,
                shuffle=False,
                sampler=None,
                batch_sampler=None,
                num_workers=num_workers,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
                worker_init_fn=None,
                persistent_workers=False,
            )
            # iterate once along dataset
            for _ in tqdm(dl, desc="Caching"):
                pass
            # re-enable transforms
            self.transform = tmp_bkp
            # activate cache usage
            self.enable_cache()
            logger.info(f"Cached {len(self)} samples.")
        else:
            logger.error(
                "No caching allowed, you might need to raise sampling.cache_max_size in the configs to allow "
                "larger datasets caching. Also make sure to set a caching limit greater than zero to this "
                "TaskDataset."
            )

    def __repr__(self) -> str:
        return f"TaskDataSet(root={self.root}, split={self.active_fold[0]}, fold={self.active_fold[1]})"

    def _parse_meta(self) -> None:
        """
        Find, check and load task_type, classes, modalities, class_to_idx, class_occ.

        :return: None
        """
        self.task_type = self.task_description.task_type
        if self.task_type not in TaskType:
            raise RuntimeError(f"Task type {self.task_type}, has to be of type TaskType.")
        # ensure sorted modality to always load image before mask (necessary for EMPTY_MASK_TOKEN)!
        self.modalities = {k: self.task_description.modalities[k] for k in sorted(self.task_description.modalities)}
        if any([mod not in Modality for mod in self.modalities]):
            raise ValueError(f"Invalid modalities in meta_info! Accepted keys must be of type {Modality}!")
        self.raw_idx_to_class = self.task_description.idx_to_class
        # sort by keys (preserves e.g. 0 to be the background class in segmentation)
        self.classes = self.get_classes_from_idx_dict(self.raw_idx_to_class)
        self.class_occ = self.task_description.class_occ
        if len(self.task_description.train_samples) > 0 and len(self.class_occ) != len(self.classes):
            raise RuntimeError("Class occurrences do not match the number of classes.")

    def select_samples(self, split: DataSplit, fold: int) -> None:
        """
        Chooses the actual samples from the task meta information. Handles splits, folds and subsets.

        :param DataSplit split: either 'train', 'val', 'full_train', 'unlabelled' or 'test'
        :param int fold: irrelevant if 'test' split, inactive fold in 'train' split and only active fold in 'val' split
        :return: None
        """
        if 0 > fold or fold > len(self.task_description.train_folds):
            raise ValueError(
                f"Invalid fold number {fold}, has to be in range 0 - " f"{len(self.task_description.train_folds)}."
            )
        if not isinstance(split, DataSplit):
            raise TypeError(f"Invalid split {split}, needs to be compatible to DataSplit class.")
        if split == DataSplit.TEST:
            self.allow_caching = False
            logger.debug(
                "Deactivated caching for test data (commonly passed once and mml assumes not to be " "preprocessed)."
            )
            self.samples = self.task_description.test_samples.values()
            self._sample_ids = list(self.task_description.test_samples.keys())
        elif split == DataSplit.TRAIN:
            data_ids = list(
                chain(
                    *self.task_description.train_folds[0:fold],
                    *self.task_description.train_folds[fold + 1 : len(self.task_description.train_folds) + 1],
                )
            )
            self.samples = [self.task_description.train_samples[data_id] for data_id in data_ids]
            self._sample_ids = data_ids
        elif split == DataSplit.FULL_TRAIN:
            self.samples = self.task_description.train_samples.values()
            self._sample_ids = list(self.task_description.train_samples.keys())
        elif split == DataSplit.UNLABELLED:
            self.samples = self.task_description.unlabeled_samples.values()
            self._sample_ids = list(self.task_description.unlabeled_samples.keys())
        elif split == DataSplit.VAL:
            # val split
            try:
                data_ids = self.task_description.train_folds[fold]
            except IndexError:
                # no val split present
                data_ids = []
            self.samples = [self.task_description.train_samples[data_id] for data_id in data_ids]
            self._sample_ids = list(data_ids)
        else:
            ValueError(f"Was not given any valid DataSplit. Options are: {DataSplit.list()}")
        self.samples = list(self.samples)
        self.active_fold = (split, fold)
        logger.debug(
            f"Selected samples based on split {split} and fold {fold}. " f"Total sample num is {len(self.samples)}."
        )
        if len(self.samples) != len(self._sample_ids):
            raise RuntimeError(f"{len(self.samples)=}  {len(self._sample_ids)=}")
        # each sample selection process requires potential re-caching
        self._use_cache = False
        if self.allow_caching and len(self) > self.caching_limit:
            logger.error("Dataset size exceeds caching limit, will deactivate.")
            self.allow_caching = False
        if self.allow_caching:
            # cache must be recreated after samples have been selected
            self._create_cache()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Main entry point for data loading. Returns loaded and transformed modalities.

        :param index: sample index int
        :return: dict with modality keys and loaded + transformed objects as values
        """
        try:
            sample = self.load_sample(index)
        except Exception as e:
            logger.warning(
                f"Skipped sample (index {index}). Exception: {str(e)}.\n" f"Loading data was: {self.samples[index]}."
            )
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            sample = self.transform(**sample)
        try:
            sample["sample_id"] = self._sample_ids[index]
        except IndexError:
            # sample ids not available in most test setups
            warnings.warn(f"Wanted to look up sample id outside {len(self._sample_ids)} (requested {index})")
            sample["sample_id"] = "NA"
        return sample

    def __len__(self) -> int:
        return len(self.samples)

    def load_sample(self, index: int) -> Dict[str, Any]:
        """
        Loads all necessary components. This based on the active modalities and the information provided there.
        Be aware that for preprocessing the raw_index_mapping is removed by default (set to None). Handle this
        separately.

        :param index: int within range(len(self.samples))
        :return: dict with modality key (str) and obj
        """
        loading_dict: SampleDescription = self.samples[index]
        sample_dict = {}
        # only load modalities requested for this task
        for mod in self.modalities:
            # treat image separately to deal with cache
            if mod == Modality.IMAGE:
                # load the image
                if self._use_cache:
                    # use cache if activated
                    sample_dict[mod] = self.shared_array[index]
                else:
                    # else load image
                    sample_dict[mod] = self.loaders[Modality.IMAGE].load(entry=loading_dict[mod])
                    # and store image in cache if desired (still unmodified from transforms)
                    if self.allow_caching:
                        self.shared_array[index] = sample_dict[mod]
            # do not load anything else but image in case of unlabeled data
            elif self.active_fold[0] == DataSplit.UNLABELLED:
                continue
            # next special case of empty segmentation mask token
            elif mod == Modality.MASK and loading_dict[mod] == EMPTY_MASK_TOKEN:
                # use image as template, but only single channel
                sample_dict[mod] = np.zeros_like(sample_dict[Modality.IMAGE][:, :, 0])
            else:
                # default case, search for applicable modality loader any load entry
                sample_dict[mod] = self.loaders[mod].load(entry=loading_dict[mod])
        # finally all modalities will be represented with their corresponding strings in the loaded batch, this enables
        # usage of kwarg unpacking ("**")
        sample_dict = {mod.value: item for mod, item in sample_dict.items()}
        return sample_dict

    @staticmethod
    def get_classes_from_idx_dict(idx_to_class: Dict[int, str]) -> List[str]:
        """
        Transforms the idx_to_class dict of a task to the actual list of classes.

        :param idx_to_class: index to class mapping as provided in task meta information
        :return: class list, ordered by increasing idx
        """
        if not all([isinstance(k, int) for k in idx_to_class.keys()]):
            raise ValueError("Only integer keys allowed.")
        if not all([isinstance(v, str) for v in idx_to_class.values()]):
            raise ValueError("Only string values allowed.")
        return list(dict.fromkeys([idx_to_class[key] for key in sorted(list(idx_to_class.keys()))]))


class TupelizedTaskDataset(Dataset):
    def __init__(self, task_dataset: TaskDataset, transform: Optional[A.Compose] = None):
        """
        Turns the output of a TaskDataset to tuples (which are dicts by default).
        Also allows to overwrite the transform.

        :param TaskDataset task_dataset: TaskDataset instance
        :param transform: (optional) if not None, overwrites the dataset transform
        """
        self.ds = task_dataset
        self.mod_order = [mod for mod in Modality if mod in self.ds.modalities]
        if transform is not None:
            self.ds.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds.__getitem__(index)
        items = [sample[mod] for mod in self.mod_order]
        return tuple(items)
