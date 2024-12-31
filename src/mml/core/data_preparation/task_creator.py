# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import functools
import logging
import os
import tempfile
import warnings
from collections import Counter
from datetime import datetime
from itertools import chain, combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import mml.core.scripts.utils  # keep like this to allow monkeypatching load_env fixture
from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.data_loading.task_attributes import (
    EMPTY_MASK_TOKEN,
    DataSplit,
    Keyword,
    License,
    Modality,
    RGBInfo,
    Sizes,
    TaskType,
)
from mml.core.data_loading.task_dataset import TaskDataset
from mml.core.data_loading.task_description import SampleDescription, TaskDescription
from mml.core.data_preparation.utils import TaskCreatorActions, TaskCreatorState, calc_means_stds_sizes
from mml.core.scripts.exceptions import InvalidTransitionError, TaskNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_N_FOLDS = 5
DEFAULT_ENSURE_BALANCED = True


def implements_action(action: TaskCreatorActions):
    """
    This is a decorator to simplify state management of the task creator. It also adds a "secret" <ignore_state>
    kwarg to most task creator methods, if this is set no state check is done.

    :param action: the action that the following function implements
    :return: a decorator
    """

    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # extract task creator instance "self" as first argument of method
            self, *args = args
            traverse = True
            if not isinstance(self, TaskCreator):
                raise TypeError('"implements_action" decorator may only be applied to methods of TaskCreator!')
            if "ignore_state" in kwargs and bool(kwargs["ignore_state"]):
                # no traversal!
                kwargs.pop("ignore_state")
                traverse = False
            if traverse:
                # test if traverse is legal (saves time compared to waiting and raising after func call)
                self._state.traverse(action=action)
            # run and return result
            result = func(self, *args, **kwargs)
            if traverse:
                # do actual transition
                self._state = self._state.traverse(action=action)
            return result

        return wrapper

    return actual_decorator


class TaskCreator:
    """
    Usage:

    (1) Creating a new task:
        (a) instantiate class with all available meta information
        (b) call find_data to locate data of the task
        (c) call split_folds (or use_existing_folds) to prepare folds
        (d) call infer_stats to calc and set means, stds and sizes of train data (use set_stats if already known)
        (e) call push_and_test to save

    (2) Modifying an existing task:
        (a) instantiate with correct dset_path
        (b) call load_existent with existing task_path
        (c) call any of the modification functions (optionally also multiple times)
        (d) if necessary call infer_stats to set means, stds and sizes of train data
        (e) call push_and_test to save

    (3) Auto creation based on task tags:
        (a) instantiate with arbitrary dset_path
        (b) call auto_create_tagged with the full name and also the preprocessing
        (c) raises a RuntimeError if not possible, else will create the task and return the path
    """

    def __init__(
        self,
        dset_path: Path,
        name: str = "default",
        task_type: TaskType = TaskType.UNKNOWN,
        desc: str = "",
        ref: str = "",
        url: str = "",
        instr: str = "",
        lic: License = License.UNKNOWN,
        release: str = "",
        keywords: Optional[List[Keyword]] = None,
    ):
        """
        Everything it needs to create a task. Use a new instance for each new task.

        :param dset_path: path to dataset root
        :param name: name of the task to be created
        :param task_type: task type of the task
        :param desc: a short description of the task
        :param ref: a reference (most likely some bibtex citation)
        :param url: an url linked to the task
        :param instr: instructions to download the task (data)
        :param lic: the license corresponding to the data of the task
        :param release: either a release date or some version of the task
        :param keywords: keywords associated to the task
        """
        # the instance calls correctly detects an existing file manager, but falls back on creating one in case
        # there has not been created one yet
        mml.core.scripts.utils.load_env()
        self.fm = MMLFileManager.instance(
            data_path=Path(os.getenv("MML_DATA_PATH")), proj_path=Path(os.getcwd()), log_path=Path(tempfile.mkdtemp())
        )
        if name in self.fm.task_index.keys():
            logger.warning(f"Task name {name} already used with prepossessings " f"{self.fm.task_index[name].keys()}.")
        if any(
            [symbol in name for symbol in [" ", "%", mml.core.scripts.utils.TAG_SEP, mml.core.scripts.utils.ARG_SEP]]
        ):
            raise ValueError(
                f'The following symbols are not allowed within (raw) task aliases: '
                f'{[" ", "%", mml.core.scripts.utils.TAG_SEP, mml.core.scripts.utils.ARG_SEP]}'
            )
        if name == self.fm.GLOBAL_REUSABLE:
            raise ValueError("Invalid task name!")
        self.current_meta = TaskDescription(
            name=name,
            description=desc,
            reference=ref,
            url=url,
            download=instr,
            license=lic,
            release=release,
            task_type=task_type,
            keywords=[] if keywords is None else keywords,
        )
        self.protocol("Started")
        self.data: Optional[Dict[DataSplit, Dict[str, SampleDescription]]] = None  # stores data paths
        self.dset_path = Path(dset_path)
        # internal state control mechanism
        self._state: TaskCreatorState = TaskCreatorState.INIT

    def __repr__(self):
        return f"TaskCreator(dset_path={self.dset_path}, _state={self._state})"

    def protocol(self, msg: str) -> None:
        """
        Method to log any processing to the creation_protocol of the meta information.

        :param str msg: message to be logged, will be formatted with datetime and appended to the creation_protocol
        :return:
        """
        msg += f" @ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}; "
        self.current_meta.creation_protocol += msg

    @implements_action(TaskCreatorActions.LOAD)
    def load_existent(self, task_path: Path) -> None:
        """
        Loads an existent task .json, useful prior to modifications (e.g. tagging or preprocessing).

        :param Path task_path: path to task .json file
        :return: None
        """
        if task_path.suffix != ".json":
            raise ValueError(f"requires .json file, was given {task_path.suffix}.")
        if not task_path.exists():
            raise FileNotFoundError(f"given task_path {task_path} does not exist!")
        if self.dset_path not in task_path.parents:
            raise ValueError(f"Invalid dset {self.dset_path} with given task path {task_path}.")
        self.current_meta = self.fm.load_task_description(task_path)
        self.protocol(f"Copied from {task_path.name}")
        self.data = None  # ensure no modifications on this attribute, already present in meta info

    @implements_action(TaskCreatorActions.FIND_DATA)
    def find_data(
        self,
        train_iterator: Optional[List[SampleDescription]] = None,
        test_iterator: Optional[List[SampleDescription]] = None,
        unlabeled_iterator: Optional[List[SampleDescription]] = None,
        idx_to_class: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Correctly identifies data tuples yielded by the provided data iterators.

        :param Optional[List[SampleDescription]] train_iterator: all train data, provided as
            dicts with (optional) keys from `~mml.core.data_loading.task_attributes.Modality`, where
            `Modality.SAMPLE_ID` is required and the corresponding value has to be unique.
            Some further potential entries are `Modality.CLASS` with `int` value,
            and `Modality.MASK` with Path to some (greyscale) image, both with vals in idx_to_class.
        :param Optional[List[SampleDescription]] test_iterator:
            test data iterator, same type as train_iterator
        :param Optional[List[SampleDescription]] unlabeled_iterator:
            unlabeled data iterator, same type as train_iterator
        :param Optional[Dict[int, str]] idx_to_class: dict mapping ints to class names (e.g.
            {0 -> background, 1 -> instrument}), may also be non-continuous (e.g. {0 -> background, 3 -> instrument})
            for subclassing or mapping indices to the same class (e.g.
            {0 -> background, 1 -> instrument, 3 -> instrument}) for merging classes, please
            use index 0 for background in segmentation tasks (all unused values will be mapped to 0). In case of only
            unlabelled data this is not necessary (otherwise it is).
        :return: None
        """
        # validate args
        if self.data is not None:
            raise RuntimeError("Adding data to already existent data in TaskCreator is invalid!")
        self.data = {}
        if not train_iterator and not unlabeled_iterator and not test_iterator:
            raise ValueError("Must provide at least one of train, test or unlabeled iterator.")
        if (train_iterator or test_iterator) and not idx_to_class:
            raise ValueError("If given train/test iterator must provide idx_to_class.")
        if idx_to_class and not all(
            [isinstance(key, int) and isinstance(value, str) for key, value in idx_to_class.items()]
        ):
            raise TypeError("idx_to_class must be of type Dict[int, str].")
        if self.current_meta.task_type == TaskType.UNKNOWN:
            raise RuntimeError("Must provide task_type before reading in data.")
        if (
            self.current_meta.task_type
            in [TaskType.CLASSIFICATION, TaskType.SEMANTIC_SEGMENTATION, TaskType.MULTILABEL_CLASSIFICATION]
            and len(set(idx_to_class.values())) < 2
        ):
            raise ValueError("task requires at least 2 classes")
        # scan over iterators
        for iterator, data_split in zip(
            [train_iterator, test_iterator, unlabeled_iterator],
            [DataSplit.FULL_TRAIN, DataSplit.TEST, DataSplit.UNLABELLED],
        ):
            if iterator is None:
                continue
            if len(iterator) == 0:
                raise ValueError(f"{data_split} iterator has length zero")
            data = {}
            if idx_to_class:
                class_occ = {val: 0 for val in set(idx_to_class.values())}
            else:
                class_occ = {}
            for data_dict in tqdm(iterator, desc=f"Scanning {data_split} data"):
                # STEP ONE: validate sample ID
                if not isinstance(data_dict, dict):
                    raise TypeError(f"The {data_split} iterator has to yield dicts.")
                if Modality.SAMPLE_ID not in data_dict:
                    raise ValueError(
                        f"Modality.SAMPLE_ID key not present in some element of iterator {data_split}, "
                        f"element: {data_dict}."
                    )
                if not isinstance(data_dict[Modality.SAMPLE_ID], str):
                    raise TypeError(
                        f"Modality.SAMPLE_ID value to be a string, was given "
                        f"{data_dict[Modality.SAMPLE_ID]} of type {type(data_dict[Modality.SAMPLE_ID])}."
                    )
                data_id = data_dict.pop(Modality.SAMPLE_ID)
                if data_id in data:
                    raise ValueError(f"Modality.SAMPLE_ID values have to be unique, found existing {data_id}.")
                data[data_id] = {}
                # STEP TWO: validate present and required modalities (only for train)
                modalities = data_dict.keys()
                if any([mod not in Modality for mod in modalities]):
                    raise TypeError(
                        f"iterator dicts have to provide keys from within {Modality}, was given " f"{modalities}."
                    )
                if data_split != DataSplit.UNLABELLED:
                    # check if required modalities are present in train & test data
                    if not any(
                        [
                            all([mod in modalities for mod in mod_list])
                            for mod_list in self.current_meta.task_type.requires()
                        ]
                    ):
                        raise RuntimeError(
                            f"For task type {self.current_meta.task_type} at least one of the "
                            f"following combinations of modalities must be fully provided for any "
                            f"sample: {self.current_meta.task_type.requires()}."
                        )
                # STEP THREE: validate individual modality entries
                for modality, value in data_dict.items():
                    # SUB-STEP
                    if modality not in self.current_meta.modalities.keys():
                        self.current_meta.modalities[modality] = ""
                    if isinstance(value, Path):
                        if self.dset_path not in value.parents:
                            raise ValueError("Data should always be stored within dataset!")
                        # make path relative to dset_path (saves storage, since redundant)
                        value = value.relative_to(self.dset_path)
                        if value.suffix not in self.current_meta.modalities[modality]:
                            self.current_meta.modalities[modality] += f"{value.suffix}; "
                    self.verify_modality_entry(
                        modality=modality, value=value, idx_to_class=idx_to_class, class_occ=class_occ
                    )
                    if isinstance(value, Path):
                        # write paths as strings into task description
                        value = str(value)
                    data[data_id][modality] = value
            self.data[data_split] = data
            # check if some class is missing in some data split
            if 0 in class_occ.values() and data_split != DataSplit.UNLABELLED:
                warnings.warn(
                    f"classes {[name for name, val in class_occ.items() if val == 0]} "
                    f"not present in data for data split {data_split}!"
                )
            if data_split == DataSplit.FULL_TRAIN:
                # in case of soft labels we produce integer counts anyway for consistency
                class_occ = {k: int(v) for k, v in class_occ.items()}
                self.current_meta.class_occ = class_occ
            logger.info(f"Found {len(data)} items in {data_split}.")
            self.protocol(f"Found {len(data)} items of {data_split}")
        self.current_meta.idx_to_class = idx_to_class

    @implements_action(TaskCreatorActions.SET_FOLDING)
    def split_folds(
        self,
        n_folds: int = DEFAULT_N_FOLDS,
        ensure_balancing: bool = DEFAULT_ENSURE_BALANCED,
        fold_0_fraction: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        """
        Splits the found data into folds for cross validation. It is necessary to call either this or the
        use_existing folds method before infer_stats.

        This method requires the following attributes to be set:
         - self.data[DataSplit.FULL_TRAIN]
         - self.current_meta.task_type
         - self.current_meta.class_occ
         - self.current_meta.idx_to_class

        This method sets the following attributes:
         - self.current_meta.train_folds
         - self.current_meta.train_samples
         - self.current_meta.test_samples
         - self.current_meta.unlabeled_samples
         - self.data

        WARNING: The splitting of folds happens deterministic to ensure reproducibility. One implication of this is
        that tasks with identical number of training samples (and identical values for n_folds) will also be split
        identical (with respect to the order of the samples in self.data[DataSplit.FULL_TRAIN]). For classification
        tasks this can be prohibited by using ensure_balancing (since sampling then also happens at class level) or
        in general by using the seed parameter.

        :param n_folds: number of folds to split into
        :param ensure_balancing: indicates if classes should be balanced across folds (only for classification tasks)
        :param Optional[float] fold_0_fraction: if set the first fold (usually used as validation split) will receive
            that fraction of samples, the rest will be distributed evenly across remaining folds. If None all folds
            will have the same size. When a value is provided it must be within (0, 1), but chosen such that least
            one sample (per class if ensure_balancing is active) is contained in each fold.
        :param int seed: controls the determinism behind splitting, default: 42
        :return: None
        """
        if n_folds < 2:
            raise ValueError(f"Splitting requires at least n_folds = 2, was provided {n_folds}.")
        if DataSplit.FULL_TRAIN in self.data:
            if fold_0_fraction is None:
                fold_0_fraction = 1 / n_folds
            if not 0 < fold_0_fraction < 1:
                raise ValueError("fold_0_fraction must be within (0, 1) but cannot be 0 or 1 itself!")
            if ensure_balancing and self.current_meta.task_type != TaskType.CLASSIFICATION:
                warnings.warn(
                    f"ensure_balancing only possible for TaskType classification, have "
                    f"{self.current_meta.task_type}, will be ignored!"
                )
                ensure_balancing = False
            if ensure_balancing:
                # check at least one sample of the smallest class is contained in each fold
                smallest_class, smallest_occ = min(self.current_meta.class_occ.items(), key=lambda x: x[1])
                all_occs = sum(self.current_meta.class_occ.values())
                if smallest_occ / all_occs * len(self.data[DataSplit.FULL_TRAIN]) * fold_0_fraction < 1:
                    raise ValueError(
                        f"fold_0_fraction of {fold_0_fraction} results in no sample of class {smallest_class} "
                        f"in fold 0. It is necessary to increase value of fold_0_fraction, reduce number "
                        f"of folds or disable balancing. {smallest_occ=} {all_occs=} {len(self.data[DataSplit.FULL_TRAIN])}"
                    )
                if (
                    smallest_occ
                    / all_occs
                    * len(self.data[DataSplit.FULL_TRAIN])
                    * (1 - fold_0_fraction)
                    / (n_folds - 1)
                    < 1
                ):
                    raise ValueError(
                        f"fold_0_fraction of {fold_0_fraction} results in no sample of class {smallest_class} "
                        f"in at least one of the folds [1, ..., n_folds]. It is necessary to decrease value of"
                        f" fold_0_fraction, reduce number of folds or disable balancing."
                    )
                # define class pools
                class_pools = {
                    class_ix: [
                        k for k, elem in self.data[DataSplit.FULL_TRAIN].items() if elem[Modality.CLASS] == class_ix
                    ]
                    for class_ix in self.current_meta.idx_to_class.keys()
                }
            else:
                # check that at least one sample is contained in each fold
                if fold_0_fraction * len(self.data[DataSplit.FULL_TRAIN]) < 1:
                    raise ValueError(
                        f"fold_0_fraction of {fold_0_fraction} results in no sample in fold 0. It is "
                        f"necessary to increase value of fold_0_fraction or reduce number of folds."
                    )
                if (1 - fold_0_fraction) * len(self.data[DataSplit.FULL_TRAIN]) < n_folds - 1:
                    raise ValueError(
                        f"fold_0_fraction of {fold_0_fraction} results in no sample in at least one of the "
                        f"folds [1, ..., n_folds]. It is necessary to decrease value of fold_0_fraction."
                    )
                # use a single pseudo class pool instead
                class_pools = {"all": list(self.data[DataSplit.FULL_TRAIN].keys())}
            # do the actual splitting, first every pool itself is split
            splits = {}
            for pool_id, pool in class_pools.items():
                length_fold_0 = round(fold_0_fraction * len(pool))
                mod = (len(pool) - length_fold_0) % (n_folds - 1)
                lengths = [length_fold_0] + [
                    (len(pool) - length_fold_0) // (n_folds - 1) + offset
                    for offset in [1] * mod + [0] * (n_folds - 1 - mod)
                ]
                splits[pool_id] = torch.utils.data.random_split(
                    pool, lengths, generator=torch.Generator().manual_seed(seed)
                )
            # now join pools for each fold
            joined_pools = list()
            for fold_idx in range(n_folds):
                # pools are concatenated
                joined = list(chain(*[list(splits[pool_id][fold_idx]) for pool_id in class_pools.keys()]))
                # now pools are shuffled, so that no class ordering happens (in case balancing was chosen)
                # seed computation is for backward compatibility
                np.random.default_rng(seed=3 * (seed - 1) + fold_idx).shuffle(joined)
                joined_pools.append(joined)
            # finally store in metadata
            self.current_meta.train_folds = joined_pools
            self.current_meta.train_samples = self.data[DataSplit.FULL_TRAIN]
        self.current_meta.test_samples = self.data[DataSplit.TEST] if DataSplit.TEST in self.data else {}
        self.current_meta.unlabeled_samples = (
            self.data[DataSplit.UNLABELLED] if DataSplit.UNLABELLED in self.data else {}
        )
        self.protocol(f"Split into {n_folds} folds")
        self.data = None

    @implements_action(TaskCreatorActions.SET_FOLDING)
    def use_existing_folds(self, fold_definition: List[List[str]]) -> None:
        """
        Replacement for the split_folds function in case there are already predefined folds.

        :param List[List[str]] fold_definition: list of lists of data ids, each list within the main list represents
            one fold, data ids must match the ones provided to find_data
        :return: None
        """
        logger.debug("Checking provided folds compatibility ...")
        assert DataSplit.FULL_TRAIN in self.data, "Call find data on train type before calling fold splitting."
        assert 2 <= len(fold_definition), f"Splitting requires at least 2 folds, was provided {len(fold_definition)}."
        # check folds are disjoint, themselves and across
        set_folds = [set(fold) for fold in fold_definition]
        for ix in range(len(fold_definition)):
            assert len(set_folds[ix]) == len(fold_definition[ix]), f"Duplicate ids in fold {ix} (zero indexed)"
        for fold_1, fold_2 in combinations(set_folds, 2):
            assert fold_1.isdisjoint(fold_2), "Shared ids between folds!"
        # check folds are valid keys
        valid_ids = set(self.data[DataSplit.FULL_TRAIN].keys())
        all_ids = set(chain(*fold_definition))
        assert all([data_id in valid_ids for data_id in all_ids]), "Invalid id provided!"
        # check if ids are unused
        assert all(
            [data_id in all_ids for data_id in valid_ids]
        ), "There are unused ids, please provide complete folderization or decrease data provided to find_data."
        logger.debug("Provided folds are compatible!")
        self.current_meta.train_folds = fold_definition
        self.current_meta.train_samples = self.data[DataSplit.FULL_TRAIN]
        self.current_meta.test_samples = self.data[DataSplit.TEST] if DataSplit.TEST in self.data else {}
        self.current_meta.unlabeled_samples = (
            self.data[DataSplit.UNLABELLED] if DataSplit.UNLABELLED in self.data else {}
        )
        self.protocol(f"Manually provided {len(fold_definition)} folds")
        self.data = None

    @implements_action(TaskCreatorActions.SET_STATS)
    def infer_stats(
        self,
        sizes: bool = True,
        mean_and_std: bool = True,
        const_size: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Calculates the training data stats. For improved speed indicating a constant size of images helps.

        :param bool sizes: if the image sizes should be gathered
        :param bool mean_and_std: if the mean and standard deviation of color channels should be gathered
        :param bool const_size: if images are known to have constant size, this helps flag improves speed
        :param torch.device device: if provided use this device, otherwise infer device based on availability
        :return: None
        """
        assert self.data is None, "Call split_folds/use_existing_folds before calling infer stats."
        temp_json = self.dset_path / "temp.json"
        counter = 0
        while temp_json.exists():
            logger.warning(f"Temp file {temp_json} detected, will try to avoid collision with other runs on data.")
            temp_json = self.dset_path / f"temp_{counter}.json"
            counter += 1
        temp_json.touch(exist_ok=False)
        self.fm.write_task_description(path=temp_json, task_description=self.current_meta, omit_warning=True)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stats = calc_means_stds_sizes(
            temp_json, sizes=sizes, means=mean_and_std, stds=mean_and_std, device=device, const_size=const_size
        )
        if sizes:
            self.current_meta.sizes = stats["sizes"]
        if mean_and_std:
            self.current_meta.stds = stats["stds"]
            self.current_meta.means = stats["means"]
        temp_json.unlink()
        self.protocol(f"Inferred stats {'sizes' if sizes else ''} {'mean and std' if mean_and_std else ''}")

    @implements_action(TaskCreatorActions.SET_STATS)
    def set_stats(
        self, means: Optional[RGBInfo] = None, stds: Optional[RGBInfo] = None, sizes: Optional[Sizes] = None
    ) -> None:
        """
        Alternative to :meth:`infer_stats` with provided means, stds and sizes. May also only set a subset of those (or
        even None).

        :param Optional[RGBInfo] means: task RGB channel means
        :param Optional[RGBInfo] stds: task RGB channel stds
        :param Optional[Sizes] sizes: task image dimensions
        :return: None
        """
        if self.data is not None:
            raise RuntimeError("Call split folds before calling set stats.")
        if means:
            self.current_meta.means = means
        if stds:
            self.current_meta.stds = stds
        if sizes:
            self.current_meta.sizes = sizes
        self.protocol(f"Set stats {'sizes' if sizes else ''} {'mean' if means else ''} {'std' if stds else ''}")

    @implements_action(TaskCreatorActions.FINISH)
    def push_and_test(self) -> Path:
        """
        Final step of task creation. Flushes the created task description and runs a test to load it.

        :return: the path of the written .json task description
        :rtype: `Path`
        """
        if len(self.current_meta.train_samples) > 0 and set(self.current_meta.idx_to_class.values()) != set(
            self.current_meta.class_occ.keys()
        ):
            raise RuntimeError("Inconsistent classes across idx_to_class and class_occ")
        if self.current_meta.task_type == TaskType.CLASSIFICATION and sum(self.current_meta.class_occ.values()) != len(
            self.current_meta.train_samples
        ):
            raise RuntimeError("Class occurrences are incorrect.")
        if self.current_meta.name == "default":
            raise RuntimeError("Task Creator was given no task name!")
        path = self.fm.get_task_path(dset_path=self.dset_path, task_alias=self.current_meta.name)
        if path.exists():
            raise FileExistsError(f"Overwriting meta info at {path} not supported!")
        self.protocol(f"Finished as {self.current_meta.name}")
        self.fm.write_task_description(path, self.current_meta)
        # decide on existing split
        if self.current_meta.train_samples:
            split = DataSplit.TRAIN
        elif self.current_meta.unlabeled_samples:
            split = DataSplit.UNLABELLED
        else:
            split = DataSplit.TEST
        self.current_meta = None  # prohibit reuse
        # check loading
        logger.info(f"Testing the loading of {path}...")
        with mml.core.scripts.utils.catch_time() as ds_timer:
            ds = TaskDataset(root=path, split=split)
        with mml.core.scripts.utils.catch_time() as sample_timer:
            _ = ds[0]
        logger.info(
            f"Testing of {path} finished, dataset loading time was {ds_timer.elapsed:5.2f} seconds, "
            f"sample loading time was {sample_timer.elapsed:5.2f} seconds."
        )
        return path

    def auto_complete(self) -> Path:
        """
        Shortcut for finishing task creation.

        :return: the task path as returned by :meth:`push_and_test`.
        """
        if self._state == TaskCreatorState.DATA_FOUND:
            self.split_folds()
        if self._state == TaskCreatorState.FOLDS_SPLIT:
            self.infer_stats()
        if self._state == TaskCreatorState.STATS_SET:
            return self.push_and_test()
        raise InvalidTransitionError(f"Auto-complete not available with state {self._state}.")

    # TASK MODIFICATION
    @implements_action(TaskCreatorActions.NONE)
    def identity(self, *args) -> None:
        """
        Dummy tag to create identical instances of a task. In contrast to naming the same task twice in the task_list,
        which will load from the same .json task description the identity tagged version creates a fresh task
        description.

        :param args: all args are ignored
        :return: None
        """
        self.protocol(f"Identity with {args}")

    @implements_action(TaskCreatorActions.SET_STATS)
    def nested_validation(self, fold_str: str, new_folds_str: str = "5") -> None:
        """
        This tag will create a nested task, useful for cross-validation techniques. It drops any previous test samples,
        and re-declares the specified fold as new test data. Afterward, the remaining train samples are re-shuffled and
        distributed into new folds according to the new_folds argument.

        :param str fold_str: the fold to be re-declared as test data
        :param str new_folds_str: the number of new folds created from the remaining train data
        :return: None
        """
        # log
        logger.info(f"Nesting data of task {self.current_meta.name} with {fold_str}.")
        self.protocol(f"Nested data with fold {fold_str}.")
        # check args
        fold_int = int(fold_str)
        new_folds_int = int(new_folds_str)
        old_folds_n = len(self.current_meta.train_folds)
        if fold_int < 0 or fold_int >= old_folds_n:
            raise ValueError(f"Invalid fold specified: {fold_int}. Must be in range [0, {old_folds_n - 1}].")
        if new_folds_int < 2:
            raise ValueError(f"Must at least create 2 folds for the nested task. Was given {new_folds_int}.")
        # to the shifting
        train_ids = list(
            chain(*[self.current_meta.train_folds[fold_ix] for fold_ix in range(old_folds_n) if fold_ix != fold_int])
        )
        test_ids = self.current_meta.train_folds[fold_int]
        self.data = {
            DataSplit.FULL_TRAIN: {s_id: self.current_meta.train_samples[s_id] for s_id in train_ids},
            DataSplit.TEST: {s_id: self.current_meta.train_samples[s_id] for s_id in test_ids},
        }
        # update class occurrences
        self.current_meta.class_occ = Counter(
            [
                self.current_meta.idx_to_class[self.current_meta.train_samples[s_id][Modality.CLASS]]
                for s_id in train_ids
            ]
        )
        self.split_folds(n_folds=new_folds_int, ensure_balancing=True, ignore_state=True)

    # ON THE FLY TAG SUPPORT (these are modification keywords in the name)
    @staticmethod
    def auto_create_tagged(full_alias: str, preprocessing: str = "none") -> Path:
        # start with a plain task creator
        creator = TaskCreator(Path(""))
        if preprocessing != "none":
            # there must be an existing base version of this tagged task, transfer this to the correct preprocessing
            if "none" not in creator.fm.task_index[full_alias]:
                raise TaskNotFoundError(f"No base task is available for {full_alias}.")
            # load base task
            rel_task_path = creator.fm.task_index[full_alias]["none"]
            creator.dset_path = (creator.fm.data_path / rel_task_path).parent
            creator.load_existent(task_path=creator.fm.data_path / rel_task_path)
            # switch base path
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="mml.core.data_loading.file_manager")
                creator.dset_path = creator.fm.get_dataset_path(raw_path=creator.dset_path, preprocessing=preprocessing)
            # infer stats
            creator.infer_stats()
            creator.protocol(msg=f"Preprocessing transferred from base to {preprocessing}.")
            return_path = creator.push_and_test()
            return return_path
        # second case, there is no base tagged task present yet, check for existing components
        splitted = full_alias.split(mml.core.scripts.utils.TAG_SEP)
        max_ix = -1
        for ix in range(len(splitted)):
            prefix = mml.core.scripts.utils.TAG_SEP.join(splitted[: ix + 1]).strip()
            if prefix in creator.fm.task_index.keys() and "none" in creator.fm.task_index[prefix]:
                max_ix = ix
        if max_ix == -1:
            raise TaskNotFoundError(f"Alias {full_alias} can not be created, because no base was found!")
        # load base task into creator
        base = mml.core.scripts.utils.TAG_SEP.join(splitted[: max_ix + 1]).strip()
        rel_task_path = creator.fm.task_index[base]["none"]
        logger.debug(f"Auto creating tagged task {full_alias} from base {base} in preprocessing {preprocessing}.")
        creator.dset_path = (creator.fm.data_path / rel_task_path).parent
        creator.load_existent(task_path=creator.fm.data_path / rel_task_path)
        # run tag processing on remaining tags
        remaining_tags = full_alias.split(mml.core.scripts.utils.TAG_SEP)[max_ix + 1 :]
        for tag_string in remaining_tags:
            tag_string = tag_string.strip()
            if "." in tag_string:
                raise ValueError("make sure to avoid dot <.> in tags, underscore <_> is used as decimal separator.")
            tag, *values = tag_string.split(mml.core.scripts.utils.ARG_SEP)
            func = creator.map_tag(tag)
            func(*values)
            creator.current_meta.name += f"{mml.core.scripts.utils.TAG_SEP}{tag_string}"
        # complete any remaining inference as well as testing and pushing
        path = creator.auto_complete()
        logger.info(f"Auto created {full_alias} from {base}.")
        return path

    def map_tag(self, tag: str) -> Callable:
        """
        Correct way to resolve a task name tag to the corresponding modifier method.

        :param str tag: a string that can be appended to a task name e.g. 'identity' (appended as '+identity')
        :return:
        """
        if tag not in TASK_CREATOR_TAG_MAP.keys():
            raise ValueError(
                f"Tag presented ({tag}) was invalid! Valid keywords are: {list(TASK_CREATOR_TAG_MAP.keys())}"
            )
        else:
            return getattr(self, TASK_CREATOR_TAG_MAP[tag])

    def verify_modality_entry(
        self, modality: Modality, value: Any, idx_to_class: Dict[int, str], class_occ: Dict[str, int]
    ) -> None:
        """
        Extendable method to verify that the entries of a modality are well formatted. Extracts a potential verificator
        from the global `MODALITY_VERIFIER_MAP` dictionary and runs the verificator. To support new modalities modify
        this global dictionary.

        :param modality:
        :param value:
        :param idx_to_class:
        :param class_occ:
        :return:
        """
        if modality not in MODALITY_VERIFIER_MAP:
            logger.debug(f"Skipped verification of modality {modality}.")
            return
        verifier_func = MODALITY_VERIFIER_MAP[modality]
        verifier_func(creator=self, value=value, idx_to_class=idx_to_class, class_occ=class_occ)
        logger.debug(f"Verified modality {modality}.")


# this is the map that resolves tagged task aliases to the respective modifiers
# use the mml-tags plugin to enable more tags
TASK_CREATOR_TAG_MAP = {"identity": "identity", "nested": "nested_validation"}


def verify_class_modality(
    creator: TaskCreator, value: Any, idx_to_class: Dict[int, str], class_occ: Dict[str, int]
) -> None:
    if not isinstance(value, int):
        raise TypeError(f"Provide int values for Modality.CLASS key instead of {type(value)}.")
    if value not in idx_to_class:
        raise ValueError(f"class value {value} not present in idx_to_class")
    class_occ[idx_to_class[value]] += 1


def verify_classes_modality(
    creator: TaskCreator, value: Any, idx_to_class: Dict[int, str], class_occ: Dict[str, int]
) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"Provide tuple values for Modality.CLASSES key instead of {type(value)}.")
    if any([element not in idx_to_class for element in value]):
        raise ValueError(f"some class of {value} not present in idx_to_class")
    for elem in value:
        class_occ[idx_to_class[elem]] += 1


def verify_softclasses_modality(
    creator: TaskCreator, value: Any, idx_to_class: Dict[int, str], class_occ: Dict[str, int]
) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"Provide tuple values for Modality.SOFT_CLASSES key instead of " f"{type(value)}.")
    if len(value) != len(idx_to_class):
        ValueError(f"length of soft labels {value} does not match idx_to_class")
    for elem_idx, elem in enumerate(value):
        class_occ[idx_to_class[elem_idx]] += elem


def verify_value_modality(
    creator: TaskCreator, value: Any, idx_to_class: Dict[int, str], class_occ: Dict[str, int]
) -> None:
    if not isinstance(value, float):
        raise TypeError(f"Provide float values for Modality.VALUE key instead of {type(value)}.")
    if min(idx_to_class) > value or max(idx_to_class) < value:
        raise ValueError(f"value {value} must be in bounds of idx_to_class (provide min/max)")
    if value.is_integer() and class_occ:
        # it might be possible that we can gather class occ
        try:
            class_occ[idx_to_class[int(value)]] += 1
        except KeyError:
            warnings.warn(
                f"Tried to create a class occurrence map for regression task, since "
                f"discovered an integer as value, but value is not present in "
                f"idx_to_class, value was {value}, will skip class occurrence gathering"
            )
            class_occ.clear()  # delete entries to prevent constant warnings


def verify_mask_modality(
    creator: TaskCreator, value: Any, idx_to_class: Dict[int, str], class_occ: Dict[str, int]
) -> None:
    if value == EMPTY_MASK_TOKEN:
        # this is the special case of an empty mask
        mask = np.zeros(shape=(1, 1), dtype=np.intc)
    else:
        if not isinstance(value, Path):
            raise TypeError(f"Masks should be provided as paths, not {type(value)}.")
        mask = cv2.imread(str(creator.dset_path / value), cv2.IMREAD_GRAYSCALE)
    for pixel_val in np.unique(mask):
        # skip the ignore value of 255
        if pixel_val == 255:
            continue
        if pixel_val not in idx_to_class:
            raise ValueError(f"mask value {pixel_val} not in idx_to_class.")
        class_occ[idx_to_class[pixel_val]] += 1


MODALITY_VERIFIER_MAP = {
    Modality.CLASS: verify_class_modality,
    Modality.CLASSES: verify_classes_modality,
    Modality.SOFT_CLASSES: verify_softclasses_modality,
    Modality.VALUE: verify_value_modality,
    Modality.MASK: verify_mask_modality,
}
