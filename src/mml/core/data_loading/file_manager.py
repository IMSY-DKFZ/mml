# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import dataclasses
import logging
import os.path
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import ijson
import orjson
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import Container, DictConfig

from mml.core.data_loading.task_description import (
    ALL_HEADER_KEYS,
    ALL_TASK_DESCRIPTION_KEYS,
    STRUCT_REQ_HEADER_KEYS,
    TaskDescription,
)
from mml.core.scripts.exceptions import MMLMisconfigurationException, TaskNotFoundError
from mml.core.scripts.model_storage import EnsembleStorage, ModelStorage
from mml.core.scripts.utils import Singleton, catch_time

logger = logging.getLogger(__name__)

# prefixes for files / folders representing tasks and datasets
DSET_PREFIX = "DSET"
TASK_PREFIX = "TASK"

# reusable number tag, call mml with reuse.key=projectREUSABLE_NUMBER_TAGinteger to load a specific file number
# (e.g. reuse.ensemble=test_proj#3)
REUSABLE_NUMBER_TAG = "#"

# default strategies to determine file paths for certain computed artefacts
# key : (type, path, enable_numbering, reusable)
DEFAULT_ASSIGNMENTS = {
    "parameters": (ModelCheckpoint, Path("PROJ_PATH") / "PARAMETERS" / "TASK_NAME" / "model.pth", True, True),
    "img_examples": (None, Path("PROJ_PATH") / "IMG_EXAMPLES" / "TASK_NAME" / "examples.png", True, False),
    "blueprint": (Container, Path("PROJ_PATH") / "BLUEPRINTS" / "TASK_NAME" / "blueprint.yaml", True, True),
    "pipeline": (Container, Path("PROJ_PATH") / "PIPELINES" / "TASK_NAME" / "pipeline.yaml", True, True),
    "models": (ModelStorage, Path("PROJ_PATH") / "MODELS" / "TASK_NAME" / "FILE_NAME", True, True),
    "ensemble": (EnsembleStorage, Path("PROJ_PATH") / "ENSEMBLES" / "TASK_NAME" / "FILE_NAME", True, True),
    "predictions": (dict, Path("PROJ_PATH") / "PREDICTIONS" / "TASK_NAME" / "FILE_NAME", True, False),
    "sample_grid": (torch.Tensor, Path("PROJ_PATH") / "PLOTS" / "sample_grid" / "grid.png", True, False),
    "backup": (None, Path("PROJ_PATH") / "BACKUP" / "FILE_NAME", True, False),
    "temp": (None, Path("TEMP_PATH") / "FILE_NAME", True, False),
}


@dataclasses.dataclass
class ReuseConfig:
    """
    Configuration for reusing certain artefacts from other projects or previous runs.
    During runtime this is an OmegaConf DictConfig and may contain additional entries!
    See :meth:`_find_reusables` on how this is used.
    """

    blueprint: Optional[Union[str, List[str]]] = None
    models: Optional[Union[str, List[str]]] = None
    parameters: Optional[Union[str, List[str]]] = None


@dataclasses.dataclass
class RemoveConfig:
    """
    Configuration for removing certain artefacts from at the end of the run.
    During runtime this is an OmegaConf DictConfig and may contain additional entries!
    See :meth:`remove_intermediates` on how this is used.
    """

    img_examples: bool = False
    blueprint: bool = False
    parameters: bool = False
    pipeline: bool = False
    predictions: bool = False
    models: bool = False
    sample_grid: bool = False
    backup: bool = False


class MMLFileManager(Singleton):
    """
    This class keeps track of the file structure of MML. It ensures a consistent checkpointing and loading strategy,
    provides aggregated listings, and handles requests for files.
    """

    # this will store all path assignments see "add_assignment_path" for more details
    _path_assignments = {}
    # will store artefacts not attachable to a specific task
    GLOBAL_REUSABLE = "_TOP_LEVEL_"

    def __init__(
        self,
        data_path: Path,
        proj_path: Path,
        log_path: Path,
        reuse_cfg: Optional[Union[DictConfig, ReuseConfig]] = None,
        remove_cfg: Optional[Union[DictConfig, RemoveConfig]] = None,
    ):
        """
        The file manager is a singleton class and usually is only generated once. Afterward it may be called from
        anywhere via `MMLFileManager.instance()`, refer to :class:`~mml.core.scripts.utils.Singleton` for more
        details on this.

        :param ~pathlib.Path data_path: path to MML data
        :param ~pathlib.Path proj_path: path to current experiment root
        :param ~pathlib.Path log_path: path to current experiment run root
        :param ReuseConfig reuse_cfg: (optional) a configuration on which files of the project should be reused
        :param RemoveConfig remove_cfg: (optional) a configuration on which files of the project should be deleted
        """
        # base paths
        self.data_path = data_path  # rather large files storage -> stores datasets
        self.proj_path = proj_path  # rather small and aggregated files -> stores exp results
        self.log_path = log_path  # subdir of proj path representing current experiment run
        if any([not path.exists() for path in [self.data_path, self.proj_path.parent]]):
            raise MMLMisconfigurationException(
                f"Some path was not found! Ensure existing:\n - {self.data_path}\n"
                f" - {self.proj_path.parent}\n Please check your "
                f"mml.env file or create these paths!"
            )
        if REUSABLE_NUMBER_TAG in self.proj_path.stem:
            raise MMLMisconfigurationException(f"project name must not contain {REUSABLE_NUMBER_TAG}!")
        # data path file system
        self.raw_data = self.data_path / "RAW"
        self.preprocessed_data = self.data_path / "PREPROCESSED"
        self.download_data = self.data_path / "DOWNLOADS"
        self.temp_data = self.log_path / "TEMP"
        self.checkpoint_path = self.log_path / "CHECKPOINTS"
        self.task_dump_path = self.log_path / "task_dump.json"
        # if reuse and remove cfg are dataclasses we need to transform to DictConfig but leave untouched if they are
        # already DictConfigs - this would break omegaconf interpolation resolving
        self.reuse_cfg = DictConfig(dataclasses.asdict(reuse_cfg)) if isinstance(reuse_cfg, ReuseConfig) else reuse_cfg
        if self.reuse_cfg and "clean_up" in self.reuse_cfg:
            raise MMLMisconfigurationException("reuse.clean_up=... is no longer supported. Use remove=... instead.")
        self.remove_cfg = (
            DictConfig(dataclasses.asdict(remove_cfg)) if isinstance(remove_cfg, RemoveConfig) else remove_cfg
        )
        # create base of data path if not existent
        for p in [self.raw_data, self.preprocessed_data, self.download_data, self.temp_data, self.checkpoint_path]:
            p.mkdir(exist_ok=True)
        # init overview on created paths (for later remove)
        self.created_paths_file = self.log_path / "created_paths.txt"
        self.created_paths_file.touch(exist_ok=True)
        # create index and reusables
        self.task_index: Dict[str, Dict[str, Path]] = {}
        self.reusables: Dict[str, Dict[str, Union[Path, List[ModelStorage]]]] = {}
        self.reload_task_index()
        self._find_reusables()

    @property
    def results_root(self) -> Path:
        """The root path of the current systems results."""
        return self.proj_path.parent

    @property
    def global_reusables(self) -> Dict[str, Path]:
        """Global reusables are not attached to a specific task."""
        if self.GLOBAL_REUSABLE not in self.reusables:
            logger.debug("No global reusables found.")
            return {}
        return self.reusables[self.GLOBAL_REUSABLE]

    def get_download_path(self, dset_name: str) -> Path:
        """
        Creates and returns a download path for some name dataset name. This will point to the same download path
        if called again, via this mechanism detecting existing downloads and continuing downloads is possible.

        :param str dset_name: string (ideally representing the dataset)
        :return: empty directory to store downloaded data
        :rtype: ~pathlib.Path
        """
        candidate = self.download_data / dset_name
        if candidate.exists():
            logger.info(f"Found and reuse already existing download folder for dataset {dset_name} @ {candidate}.")
        candidate.mkdir(exist_ok=True, parents=False)
        return candidate

    def get_dataset_path(
        self, dset_name: Optional[str] = None, raw_path: Optional[Path] = None, preprocessing: Optional[str] = None
    ) -> Path:
        """
        Creates and/or returns a correct dataset directory for the given preprocessing ID. Note that this should only
        be called once for the raw case, but can be called multiple times for (even identical) preprocessing IDs.

        :param Optional[str] dset_name: name of the dataset (provide this in case a new raw dataset is created)
        :param Optional[~pathlib.Path] raw_path: existing path to the raw version of a dataset (provide this in case
            a preprocessing is desired)
        :param Optional[str] preprocessing: preprocessing ID, None for raw data

        :raises AssertionError: in case exactly one of preprocessing and raw_path is given
        :raises AssertionError: in case not exactly one of raw_path and dset_name is given
        :raises FileExistsError: in case dset_name has been given before

        :return: path to store dataset files
        :rtype: ~pathlib.Path
        """
        assert preprocessing not in ["None", "none"], (
            "Use None value instead none string to call get_dataset_path for raw data."
        )
        assert (raw_path is None) == (preprocessing is None), "See usage of get_dataset_path."
        assert (dset_name is None) != (raw_path is None), "See usage of get_dataset_path."
        base = self.raw_data if preprocessing is None else self.preprocessed_data / preprocessing
        base.mkdir(exist_ok=True)
        if preprocessing:
            dset_name = self.undo_prefix(raw_path.name)
            # check if this dataset has already (partly) being preprocessed
            existing = self.get_all_dset_names()
            if preprocessing in existing:
                if dset_name in existing[preprocessing]:
                    warnings.warn(
                        UserWarning(
                            f"Dataset {dset_name} has already partly existing preprocessing {preprocessing}. "
                            f"This could be because of a previous run on the same task or a related task on the"
                            f" same dataset. Be aware that data might be overwritten, which can cause problems "
                            f"if you manipulated the preprocessing pipeline of this ID or use random augmentations "
                            f"on different modalities that are partly shared with other tasks!"
                        )
                    )
                    return existing[preprocessing][dset_name]
        candidate = base / f"{DSET_PREFIX}_{dset_name}"
        if candidate.exists():
            raise FileExistsError(
                f"Dataset {dset_name} already present at {candidate}! Please choose a different name."
            )
        candidate.mkdir(parents=False, exist_ok=False)
        logger.debug(f"Created RAW folder for dataset {dset_name}.")
        return candidate

    def get_all_dset_names(self) -> Dict[str, Dict[str, Path]]:
        """
        Returns all found dataset names. Datasets are clustered by their preprocessing.

        :return: dict with preprocessing key and dict value that itself corresponds to dataset names as key and root
                path as value, note that the literal string none is used for not preprocessed data
        :rtype: Dict[str, Dict[str, Path]]
        """
        all_dsets = {"none": {}}
        # first scan RAW folder -> preprocessing id is none in that case
        for dataset in self.raw_data.iterdir():
            if not dataset.is_dir() or DSET_PREFIX not in dataset.name:
                continue
            all_dsets["none"][self.undo_prefix(dataset.name)] = dataset
        # now preprocessed folder
        for preprocess in self.preprocessed_data.iterdir():
            if not preprocess.is_dir():
                continue
            all_dsets[preprocess.name] = {}
            for dataset in preprocess.iterdir():
                if not dataset.is_dir() or DSET_PREFIX not in dataset.name:
                    continue
                all_dsets[preprocess.name][self.undo_prefix(dataset.name)] = dataset
        return all_dsets

    def get_task_path(self, dset_path: Path, task_alias: str) -> Path:
        """
        Creates and returns a correct task file path.

        :param dset_path: dataset path of the task
        :param task_alias: name of the task (is used as abbreviation internally and initialises dir name)
        :return: path to put task .json file
        """
        assert dset_path.exists(), f"Invalid dataset path {dset_path}, use get_dataset_path to create a valid path."
        if self.raw_data in dset_path.parents:
            preprocessed = "none"
        elif self.preprocessed_data in dset_path.parents:
            preprocessed = list(dset_path.relative_to(self.preprocessed_data).parents)[-2].name
        else:
            raise ValueError(f"Given dset_path {dset_path} neither extends raw nor preprocessed paths!")
        if task_alias in self.task_index:
            if preprocessed in self.task_index[task_alias]:
                raise ValueError(f"Alias {task_alias} already used in preprocessed {preprocessed}!")
        cleaned = task_alias.replace(" ", "_").replace("--", "")
        candidate = dset_path / f"{TASK_PREFIX}_{cleaned}.json"
        if candidate.exists():
            raise FileExistsError(f"TaskFile @ {candidate} already exists.")
        return candidate

    def reload_task_index(self) -> None:
        """
        Scans self.raw_data and self.preprocessed_data for all available tasks, thereby creates a library
        concerning aliases and also preprocessings. Task index is a dict with dicts, hierarchy is name ->
        preprocessing -> (relative) path.

        :return: None
        """
        with catch_time() as timer:
            task_index = {}
            # first scan RAW folder -> preprocessing id is none in that case
            for dataset in self.raw_data.iterdir():
                if not dataset.is_dir():
                    continue
                for task in dataset.glob("*.json"):
                    # will ignore e.g. temp files
                    if TASK_PREFIX not in task.stem:
                        continue
                    alias = self.load_task_description_header(task).name
                    if alias == "":
                        continue
                    if alias in task_index:
                        logger.warning(
                            f"Duplicated task name {alias} at {task} and {task_index[alias]['none']}, "
                            f"this overwrites one of them in (more or less) random order!"
                        )
                    task_index[alias] = {"none": task.relative_to(self.data_path)}
            # now preprocessed folder
            for preprocess in self.preprocessed_data.iterdir():
                if not preprocess.is_dir():
                    continue
                for dataset in preprocess.iterdir():
                    if not dataset.is_dir():
                        continue
                    for task in dataset.glob("*json"):
                        if TASK_PREFIX not in task.stem:
                            continue
                        alias = self.load_task_description_header(task).name
                        if alias == "":
                            continue
                        if alias in task_index:
                            if preprocess.name in task_index[alias]:
                                logger.warning(
                                    f"Duplicated task name {alias} at {task} and {task_index[alias][preprocess.name]}, "
                                    f"this overwrites one of them in (more or less) random order!"
                                )
                            task_index[alias][preprocess.name] = task.relative_to(self.data_path)
                        else:
                            task_index[alias] = {preprocess.name: task.relative_to(self.data_path)}
            self.task_index = task_index
        logger.debug(f"(Re)loaded task index and found {len(self.task_index)} aliases in {timer.elapsed:5.2f} seconds.")

    def add_to_task_index(self, path: Path) -> None:
        """
        Adds a single task path to the task index.

        :param path: path to .json to be added
        :return: None
        """
        assert path.exists()
        assert path.suffix == ".json"
        task_name = self.load_task_description_header(path).name
        preprocess = (
            "none" if self.raw_data in path.parents else list(path.relative_to(self.preprocessed_data).parents)[-2].name
        )
        if task_name not in self.task_index:
            self.task_index[task_name] = {}
        assert preprocess not in self.task_index[task_name]
        self.task_index[task_name][preprocess] = path
        logger.debug(f"Added task {task_name} with preprocessing {preprocess} to task index.")

    @staticmethod
    def load_task_description(path: Path) -> TaskDescription:
        """
        Returns TaskDescription of given task.

        :param Path path: path to .json file
        :return: loaded TaskDescription with all meta information
        """
        if not path.exists():
            raise FileNotFoundError(f"Meta task file not found at {path}!")
        with open(str(path), "rb") as f:
            data_dict = orjson.loads(f.read())
        if any([key not in data_dict for key in ALL_TASK_DESCRIPTION_KEYS]):
            raise RuntimeError(
                f"Task keys ({data_dict.keys()}) do not cover all required keys ({ALL_TASK_DESCRIPTION_KEYS})."
            )
        task_description = TaskDescription.from_json(data_dict)
        logger.debug(f"Successfully loaded task description from {path}.")
        return task_description

    @staticmethod
    def write_task_description(path: Path, task_description: TaskDescription, omit_warning: bool = False) -> None:
        """
        Stores meta information of a task at the given path.

        :param Path path: path to store .json file
        :param TaskDescription task_description: TaskDescription of a task
        :param bool omit_warning: if True will raise no warning even if the file already exists
        :return: None
        """
        logger.info(f"Writing task description at {path}.")
        if path.exists() and not omit_warning:
            warnings.warn(f"Overwriting existing task meta information at {path}!", UserWarning)
        data_dict = task_description.to_json()
        with catch_time() as writing_timer:
            try:
                with open(str(path), "wb") as f:
                    f.write(orjson.dumps(data_dict))
            # if writing description fails remove the created file again
            except Exception as e:
                if path.exists():
                    path.unlink()
                raise e
        logger.debug(f"Task description writing time was {writing_timer.pretty_time}.")

    @staticmethod
    def load_task_description_header(path: Path) -> TaskDescription:
        """
        Similar to load_meta, but recovers only the header information necessary to construct TaskStruct, without the
        details regarding folds and sample paths. Useful for large .json files to save time on MML initialisation.

        :param Path path: path to .json file
        :return: TaskDescription with only meta information
        """
        if not path.exists():
            raise FileNotFoundError(f"Task description file not found at {path}!")
        data_dict = {}
        with open(str(path), "rb") as f:
            for key, value in ijson.kvitems(f, "", use_float=True, buf_size=32768):
                if key in ALL_HEADER_KEYS:
                    data_dict[key] = value
                # after loading all required keys, stop parsing the json -> this is what saves time here
                if len(data_dict) == len(ALL_HEADER_KEYS):
                    break
        if any([key not in data_dict for key in ALL_HEADER_KEYS]):
            raise RuntimeError(
                f"Task keys ({data_dict.keys()}) do not cover all required keys ({ALL_HEADER_KEYS}). "
                f"Did you miss to migrate your database after an mml-core update?"
            )
        task_description = TaskDescription.from_json(data_dict)
        logger.debug(f"Successfully loaded task description header from {path}.")
        return task_description

    def get_task_info(self, task_name: str, preprocess: str) -> dict:
        """
        Locates (if possible) the preprocessed task with provided name. Falls back to raw data in case preprocess is
        not available. Returns dict that can be used to construct a TaskStruct object.

        :param str task_name: name of the task
        :param str preprocess: a preprocess id (e.g. 'none' for raw task)
        :return: kwargs required for TaskStruct
        """
        if task_name not in self.task_index:
            raise TaskNotFoundError(f"Task {task_name} not listed in task index!")
        if preprocess not in self.task_index[task_name]:
            # if saved preprocess is not available, load raw version instead
            if "none" not in self.task_index[task_name]:
                raise TaskNotFoundError(f"No valid loading for task {task_name} given preprocess {preprocess}.")
            logger.debug(f"Falling back to loading non-preprocessed data for task {task_name}.")
            preprocess = "none"
        target_path = self.data_path / self.task_index[task_name][preprocess]
        # for rather large .json files only partly parse, else orjson is faster than partial parsing
        if os.path.getsize(target_path) > 2**14:
            task_description = self.load_task_description_header(target_path)
        else:
            task_description = self.load_task_description(target_path)
        info_kwargs = {"preprocessed": preprocess, "relative_root": self.task_index[task_name][preprocess]}
        for key in STRUCT_REQ_HEADER_KEYS:
            info_kwargs[key] = getattr(task_description, key)
        return info_kwargs

    def _find_reusables(self) -> None:
        """
        Scans projects based on the reuse_cfg for existing results that may be recycled. Reusables are stored as
        a dict with task name keys (and the global reusable key) and dicts corresponding to what may be reused (e.g.
        'blueprint' or 'pipeline') and values are paths to the respective files.

        :return: None
        """
        if self.reuse_cfg is None:
            logger.debug("File manager has no reuse config to find reusables.")
            return
        base_proj_path = self.proj_path.parent
        reusables: Dict[str, Dict[str, Union[Path, List[ModelStorage]]]] = {}
        # warn for unregistered reuse configuration entries
        for attribute in self.reuse_cfg:
            if attribute.split(REUSABLE_NUMBER_TAG)[0] not in self._path_assignments:
                warnings.warn(
                    f"You specified reuse of {attribute} but no path has been assigned. Make sure to import "
                    f"any necessary package that defines this path assignment. Nothing will be reused for "
                    f"now.",
                    UserWarning,
                )
        # iterate through registered path assignments and check for reuse config entry
        for key, assignment in self._path_assignments.items():
            attr_cls, assigned_path, attr_numbering, attr_reusable = assignment
            # check if path is designed reusable
            if not attr_reusable:
                continue
            # get and check reuse source project
            proj_or_projs = getattr(self.reuse_cfg, key, None)
            # no reuse specified
            if proj_or_projs is None:
                continue
            # reuse may be either one or multiple projects
            if isinstance(proj_or_projs, str):
                proj_or_projs = [proj_or_projs]
            else:
                warnings.warn(
                    "Requested multi-project reusables. For all but model reusing this will lead to the "
                    "behaviour that projects are inspected in order as listed and multiple present artefacts "
                    "may override each other. Model artefacts are ALL loaded and placed in a single list."
                )
            for proj_plus_number in proj_or_projs:
                proj = proj_plus_number.split(REUSABLE_NUMBER_TAG)[0]
                number = (
                    int(proj_plus_number.split(REUSABLE_NUMBER_TAG)[1])
                    if REUSABLE_NUMBER_TAG in proj_plus_number
                    else None
                )
                if number is not None and not attr_numbering:
                    raise ValueError(
                        f"path assignment for {key} does not have numbering enabled - you may not use "
                        f"{REUSABLE_NUMBER_TAG} within reuse.{key}=..."
                    )
                proj_path = base_proj_path / proj
                if not proj_path.exists():
                    raise MMLMisconfigurationException(
                        f"specified project {proj} for reuse of {key} not found (@{proj_path})."
                    )
                if attr_reusable == self.GLOBAL_REUSABLE:
                    # global reusable -> follow path into folder
                    # remove project and file name
                    final_path_folder = proj_path / Path(*assigned_path.parts[1:]).parent
                    # check for non-existing or empty directory
                    if not final_path_folder.exists() or next(final_path_folder.iterdir(), None) is None:
                        warnings.warn(f"Specified project {proj} for reuse of {key} seems to have none such")
                        continue
                    if number is not None:
                        number_map = {int(p.stem.split("_")[-1]): p for p in final_path_folder.iterdir()}
                        if number not in number_map:
                            raise ValueError(
                                f"no file with number {number} found in proj {proj} for reusing key {key}."
                                f" Folder is {final_path_folder}."
                            )
                        latest_file = number_map[number]
                    else:
                        latest_file = max(final_path_folder.iterdir(), key=os.path.getctime)
                    assert latest_file.is_file()
                    if self.GLOBAL_REUSABLE not in reusables:
                        reusables[self.GLOBAL_REUSABLE] = {}
                    reusables[self.GLOBAL_REUSABLE][key] = latest_file
                    logger.debug(f"Found global reusable {key} from project {proj} @ {latest_file}.")
                else:
                    # otherwise task specific reusables, follow individual task paths
                    attr_path = proj_path / assigned_path.parts[1]
                    if not attr_path.exists():
                        warnings.warn(f"Specified project {proj} for reuse of {key} seems to have none such")
                    else:
                        for task_path in attr_path.iterdir():
                            if "%" in task_path.name:
                                # these are old style "task_id" based paths
                                warnings.warn(
                                    'Loading reusables with old style path assignments ("task_id"). Backward '
                                    "compatibility may break in the future!",
                                    DeprecationWarning,
                                )
                                task_name = "_".join(task_path.name.split("%")[-1].split("_")[1:])
                            else:
                                # new style "task_name" based paths
                                task_name = task_path.name
                            if task_name not in reusables:
                                reusables[task_name] = {}
                            if key == "models":
                                if number is not None:
                                    raise MMLMisconfigurationException(
                                        "May not specify a specific model to reuse -all models will be loaded."
                                    )
                                if "models" not in reusables[task_name]:
                                    reusables[task_name]["models"] = []
                                for storage_path in task_path.iterdir():
                                    assert storage_path.is_file()
                                    storage = ModelStorage.from_json(storage_path, results_root=self.results_root)
                                    reusables[task_name]["models"].append(storage)
                                    logger.debug(f"Found reusable model from project {proj} @ {storage_path}.")
                            else:
                                # check for non-existing or empty directory
                                if not task_path.exists() or next(task_path.iterdir(), None) is None:
                                    continue
                                if number is not None:
                                    number_map = {int(p.stem.split("_")[-1]): p for p in task_path.iterdir()}
                                    import IPython

                                    IPython.embed()
                                    if number not in number_map:
                                        raise ValueError(
                                            f"no file with number {number} found in proj {proj} for reusing key {key}."
                                            f" Folder is {task_path}."
                                        )
                                    latest_file = number_map[number]
                                else:
                                    latest_file = max(task_path.iterdir(), key=os.path.getctime)
                                assert latest_file.is_file()
                                reusables[task_name][key] = latest_file
                                logger.debug(f"Found reusable {key} from project {proj} @ {latest_file}.")
        self.reusables = reusables

    @classmethod
    def add_assignment_path(
        cls,
        obj_cls: Optional[type],
        key: str,
        path: Union[Path, str],
        enable_numbering: bool = True,
        reusable: Union[bool, str] = False,
    ) -> None:
        """
        Adds a custom path assignment to the file manager. A necessary location to do the assignment is before the
        initialization of the file manager (note this is a class method), which could be just before starting the
        :func:`~mml.cli.main` inside your code or the 'activate.py' of your plugin.
        Once the assignment is done, a new path can be requested via the :meth:`construct_saving_path` method.
        Furthermore, the path assignments control the reuse functionality of the file manager.

        :param Optional[type] obj_cls: the class of objects you want to create a path for, this is used for
            double-checking during usage, provide None if you want to omit this step
        :param str key: the key you want to refer your path to, this must be unique, raises :exc:`KeyError` if the
            key is already in use
        :param Union[~pathlib.Path, str] path: the desired path to store the data, it must start either with `PROJ_PATH`
            or `TEMP_PATH`, which will later be replaced with the file managers attr:`proj_path` respectively
            :attr:`temp_data` otherwise raises a :exc:`ValueError`. The path may use the following further
            special tokens, that will be replaced during actual path creation: `TASK_NAME` and `FILE_NAME`.
            `TASK_NAME` is a placeholder that will be replaced during
            :meth:`construct_saving_path` and is necessary for the reuse functionality (see `reusable` below).
            `FILE_NAME` will
            allow naming the file during the actual call to :meth:`construct_saving_path` and is only allowed as the
            last part of the path.
        :param bool enable_numbering: boolean deciding if the path is static or will dynamically increase when a file
            already exists. Default: ``True``
        :param Union[bool, str] reusable: if True the paths should be reusable. This allows to set
            ``reuse.key=project`` (or even ``reuse.key=[project1,project2]`` to load from multiple projects, where the
            last found artefact persists) when starting :mod:`mml` and automatically attach the latest path under
            `PROJ_PATH/<ATTRIBUTE>/TASK_NAME` to each task structs
            `:attr:`~mml.core.data_loading.task_struct.TaskStruct.paths` dictionary with ``key`` as a key. This
            requires path to fit the format `PROJ_PATH/<ATTRIBUTE>/TASK_NAME/<some_file_name>`, where `<ATTRIBUTE>` is
            required to be capitalized by convention, otherwise a raises an ValueError. If the string
            `:attr:`~mml.core.data_loading.file_manager.MMLFileManager.GLOBAL_REUSABLE` is used
            instead any found reusable is attached to the `_TOP_LEVEL_` entry
            to be reached via `:attr:`~mml.core.data_loading.file_manager.MMLFileManager.global_reusables` property.
            This does not require the path format previously specified.
        :raises KeyError: if key is already used for a path construction
        :raises ValueError: If either
            * path does not start with `PROJ_PATH` or `TEMP_PATH`
            * the path has no suffix to indicate a file type (exception if `FILE_NAME` is the last path segment)
            * `FILE_NAME` is used as a non-final part of the path
            * the cls argument is neither None nor a class
            * reusable=True but path does not match the described requirements
            * ``..`` or ``~`` in path
            * plus some more checks

        :return: None
        """
        if key in cls._path_assignments:
            raise KeyError(f"Key {key} already used by file manager for path assignments.")
        path = Path(path)
        if path.parts[0] != "PROJ_PATH" and path.parts[0] != "TEMP_PATH":
            raise ValueError("assignment path must start with either PROJ_PATH or TEMP_PATH")
        if not path.name == "FILE_NAME" and path.suffix == "":
            raise ValueError("assignment path must end either with FILE_NAME or provide a suffix")
        if "FILE_NAME" in path.parts[:-1]:
            raise ValueError("FILE_NAME may be only used as a token for the last part of the path")
        if obj_cls is not None and not isinstance(obj_cls, type):
            raise ValueError(f"given cls={obj_cls} is not a class nor None")
        # does reusability apply?
        global_reusable = False
        if not isinstance(reusable, bool):
            if reusable != cls.GLOBAL_REUSABLE:
                raise ValueError(f"reusable must either be boolean or equal {cls.GLOBAL_REUSABLE}")
            if path.parts[0] != "PROJ_PATH":
                raise ValueError("Reusable must be non-temporary - use PROJ_PATH as first path entry.")
            if "TASK_NAME" in path.parts:
                raise ValueError("Global reusable must be task independent!")
            global_reusable = True
        if reusable and not global_reusable:
            reusable_err_msg = (
                "Reusable path assignment requested, but path does not match requirements! Read documentation."
            )
            if len(path.parts) != 4:
                raise ValueError(reusable_err_msg)
            # automatically attachable reusables must match this pattern
            base, attr_id, task_name, file_name = tuple(path.parts)
            if base != "PROJ_PATH" or not attr_id.isupper() or task_name != "TASK_NAME":
                raise ValueError(reusable_err_msg)
        # some sneaky corner cases
        if key in ["checkpoints"]:
            raise ValueError("Checkpoints are handled from within the scheduler.")
        all_tokens = ["PROJ_PATH", "TEMP_PATH", "FILE_NAME", "TASK_NAME"]
        for part in path.parts:
            if any([token in part and not token == part for token in all_tokens]):
                raise ValueError(f"Token misuse in path {path}!")
        for token in all_tokens:
            if path.parts.count(token) > 1:
                raise ValueError(f"Token {token} is used multiple times in path {path}!")
        if "-" in key or ":" in key:
            raise ValueError("Avoid using - or : in key, use _ instead!")
        if "~" in path.parts or ".." in path.parts:
            raise ValueError("Avoid using ~ or .. in path!")
        if REUSABLE_NUMBER_TAG in path.stem:
            raise ValueError(f"file name must not contain {REUSABLE_NUMBER_TAG}!")
        cls._path_assignments[key] = (obj_cls, path, enable_numbering, reusable)

    def construct_saving_path(
        self, obj: object, key: str, task_name: Optional[str] = None, file_name: Optional[str] = None
    ) -> Path:
        """
        All file savings are organised here to avoid unwanted interactions from different applications.

        :param Any obj: object to be saved (if the object itself are files, simply give None)
        :param str key: string, must be in the DEFAULT_ASSIGNMENTS or manually assigned previously
            (see :meth:`add_assignment_path`)
        :param Optional[str] task_name: (optional) name of the task, only necessary if TASK_NAME in the assignment
            pattern
        :param Optional[str] file_name: (optional) file name, only necessary if FILE_NAME in the assignment pattern
        :return: path to save the object
        """
        if task_name is not None and "%" in task_name:
            raise ValueError("Constructing saving path with task_id is deprecated. Please use task_name instead!")
        if key not in self._path_assignments:
            raise KeyError(
                f"Wanted to construct saving path with key {key}, which is unknown. Did you miss to "
                f"add_assignment_path?"
            )
        if file_name is not None and REUSABLE_NUMBER_TAG in file_name:
            raise ValueError(f"symbol {REUSABLE_NUMBER_TAG} is not allowed in file name!")
        obj_class, path, enable_numbering, _ = self._path_assignments[key]
        if obj_class is not None:
            assert isinstance(obj, obj_class), (
                f"Expected obj with type {obj_class} for saving key {key}, but got {type(obj)}."
            )
        if "TASK_NAME" in path.parts and task_name is None:
            raise ValueError(f"Must provide task_name for {key=}")
        if "FILE_NAME" in path.parts and file_name is None:
            raise ValueError(f"Must provide file_name for {key=}")
        # parse special elements of path
        assignment_tokens = {
            "PROJ_PATH": str(self.proj_path),
            "TEMP_PATH": str(self.temp_data),
            "FILE_NAME": file_name,
            "TASK_NAME": task_name,
        }
        for token, replacement in assignment_tokens.items():
            if replacement is not None:  # capture non set file_name and/or task_name
                path = Path(str(path).replace(token, replacement))
        # create file structure above
        path.parent.mkdir(exist_ok=True, parents=True)
        # number file if requested
        if enable_numbering:
            logger.debug(f"Saving some {key} at {path}! This triggers an enumeration number to avoid overwriting!")
            path_found = False
            retries = 0
            while not path_found:
                if retries > 10:
                    raise RuntimeError(f"Was not able to create path for {obj=} {key=} {task_name=} {file_name=}.")
                nums = [
                    int(p.stem.split("_")[-1]) for p in path.parent.iterdir() if path.stem in p.stem and "_" in p.stem
                ]
                nums += [0]  # assure nums is not empty
                new_name = path.stem + "_" + str(max(nums) + 1).zfill(4) + path.suffix
                path = path.parent / new_name
                try:
                    path.touch(exist_ok=False)
                    path_found = True
                except FileExistsError:
                    logger.error(f"Race condition for storing at {path} encountered. Will retry")
                    retries += 1
        # log created path
        with open(self.created_paths_file, "a") as file:
            file.write(f"{key}:{path}\n")
        return path

    def remove_intermediates(self) -> None:
        """
        Based on the clean_up settings of the reuse_config deletes intermediate results within this project to minimize
        disk storage. Other intermediates like the checkpoints path and the temp path are also cleared.

        :return: None
        """
        # clean up config
        if self.remove_cfg:
            clean_up_keys = set([k for k, v in self.remove_cfg.items() if bool(v)])
            for key in clean_up_keys:
                if key not in self._path_assignments:
                    warnings.warn(f"Requested deletion of paths with key {key} but this has never been registered!")
            with open(self.created_paths_file, "r") as file:
                created_paths = file.readlines()
            found_keys = set()
            unlinked_counter = 0
            for line in created_paths:
                key, path = line.strip().split(":")
                if key in clean_up_keys:
                    found_keys.add(key)
                    Path(path).unlink()
                    unlinked_counter += 1
            # generate report
            logger.info(f"A total of {len(created_paths)} paths have been created during this run.")
            if unlinked_counter > 0:
                logger.info(f"File manager removed {unlinked_counter} files of types {found_keys}.")
            not_found = clean_up_keys - found_keys
            if not_found:
                logger.info(f"Although requested {not_found} cleanup, no paths were created!")
        else:
            logger.info("HINT: Specify reuse.clean_up config to reduce memory requirements.")
        # checkpoints are auto deleted
        shutil.rmtree(self.checkpoint_path)
        logger.debug("Removed CHECKPOINTS folder.")
        # temp folder is auto deleted
        shutil.rmtree(self.temp_data)
        logger.debug("Removed TEMP folder.")

    @staticmethod
    def undo_prefix(dir_name: str) -> str:
        """
        Reverts the application of task/dset prefix adding.

        :param dir_name: string to be applied on
        :return: non-prefixed string
        """
        return "_".join(dir_name.split("_")[1:])

    def get_pp_definition(self, preprocessing: str) -> Path:
        """
        Return a definition copy of a created preprocessing folder. Can be used to check whether a preprocessing
        definition has changed since it's processing. If no file has been created so far a new one will be created.

        :param preprocessing: the ID of the preprocessing
        :return: path to a file to store a preprocessing definition
        """
        if preprocessing == "none":
            raise ValueError("No preprocessing definition will be created for unpreprocessed data!")
        path = self.preprocessed_data / preprocessing / "_definition.yaml"
        path.parent.mkdir(exist_ok=True)
        return path


# add default assignments when this file is loaded
for key, assignment in DEFAULT_ASSIGNMENTS.items():
    obj_cls, path, numbering, reuse = assignment
    MMLFileManager.add_assignment_path(obj_cls=obj_cls, key=key, path=path, enable_numbering=numbering, reusable=reuse)
