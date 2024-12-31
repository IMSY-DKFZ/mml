# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import functools
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import orjson
from omegaconf import DictConfig

from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.data_loading.task_attributes import Keyword, Modality, RGBInfo, Sizes, TaskType
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.scripts.decorators import deprecated
from mml.core.scripts.exceptions import TaskNotFoundError
from mml.core.scripts.model_storage import ModelStorage
from mml.core.scripts.utils import TAG_SEP, catch_time

logger = logging.getLogger(__name__)


class TaskStruct:
    """
    Object to handle tasks on a meta level in the framework. Contains basic information on location of data and
    links to intermediate and final results. Will be instantiated by the TaskFactory. During runtime results
    corresponding to the dataset will also be stored within the object (such as trained models, calculated FIM and
    performance).
    """

    def __init__(
        self,
        name: str,
        task_type: TaskType,
        means: RGBInfo,
        stds: RGBInfo,
        sizes: Sizes,
        class_occ: Dict[str, int],
        keywords: List[Keyword],
        idx_to_class: Dict[int, str],
        modalities: Dict[Modality, str],
        relative_root: str,
        preprocessed: str,
    ):
        # this used later on to identify the task
        self.name = name

        # permanent task attributes e.g. holds task type, mean and std of train set, relative root path, etc.
        self.task_type = task_type
        self.means = means
        self.stds = stds
        self.sizes = sizes
        self.class_occ = class_occ
        self.relative_root = Path(relative_root)  # relative to MMLFileManager.data_path
        self.preprocessed = preprocessed
        self.keywords = keywords
        self.idx_to_class = idx_to_class
        self.modalities = modalities
        if self.target and self.target not in self.modalities:
            warnings.warn(f"Corrupted target for task {self.name}: {self.target}")
        # non-permanent attributes, these correspond to experiment specific settings, e.g. a performance, a model
        # trained for the task, auto-augmentation, checkpoints, FIM, features, heads, etc...
        # be aware that in order to store and load them they should only consist of (stacked) default builtin types!
        # e.g. str, int, dict, list, ...
        self.paths: Dict[str, Path] = {}
        self.models: List[ModelStorage] = []

        logger.debug(f"Created TaskStruct-object for task {self.name}.")

    @staticmethod
    def non_permanent_task_attributes() -> Dict[str, Tuple[Callable, Callable]]:
        """
        Returns a dict of task attributes that are not part of the task meta information but are computed by MML.
        The value within the dict is a tuple of callables representing and instantiating the object (to be compatible
        with yaml safe loading and dumping).

        :return: dict with str keys, which are the names of task struct attributes that might be set during MML runtime
                 and tuple vals corresponding to (representer, instantiator)
        """
        attrs = {}

        # paths attr
        def path_representer(path_attr: Dict[str, Path]) -> Dict[str, str]:
            return {k: str(v) for k, v in path_attr.items()}

        def path_instantiator(path_repr: Dict[str, str]) -> Dict[str, Path]:
            return {k: Path(v) for k, v in path_repr.items()}

        attrs["paths"] = (path_representer, path_instantiator)

        # models attr
        def models_representer(model_attr: List[ModelStorage]) -> List[str]:
            paths = [str(m._stored) for m in model_attr]
            if any(p is None for p in paths):
                raise RuntimeError(
                    "Models that are attached to TaskStructs can only be dumped after a scheduler step "
                    "if the model has been stored before."
                )
            return paths

        def models_instantiator(model_repr: List[str]) -> List[ModelStorage]:
            return [ModelStorage.from_json(path=Path(path)) for path in model_repr]

        attrs["models"] = (models_representer, models_instantiator)
        return attrs

    @functools.cached_property
    def num_samples(self) -> int:
        """
        Number of training (or unlabeled) samples. See also
        `~mml.core.data_loading.task_description.TaskDescription.num_samples`.
        :rtype: int
        """
        # loading only supported with file manager
        if not MMLFileManager.exists():
            raise RuntimeError("TaskStruct supports num_samples only with initiated MMLFileManager.")
        fm = MMLFileManager.instance()
        return fm.load_task_description(fm.data_path / self.relative_root).num_samples

    @property
    def num_classes(self) -> int:
        return len(set(self.idx_to_class.values()))

    @property
    def target(self) -> Optional[Modality]:
        if self.task_type == TaskType.CLASSIFICATION:
            return Modality.CLASS
        elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            return Modality.MASK
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return Modality.CLASSES if Modality.CLASSES in self.modalities else Modality.SOFT_CLASSES
        elif self.task_type == TaskType.REGRESSION:
            return Modality.VALUE
        elif self.task_type == TaskType.NO_TASK:
            return None
        else:
            raise RuntimeError("Unable to determine target!")

    @property
    @deprecated(reason="task_struct.id is deprecated, use task.name instead", version="0.12.0")
    def id(self) -> str:
        return self.name

    def __str__(self) -> str:
        infos = [
            f"Task name: {self.name}",
            f"Task type: {self.task_type}",
            f"Num classes: {self.num_classes}",
            f"Means: {self.means}",
            f"Stds: {self.stds}",
            f"Sizes: {self.sizes}",
            f"Class occ: {self.class_occ}",
            f"Preprocessed: {self.preprocessed}",
            f"Task keywords: {[kw.value for kw in self.keywords]}",
        ]
        for attr in self.non_permanent_task_attributes().keys():
            infos.append(f"{attr}: {getattr(self, attr)}")
        return "\n".join(infos)

    def __repr__(self):
        return f"TaskStruct({self.name})"


class TaskStructFactory:
    """
    Manages to load all necessary TaskStructs for an experiment. Stores created objects and aggregates information (like
    sizes) across multiple tasks.
    """

    def __init__(self, cfg: DictConfig, load: bool = False):
        self.cfg = cfg
        self.fm = MMLFileManager.instance()
        self.container = []
        self.sizes = Sizes()
        self.reset_sizes()
        # load old factory dump
        if load:
            self.loading_old_dump()

    def reset_sizes(self) -> None:
        """
        Sets the internal sizes back.

        :return: None
        """
        self.sizes.min_height = 100000
        self.sizes.max_height = 0
        self.sizes.min_width = 100000
        self.sizes.max_width = 0

    def set_task_struct_defaults(self, task_struct: TaskStruct):
        """
        Based on reuse configs this sets the defaults within the task struct regarding previous results. Currently
        only supports Path and ModelStorage objects!

        :param TaskStruct task_struct: task_struct of the task that values should be loaded
        :return: None
        """
        if task_struct.name in self.fm.reusables:
            for k, v in self.fm.reusables[task_struct.name].items():
                if k == "models":
                    assert isinstance(v, list)
                    for storage in v:
                        assert isinstance(storage, ModelStorage)
                    task_struct.models = v
                    logger.debug(f"Attached {len(v)} reusable models to {task_struct.name}.")
                else:
                    assert isinstance(v, Path)
                    assert v.exists()
                    task_struct.paths[k] = v
                    logger.debug(f"Set {k} path of {task_struct.name} to {v}.")
        else:
            logger.debug(f"No reusable for task {task_struct.name}")

    def loading_old_dump(self) -> None:
        """
        Loading is useful if an experiment was aborted and is re-initialized.

        :return: None
        """

        logger.info(f"Loading task dump from {self.fm.task_dump_path}.")
        if not self.fm.task_dump_path.exists():
            raise FileNotFoundError(
                f"Specified exp folder ({self.fm.task_dump_path.parent}) requested for loading "
                f"TaskFactory dump is not existing or has incorrectly saved dump (requires "
                f"{self.fm.task_dump_path.name} file)."
            )
        with open(str(self.fm.task_dump_path), "rb") as f:
            all_tasks_dict = orjson.loads(f.read())
        logger.info(f"Starting loading of {len(all_tasks_dict)} tasks...")
        for name, task_dict in all_tasks_dict.items():
            created = self.create_task_struct(name, return_ref=True)
            for attr, (_, instantiator) in TaskStruct.non_permanent_task_attributes().items():
                if attr in task_dict.keys():
                    setattr(created, attr, instantiator(task_dict[attr]))
        # report sizes
        logger.debug(f"Sizes of factory are: {self.sizes}.")
        logger.info(f"Successfully loaded. Container includes {len(self.container)} task structs.")

    def dump(self, clear_container=False) -> None:
        """
        Stores current tasks and their attributes.

        :param clear_container: if true deletes currently loaded tasks afterwards
        :return: None
        """
        all_tasks_dict = {}
        for task in self.container:
            task_dict = {}
            for attr, (representer, _) in TaskStruct.non_permanent_task_attributes().items():
                if getattr(task, attr) is not None:
                    task_dict[attr] = representer(getattr(task, attr))
            all_tasks_dict[task.name] = task_dict
        with open(str(self.fm.task_dump_path), "wb") as f:
            f.write(orjson.dumps(all_tasks_dict))
        logger.debug(f"Dumped {len(all_tasks_dict)} tasks @ {self.fm.task_dump_path}.")
        if clear_container:
            self.container = []
            self.reset_sizes()

    def create_task_struct(self, name: str, return_ref=False) -> Union[None, TaskStruct]:
        """
        Creates a task struct object via loading necessary information from the meta info json file and adding
        reusable information (e.g. intermediate results from previous experiments) as adaption of the already
        preprocessed version of the task. Finally, the task struct is added to the internal container.

        :param name: name of the task to be created
        :param return_ref: if true returns a reference to the created struct, else returns None
        :return: either the created task struct or None
        """
        if self.check_exists(name=name):
            logger.error(f"Task struct {name} to produce already present in the factory container.")
            if return_ref:
                return self.get_by_name(name=name)
            else:
                return
        # make sure to remove duplicate tag
        undup_name = undup_names([name])[0]
        # next check if this is a base task that has not yet been created
        if (TAG_SEP not in undup_name) and (undup_name not in self.fm.task_index):
            # the task is not a tagged one and the base is not present, raise error
            raise TaskNotFoundError(
                f"Was not able to locate task {undup_name}. You may need to call "
                f"<mml create ...> with your current task setting."
            )
        # next check if this is a tagged task with missing entry with respect to preprocessing
        if (TAG_SEP in undup_name) and (
            undup_name not in self.fm.task_index or self.cfg.preprocessing.id not in self.fm.task_index[undup_name]
        ):
            if undup_name not in self.fm.task_index:
                # task not yet present at all in task index of the file manager, try to auto generate base task
                logger.info(f"Task {undup_name} not existent yet. Will try to create.")
                try:
                    with catch_time() as timer:
                        path = TaskCreator(dset_path=Path("")).auto_create_tagged(
                            full_alias=undup_name, preprocessing="none"
                        )
                    logger.debug(f"Task created successfully within {timer.elapsed:5.2f} seconds.")
                except TaskNotFoundError:
                    raise RuntimeError(f"Unable to auto_create {undup_name} with pp {self.cfg.preprocessing.id}.")
                # add to task index of file manager
                self.fm.add_to_task_index(path)
            # check for inconsistencies
            if "none" not in self.fm.task_index[undup_name]:
                raise RuntimeError(
                    f"MML detected a tagged task ({undup_name}) that exists with some "
                    f"preprocessing(s) ({list(self.fm.task_index[undup_name].keys())}), but "
                    f"no raw version has been found. This may be either because the raw version "
                    f"has been removed or you used a previous version of MML to create this "
                    f"tagged task. From MML 0.13.0 on tagged preprocessing will only be created "
                    f"with a base tagged task. Consider removing all preprocessed version of "
                    f"this task to create from scratch "
                    f"({list(self.fm.task_index[undup_name].values())})."
                )
            # next check if we need to create a preprocessed version
            if self.cfg.preprocessing.id not in self.fm.task_index[undup_name]:
                base_task = undup_name[: undup_name.find(TAG_SEP)]
                if self.cfg.preprocessing.id in self.fm.task_index[base_task]:
                    # this indicates the case that we can leverage existing preprocessing!
                    # preprocessed tagged task not yet present in task index of the file manager
                    logger.info(
                        f"Generating description of {undup_name} for preprocessing {self.cfg.preprocessing.id}."
                    )
                    try:
                        with catch_time() as timer:
                            path = TaskCreator.auto_create_tagged(
                                full_alias=undup_name, preprocessing=self.cfg.preprocessing.id
                            )
                        logger.debug(f"Task created successfully within {timer.elapsed:5.2f} seconds.")
                    except TaskNotFoundError:
                        raise RuntimeError(f"Unable to auto_create {undup_name} with pp {self.cfg.preprocessing.id}.")
                    # add to task index of file manager
                    self.fm.add_to_task_index(path)
        # generate struct from meta info provided by file manager
        def_kwargs = self.fm.get_task_info(task_name=undup_name, preprocess=self.cfg.preprocessing.id)
        if def_kwargs["name"] != name:
            raise RuntimeError(f'Received incorrect task information for task {name} (got {def_kwargs["name"]}).')
        new_task = TaskStruct(**def_kwargs)
        self.container.append(new_task)
        # apply defaults to task struct
        self.set_task_struct_defaults(new_task)
        # update sizes
        self.sizes.min_height = min(self.sizes.min_height, new_task.sizes.min_height)
        self.sizes.max_height = max(self.sizes.max_height, new_task.sizes.max_height)
        self.sizes.min_width = min(self.sizes.min_width, new_task.sizes.min_width)
        self.sizes.max_width = max(self.sizes.max_width, new_task.sizes.max_width)
        logger.debug(f"New factory sizes are: {self.sizes}")
        if return_ref:
            return new_task

    def get_by_name(self, name: str) -> TaskStruct:
        """
        Returns the internally stored task_struct corresponding to >name<. Raises an error if not found (or returns
        only false if >test< is true).

        :param name: task name
        :param test: if true does not raise an error
        :return: either the task_struct or False if not found and in test mode
        """
        for task in self.container:
            if task.name == name:
                return task
        msg = f"Was not able to find requested dataset {name} in the container of produced task structs."
        raise TaskNotFoundError(msg)

    def check_exists(self, name: str) -> bool:
        """
        Checks whether a given task is present in the container.

        :param str name: task name
        :return: True iff task is within container
        :rtype: bool
        """
        try:
            self.get_by_name(name=name)
        except TaskNotFoundError:
            return False
        return True


def undup_names(moded_names_list):
    """
    This function removes the "duplicate"-suffixes of tasks that is added if some tasks are present for multiple times.

    :param moded_names_list: list of strings, task names potentially including the "duplicate"-suffix
    :return: list of strings, tasks names without the suffix (if suffix is not present the name stays equal)
    """
    return list(
        map(
            lambda x: str(x)[: (lambda y: None if y == -1 else y)(str(x).find(f"{TAG_SEP}duplicate"))], moded_names_list
        )
    )
