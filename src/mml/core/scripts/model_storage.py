# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import datetime
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import orjson

if TYPE_CHECKING:
    from mml.core.data_loading.task_struct import TaskStruct

logger = logging.getLogger(__name__)


@dataclass
class ModelStorage:
    """
    Lightweight wrapper for everything to reproduce, load and compare trained models. Basically consists of a path to
    a saved pipeline, a path to saved parameters and a performance value, indicating validation metric after training.

    :param Path pipeline: path to a stored `~mml.core.scipts.pipeline_configuration.PipelineCfg`
    :param Path parameters: path to stored model parameters
    :param float performance: validation score of the model, usually best/last epoch loss value, might be used for model selection
    :param float training_time: training time in seconds
    :param Optional[str] task: in simple supervised settings this may indicate the target task trained for
    :param Optional[int] fold: may indicate the fold number used
    :param Dict[str, Path] predictions: (optional) predictions that have been made with this model
    :param list metrics: (optional) detailed training and validation metrics
    """

    pipeline: Path
    parameters: Path
    performance: float
    training_time: float = -1.0
    created: Optional[datetime.datetime] = None
    task: Optional[str] = None
    fold: Optional[int] = None
    predictions: Dict[str, Path] = field(default_factory=dict)
    metrics: list = field(default_factory=lambda: [])
    _stored: Optional[Path] = None

    def __str__(self):
        return f"ModelStorage(task={self.task}, fold={self.fold}, created={self.created})"

    def __repr__(self):
        return self.__str__()

    def store(
        self, task_struct: Optional["TaskStruct"] = None, path: Optional[Path] = None, fold: Optional[int] = None
    ) -> Path:
        """
        Saves the model storage. If struct is given it creates a new path and returns it, if path is given otherwise
        uses that if None is given it tries to look up if the storage has been loaded previously and will update that
        location.

        :param Optional[TaskStruct] task_struct: task struct corresponding to the task the model was trained on, will be
         used to determine the path.
        :param Optional[Path] path: (optional) if a path already exists for this storage, overwrite it, raises an error
         if the presented path does not exist yet
        :param Optional[int] fold: (optional) if a fold is specified and path is None, the file name will be
         fold_{fold}.json, otherwise the file name falls back to model_storage.json.
        :return: the path the storage was saved to
        """
        if path is None and task_struct is None and self._stored:
            path = self._stored
            logger.debug("Found previous location of ModelStorage. Will update that location.")
        if sum([path is None, task_struct is None]) in [0, 2]:
            raise ValueError("Provide either path or task_struct argument exclusively for a newly created ModelStorage")
        if task_struct and self.task is not None and self.task != task_struct.name:
            warnings.warn(f"Storing model given struct from {task_struct.name} but model task was set to {self.task}!")
        if fold and self.fold is not None and self.fold != fold:
            warnings.warn(f"Storing model given fold {fold} but model fold was set to {self.fold} before!")

        # import locally to avoid circular import
        from mml.core.data_loading.file_manager import MMLFileManager

        fm = MMLFileManager.instance()

        if (
            (fm.results_root not in self.parameters.parents)
            or (fm.results_root not in self.pipeline.parents)
            or any(fm.results_root not in pred.parents for pred in self.predictions.values())
        ):
            raise RuntimeError("Error while checking file path hierarchy.")

        if path is None:
            # check for valid task_name
            path = fm.construct_saving_path(
                obj=self,
                key="models",
                task_name=task_struct.name,
                file_name=f"fold_{fold}.json" if fold else "model_storage.json",
            )
        else:
            if path.exists():
                path.unlink()
                logger.info(f"Updating model storage at {path}.")
            else:
                raise FileNotFoundError("Provided path for Model Storage does not exist.")
        if self.created is None:
            self.created = datetime.datetime.now().replace(microsecond=0)
        data = {
            "pipeline": str(self.pipeline.relative_to(fm.results_root)),
            "parameters": str(self.parameters.relative_to(fm.results_root)),
            "performance": self.performance,
            "training_time": self.training_time,
            "task": self.task,
            "fold": self.fold,
            "created": self.created.isoformat(timespec="seconds"),
            "metrics": self.metrics,
            "predictions": {k: str(v.relative_to(fm.results_root)) for k, v in self.predictions.items()},
        }
        # use orjson to store and load
        with open(str(path), "wb") as f:
            f.write(orjson.dumps(data))
        logger.debug(f"Dumped model storage at {path}.")
        self._stored = path
        return path

    @classmethod
    def from_json(cls, path: Path, results_root: Optional[Path] = None) -> "ModelStorage":
        """
        Counterpart to saving the storage. Creates the storage object from a file.

        :param Path path: path to load the storage from
        :param Path results_root: the current systems' results root, if not provided will be tried to be inferred
        :return: a model storage dataclass
        """
        # may not be inferred while the first run of reusables is processed
        if results_root is None:
            # import locally to avoid circular import
            from mml.core.data_loading.file_manager import MMLFileManager

            fm = MMLFileManager.instance()
            results_root = fm.results_root
        # read in data
        with open(str(path), "rb") as f:
            data = orjson.loads(f.read())
        # backward compatibility
        if "predictions" not in data:
            data["predictions"] = {}
        if "task" not in data:
            data["task"] = None
        if "fold" not in data:
            data["fold"] = None
        if "created" not in data:
            data["created"] = None
        return cls(
            pipeline=results_root / data["pipeline"],
            parameters=results_root / data["parameters"],
            performance=data["performance"],
            training_time=data["training_time"],
            task=data["task"],
            fold=data["fold"],
            created=datetime.datetime.fromisoformat(data["created"]) if data["created"] else None,
            metrics=data["metrics"],
            predictions={k: results_root / v for k, v in data["predictions"].items()},
            _stored=path,
        )


@dataclass
class EnsembleStorage:
    """
    An EnsembleStorage represents a collection of models that are applied jointly on a task.
    """

    performance: float
    weights: List[float] = field(default_factory=list)
    members: List[Path] = field(default_factory=list)
    predictions: Dict[str, Path] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    search_params: dict = field(default_factory=dict)
    _stored: Optional[Path] = None

    def get_members(self) -> List[ModelStorage]:
        """
        Loads the actual ModelStorage members of the Ensemble from disk.
        :return: list of ModelStorage instances
        """
        return [ModelStorage.from_json(path=path) for path in self.members]

    @property
    def tasks(self) -> Set[str]:
        """
        Tasks of the members.
        """
        loaded_members = self.get_members()
        return set([member.task for member in loaded_members])

    @property
    def folds(self) -> List[int]:
        """
        Folds used by the members.
        """
        loaded_members = self.get_members()
        tasks = set([member.task for member in loaded_members])
        if len(tasks) != 1 or None in tasks:
            warnings.warn("Since Ensemble members have been training on inconsistent tasks, folds has no real meaning!")
        return [member.fold for member in loaded_members]

    def store(
        self, task_struct: Optional["TaskStruct"] = None, path: Optional[Path] = None, file_name: str = "ensemble.json"
    ) -> Path:
        """
        Saves the model ensemble. If struct is given it creates a new path and returns it, if path is given otherwise
        uses that if None is given it tries to look up if the storage has been loaded previously and will update that
        location.

        :param Optional[TaskStruct] task_struct: task struct corresponding to the task the ensemble was optimised on,
            will be used to determine the path. Either task_struct or path must be provided.
        :param Optional[Path] path: (optional) if a path already exists for this storage, overwrite it, raises an error
            if the presented path does not exist yet
        :param str file_name: only relevant for task_struct variant, determines the naming of the json file
        :return: the path the storage was saved to
        """
        if path is None and task_struct is None and self._stored:
            path = self._stored
            logger.debug("Found previous location of EnsembleStorage. Will update that location.")
        if sum([path is None, task_struct is None]) in [0, 2]:
            raise ValueError("Provide either path or task_struct argument exclusively.")
        if len(self.members) == 0:
            raise RuntimeError("No members to store for this Ensemble.")
        if len(self.weights) == 0:
            logger.info("No weights set for model ensemble, will use uniform weighing.")
            self.weights = [1 / len(self.members)] * len(self.members)
        if len(self.members) != len(self.weights):
            raise RuntimeError("Weight count does not match number of members.")
        loaded_members = self.get_members()
        all_tasks = set([member.task for member in loaded_members])
        if len(all_tasks) != 1:
            warnings.warn(f"Ensemble might have been trained for multiple tasks! ({all_tasks}")
        if task_struct and all_tasks and task_struct.name not in all_tasks:
            warnings.warn(
                f"Storing ensemble given struct from {task_struct.name} but ensemble has no member "
                f"associated to that task (only to {all_tasks})!"
            )

        # import locally to avoid circular import
        from mml.core.data_loading.file_manager import MMLFileManager

        fm = MMLFileManager.instance()

        if any(fm.results_root not in member.parents for member in self.members) or any(
            fm.results_root not in pred.parents for pred in self.predictions.values()
        ):
            raise RuntimeError("Error while checking file path hierarchy.")

        if path is None:
            if Path(file_name).suffix != ".json":
                raise ValueError('File name must have suffix ".json".')
            # check for valid task_name
            path = fm.construct_saving_path(obj=self, key="ensemble", task_name=task_struct.name, file_name=file_name)
        else:
            if path.exists():
                path.unlink()
                logger.info(f"Updating ensemble storage at {path}.")
            else:
                raise FileNotFoundError("Provided path for Ensemble Storage does not exist.")
        data = {
            "performance": self.performance,
            "weights": self.weights,
            "members": [str(member.relative_to(fm.results_root)) for member in self.members],
            "predictions": {k: str(v.relative_to(fm.results_root)) for k, v in self.predictions.items()},
            "metrics": self.metrics,
            "search_params": self.search_params,
        }
        # use orjson to store and load
        with open(str(path), "wb") as f:
            f.write(orjson.dumps(data))
        logger.debug(f"Dumped ensemble storage at {path}.")
        self._stored = path
        return path

    @classmethod
    def from_json(cls, path: Path) -> "EnsembleStorage":
        """
        Counterpart to saving the storage. Creates the storage object from a file.

        :param path: path to load the storage from
        :return: an ensemble storage dataclass
        """
        # import locally to avoid circular import
        from mml.core.data_loading.file_manager import MMLFileManager

        fm = MMLFileManager.instance()

        with open(str(path), "rb") as f:
            data = orjson.loads(f.read())
        return cls(
            members=[fm.results_root / entry for entry in data["members"]],
            weights=data["weights"],
            performance=data["performance"],
            predictions={k: fm.results_root / v for k, v in data["predictions"].items()},
            metrics=data["metrics"],
            search_params=data["search_params"],
            _stored=path,
        )
