# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import dataclasses
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
import scipy.stats
from omegaconf import OmegaConf

from mml.cli import main
from mml.core.data_loading.task_attributes import Keyword, TaskType
from mml.core.data_preparation.registry import get_dset_for_task
from mml.core.scripts.utils import TAG_SEP
from mml.interactive import _check_init
from mml.interactive.loading import default_file_manager

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AllTasksInfos:
    """
    A class to store all standard meta information on a set of tasks.
    """

    num_classes: Dict[str, int]
    num_samples: Dict[str, int]
    imbalance_ratios: Dict[str, float]
    datasets: Dict[str, str]
    keywords: Dict[str, Set[Keyword]]
    task_types: Dict[str, TaskType]
    domains: Dict[str, Keyword]
    dimensions: Dict[str, int]
    max_resolution: Dict[str, int]
    min_resolution: Dict[str, int]
    small_tasks: List[str]
    medium_tasks: List[str]
    large_tasks: List[str]

    def __repr__(self):
        return f"AllTasksInfos({len(self.num_classes)} tasks: {list(self.num_classes.keys())})"

    def __post_init__(self):
        """
        Automatically called after instantiation of an instance. Performs consistency check.
        :return:
        """
        self.check_consistency()

    def check_consistency(self):
        """
        Performs assertions that all information cover the same set of tasks.
        :return:
        """
        assert (
            self.num_classes.keys()
            == self.num_samples.keys()
            == self.keywords.keys()
            == self.max_resolution.keys()
            == self.min_resolution.keys()
        )
        assert (self.dimensions.keys() == self.num_classes.keys()) or len(self.dimensions) == 0
        # assert set(self.small_tasks) != set(self.medium_tasks) != set(self.large_tasks) != set(self.small_tasks)
        assert set(self.num_classes) == set(self.small_tasks + self.medium_tasks + self.large_tasks)

    def get_transformed(self, transforms: Sequence[str] = ("boxcox", "normalize")) -> "AllTasksInfos":
        """
        Allows to receive a modified instance of the task information where a couple of attributes are transformed.

        transformed attributes are: 'num_classes', 'num_samples', 'imbalance_ratios', 'dimensions', 'max_resolution',
            'min_resolution'

        available single transforms are: 'boxcox', 'normalize', 'zscore'

        :param Sequence[str] transforms: a sequence of legal transforms
        :return: a modified version of the task information, transforms have been applied on all attributes listed above

        """
        if len(transforms) == 0:
            return self
        transform, *remaining_transforms = transforms
        to_replace = [
            "num_classes",
            "num_samples",
            "imbalance_ratios",
            "dimensions",
            "max_resolution",
            "min_resolution",
        ]
        replace_dict = {}
        for _att in to_replace:
            _dict = getattr(self, _att)
            _vals = np.asarray(list(_dict.values()))
            if transform == "boxcox":
                _transformed, lmbda = scipy.stats.boxcox(_vals)
                logger.debug(f"boxcox lambda for {_att} is {lmbda}")
            elif transform == "zscore":
                _transformed = scipy.stats.zscore(_vals)
            elif transform == "log":
                _transformed = np.log(_vals)
            elif transform == "normalize":
                _transformed = _vals / _vals.max()
            else:
                raise ValueError(f"transform {transform} not available")
            replace_dict[_att] = dict(zip(_dict.keys(), _transformed))
        return dataclasses.replace(self, **replace_dict).get_transformed(remaining_transforms)

    def store_csv(self, path: Path) -> None:
        """
        Reformat meta information and write as a csv file.

        :param Path path: path to store csv file
        :return: None
        """
        all_tasks = sorted(self.num_samples.keys())
        task_infos = []
        for t in all_tasks:
            info_dict = {"name": t}
            for attr in [
                "task_types",
                "num_classes",
                "num_samples",
                "domains",
                "imbalance_ratios",
                "datasets",
                "keywords",
                "dimensions",
                "max_resolution",
                "min_resolution",
            ]:
                attr_dict = getattr(self, attr)
                try:
                    elem = attr_dict[t]
                    if attr == "keywords":  # Set[Keyword]
                        elem = [str(_kw) for _kw in sorted(elem)]
                    elif attr == "task_types" or attr == "domains":  # TaskType or Keyword
                        elem = str(elem)
                    info_dict[attr] = elem
                except KeyError:
                    # some entries might be missing
                    pass
            task_infos.append(info_dict)
        df = pd.DataFrame(task_infos)
        df.to_csv(path)

    @classmethod
    def from_csv(cls, path: Path) -> "AllTasksInfos":
        """
        Load stored AllTasksInfos from a csv file.

        :param path: path to load csv file
        :return: AllTasksInfos
        """
        df = pd.read_csv(path).set_index("name")
        kwargs = {}
        for attr in [
            "task_types",
            "num_classes",
            "num_samples",
            "domains",
            "imbalance_ratios",
            "datasets",
            "keywords",
            "dimensions",
            "max_resolution",
            "min_resolution",
        ]:
            kwargs[attr] = df[attr].to_dict() if attr in df.columns else {}
            if attr == "keywords":  # Dict[str, Set[Keyword]]
                for task in kwargs[attr]:
                    if kwargs[attr][task]:
                        kwargs[attr][task] = set(
                            [Keyword.from_str(_kw.strip(" \[\]'")) for _kw in kwargs[attr][task].split(",")]
                        )
            elif attr == "task_types":  # Dict[str, TaskType]
                kwargs[attr] = {task: TaskType(entry) for task, entry in kwargs[attr].items() if isinstance(entry, str)}
            elif attr == "domains":  # Dict[str, Keyword]
                kwargs[attr] = {task: Keyword(entry) for task, entry in kwargs[attr].items() if isinstance(entry, str)}

        small_tasks = [t for t, size in kwargs["num_samples"].items() if size < 1000]
        medium_tasks = [t for t, size in kwargs["num_samples"].items() if 1000 <= size < 10000]
        large_tasks = [t for t, size in kwargs["num_samples"].items() if size >= 10000]
        all_infos = cls(small_tasks=small_tasks, medium_tasks=medium_tasks, large_tasks=large_tasks, **kwargs)
        return all_infos


def get_task_infos(task_list: List[str], dims: Optional[str] = None) -> AllTasksInfos:
    """
    Most convenient way to receive a :class:AllTasksInfos instance. Provide a list of aliases and optional a project
    name that computed dimensions before.

    :param List[str] task_list: list of task names, tasks must be available on the machine (run create before if
        not)
    :param Optional[str] dims: (optional) project name that computed dimensions with mml dim proj=THIS_ARG
    :return: relevant meta information on all tasks combined in one object
    :rtype: AllTasksInfos
    """
    _check_init()
    if dims:
        r_conf = OmegaConf.create({"dimension": dims})
        try:
            from mml_dimensionality.scripts.utils import load_dim
        except ImportError:
            raise ImportError("Install mml-dimensionality to use the dim functionality of get_task_infos.")
    else:
        load_dim = None
        r_conf = OmegaConf.create({})
    # initialize empty dicts
    num_classes = {}
    num_samples = {}
    imbalance_ratios = {}
    keywords = {}
    domains = {}
    dimensions = {}
    task_types = {}
    max_resolution = {}
    min_resolution = {}
    domain_list = [
        Keyword.DERMATOSCOPY,
        Keyword.LARYNGOSCOPY,
        Keyword.GASTROSCOPY_COLONOSCOPY,
        Keyword.LAPAROSCOPY,
        Keyword.NATURAL_OBJECTS,
        Keyword.HANDWRITINGS,
        Keyword.CATARACT_SURGERY,
        Keyword.FUNDUS_PHOTOGRAPHY,
        Keyword.MRI_SCAN,
        Keyword.X_RAY,
        Keyword.CT_SCAN,
        Keyword.CLE,
        Keyword.CAPSULE_ENDOSCOPY,
        Keyword.ULTRASOUND,
    ]
    overarching_dataset = {}
    with default_file_manager(reuse_config=r_conf) as fm:
        for task in task_list:
            meta = fm.get_task_info(task, preprocess="none")
            num_classes[task] = len(set(meta["idx_to_class"].values()))
            task_desc = fm.load_task_description(fm.data_path / meta["relative_root"])
            num_samples[task] = task_desc.num_samples
            try:
                imbalance_ratios[task] = max(meta["class_occ"].values()) / min(meta["class_occ"].values())
            except ZeroDivisionError:
                logger.error(f"Division by zero while computing imbalance ratio for task {task}.")
            except ValueError:
                logger.error(f"Unable to compute imbalance ratio for task {task}. Class occ = {meta['class_occ']}")
            keywords[task] = set(meta["keywords"])
            task_types[task] = meta["task_type"]
            sizes = meta["sizes"]
            min_resolution[task] = sizes.min_height * sizes.min_width
            max_resolution[task] = sizes.max_height * sizes.max_width
            domain_candidates = [d for d in keywords[task] if d in domain_list]
            if len(domain_candidates) != 1:
                logger.error(f"Error while searching domain for task {task}, have candidates {domain_candidates}")
            else:
                domains[task] = domain_candidates[0].value
            try:
                plain_task = task.split(TAG_SEP)[0]
                overarching_dataset[task] = get_dset_for_task(task_name=plain_task)
            except KeyError:
                logger.error(f"No registered dataset for task {task}.")
            if dims:
                dimensions[task] = load_dim(fm.reusables[task]["dimension"])
    small_tasks = [t for t, size in num_samples.items() if size < 1000]
    medium_tasks = [t for t, size in num_samples.items() if 1000 <= size < 10000]
    large_tasks = [t for t, size in num_samples.items() if size >= 10000]
    all_infos = AllTasksInfos(
        num_classes=num_classes,
        num_samples=num_samples,
        imbalance_ratios=imbalance_ratios,
        keywords=keywords,
        domains=domains,
        dimensions=dimensions,
        max_resolution=max_resolution,
        min_resolution=min_resolution,
        small_tasks=small_tasks,
        medium_tasks=medium_tasks,
        large_tasks=large_tasks,
        datasets=overarching_dataset,
        task_types=task_types,
    )
    return all_infos


#  Helper functions for rendering and running mml calls in different scenarios


class JobPrefixRequirements:
    """The job prefix requirements to a job. Basically resolves how to invoke mml on the system."""

    def get_prefix(self) -> str: ...


class DefaultRequirements(JobPrefixRequirements):
    """The default how to call MML from e.g. a local machine (assuming it to be installed and the environment to
    be loaded."""

    def get_prefix(self) -> str:
        return "mml"


@dataclasses.dataclass
class MMLJobDescription:
    """
    Combined description of an MML call. Includes prefix requirements, config options and a multirun flag for hpo.
    """

    prefix_req: JobPrefixRequirements
    mode: str
    config_options: Dict[str, Union[str, float, List[Union[str, int, float]], int]]
    multirun: bool = False

    def render(self) -> str:
        """
        Actually renders the job description.

        :return: A string that might be pasted into a terminal to start the job described.
        """
        parts = [self.prefix_req.get_prefix(), self.mode]
        for key, option in self.config_options.items():
            if isinstance(option, str) and " " in option:
                raise ValueError("Found whitespace in JobDescription!")
            # no check for interrupting str in list
            if isinstance(option, list):
                option = str(option).replace(" ", "")
            if key == "mode":
                raise ValueError(
                    "Providing mode inside config_options is not supported anymore, provide mode "
                    "directly to MMLJobDescription."
                )
            parts.append(f"{key}={option}")
        if self.multirun:
            parts.append("--multirun")
        return " ".join(parts)

    def run(self, runner: "JobRunner") -> Optional[float]:
        """
        Runs the job with the given runner.

        :param JobRunner runner: the runner to run the job.
        :return: Potentially a float that represents the return value of the specified experiment (not guaranteed)
        """
        return runner.run(job=self)


class JobRunner:
    """The runner that invokes the rendered MML call."""

    def run(self, job: MMLJobDescription): ...


class EmbeddedJobRunner(JobRunner):
    """
    The embedded runner allows to start mml directly from within the same python interpreter, hence any previous
    variables, imports, etc. are available during runtime. This also allows to receive the return value of MML.
    """

    def run(self, job: MMLJobDescription):
        # override sys.argv
        sys.argv = job.render().split(" ")
        return main()


class SubprocessJobRunner(JobRunner):
    """
    The subprocess runner only inherits the virtual environment but starts a new process including a new interpreter.
    Any variables in the current interpreter will not be available during this run. It does not receive any return
    values of an experiment.
    """

    def run(self, job: MMLJobDescription):
        subprocess.run(job.render().split(" "))


# convenience function for producing long outputs
def write_out_commands(
    cmd_list: List[MMLJobDescription],
    name: str = "output",
    seperator: Optional[str] = "sleep 2\n",
    max_cmds: Optional[int] = None,
) -> None:
    """
    Writes a list of :class:MMLJobDescription into a file that may be called by a shell afterward. This is particularly
    useful if the commands should be transferred to a different host via ssh, e.g. with::

        ssh user@host 'bash -s' < /path/to/output.txt

    :param List[MMLJobDescription] cmd_list: list of commands
    :param Optional[str] name: a file name to relate cmds to a common project or experiment, defaults to 'output'
    :param Optional[str] seperator: (optional) a line seperator, useful if e.g. sleep X should delay cmd submission to
        a cluster
    :param Optional[int] max_cmds: (optional) max number of cmds per file, will split into consecutive files if more
        cmds are present
    :return:
    """
    if seperator is None:
        seperator = ""
    else:
        if len(seperator) > 0 and seperator[-1] != "\n":
            raise ValueError(
                "Seperator does not end line, this may cause interference with cmds and should be avoided!"
            )
    if max_cmds is None:
        max_cmds = len(cmd_list) + 1
    num_splits = math.ceil(len(cmd_list) / max_cmds)
    for split_idx in range(num_splits):
        file_name = name
        if num_splits > 1:
            file_name += f"_{split_idx}"
        file_name += ".txt"
        out = ""
        for cmd in cmd_list[split_idx * max_cmds : (split_idx + 1) * max_cmds]:
            out = out + cmd.render() + "\n" + seperator
        with open(Path(os.path.abspath("")) / file_name, "w") as file:
            file.write(out)
        print(f"Stored {len(cmd_list[split_idx * max_cmds: (split_idx + 1) * max_cmds])} commands at {file_name}.")
