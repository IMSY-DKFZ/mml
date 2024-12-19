# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import shutil
import sys

import humanize
from _collections import OrderedDict
from omegaconf import DictConfig

from mml.core.data_preparation.registry import get_dset_for_task
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import TAG_SEP

logger = logging.getLogger(__name__)


class CleanScheduler(AbstractBaseScheduler):
    """
    AbstractBaseScheduler implementation for the cleaning of files. Includes the following subroutines:
    - temp
    - download
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(CleanScheduler, self).__init__(cfg=cfg, available_subroutines=["temp", "download"])
        logger.info(
            "Clean mode helps removing file artefacts, cleaning your MML setup but also may disrupt running "
            "MML processes. Please ensure not to run in parallel with any other MML command. By default "
            "clean runs with mode.force=false and requires interactive approval, you may choose to set "
            "force=true to omit this interactive part."
        )
        self.sizes = []

    def prepare_exp(self) -> None:
        """
        We skip task struct creation, since it is not needed or may not be finished.
        """
        logger.debug("Skip task struct creation.")

    def create_routine(self):
        """
        This scheduler implements two subroutines, one for temp files cleaning and one for downloads cleaning.

        :return: None
        """

        if self.cfg.mode.all:
            task_list = [task.split(TAG_SEP)[0] for task in self.fm.task_index.keys()]
            # for backward compatibility with previous tag separators
            task_list = [task.split(" ")[0] for task in task_list]
        else:
            task_list = [task.split(TAG_SEP)[0] for task in self.cfg.task_list]
        # remove duplicates (e.g. via multiple existing tagged variants of a task)
        task_list = list(OrderedDict.fromkeys(task_list))
        # -- add temp commands
        if "temp" in self.subroutines:
            for task in task_list:
                self.commands.append(self.remove_temp_files)
                self.params.append([task])
        # -- add download commands
        if "download" in self.subroutines:
            # run only once per dset
            all_dsets = dict()  # use dict instead of set zo ensure order
            for task in task_list:
                try:
                    dset = get_dset_for_task(task)
                    all_dsets[dset] = None
                except KeyError:
                    logger.error("No registered dset link for task {}".format(task))
            for dset in all_dsets:
                self.commands.append(self.remove_downloads)
                self.params.append([dset])

    def remove_temp_files(self, task_name: str) -> None:
        """
        Routine to remove temporary files that may remain as artefacts during task creation. They are located inside
        the data path either below RAW or PREPROCESSED inside the dataset folders and are named temp.json or temp_X.json
        with integer X.

        :param str task_name: name of the task to remove temp files (will concern both RAW and PREPROCESSED variants)
        :return: None
        """
        logger.info("Starting removing of temp files for " + self.highlight_text(task_name))
        # collect all files
        to_remove = []
        dset_roots = [self.fm.raw_data] + list(self.fm.preprocessed_data.iterdir())
        for root in dset_roots:
            for dset_path in root.iterdir():
                # glob over possible files
                for file_path in dset_path.glob("temp*.json"):
                    try:
                        if self.fm.load_task_description_header(file_path).name == task_name:
                            to_remove.append(file_path)
                    except RuntimeError:
                        # potentially the task description is formatted in an old way
                        to_remove.append(file_path)
        if len(to_remove) > 0:
            logger.warning(f"Found {len(to_remove)} temp files for task {task_name}. Will remove now.")
        else:
            logger.info(f"Found no temp file for task {task_name}.")
        for tmp_file in to_remove:
            if not self.cfg.mode.force:
                print(f"Are you sure to remove {tmp_file}? [Y/n]")
                answer = input()
                if answer.lower() not in ["y", "n", ""]:
                    print("Invalid input, will cancel clean scheduler.")
                    self.log_cumulative_sizes()
                    sys.exit(1)
                if answer.lower() == "n":
                    print(f"Will not remove {tmp_file}.")
                    continue
            self.sizes.append(tmp_file.stat().st_size)
            tmp_file.unlink()
        logger.info("Finished removing temp files for " + self.highlight_text(task_name))

    def remove_downloads(self, dset_name: str) -> None:
        """
        Routine to remove the downloads of a dataset. May remove data for multiple tasks at once! Make sure that
        original download data is still available later on for full reproducibility.

        :param str dset_name: name of the dset
        :return: None
        """
        logger.info("Starting removing downloads for " + self.highlight_text(dset_name))
        d_path = self.fm.get_download_path(dset_name=dset_name)
        # gather size
        size = sum(f.stat().st_size for f in d_path.glob("**/*") if f.is_file())
        logger.warning(
            f"Will remove {humanize.naturalsize(size, gnu=True)} bytes in downloads for dataset "
            f"{dset_name} at {d_path}."
        )
        do_remove = True
        if not self.cfg.mode.force:
            print(f"Are you sure to remove {d_path}? [Y/n]")
            answer = input()
            if answer.lower() not in ["y", "n", ""]:
                print("Invalid input, will cancel clean scheduler.")
                self.log_cumulative_sizes()
                sys.exit(1)
            if answer.lower() == "n":
                print(f"Will not remove {d_path}.")
                do_remove = False
        if do_remove:
            self.sizes.append(size)
            shutil.rmtree(d_path)
        logger.info("Finished removing downloads for " + self.highlight_text(dset_name))

    def log_cumulative_sizes(self) -> None:
        logger.info(f"Cumulative sizes for removed files are {humanize.naturalsize(sum(self.sizes), gnu=True)} bytes.")

    def before_finishing_hook(self):
        self.log_cumulative_sizes()
