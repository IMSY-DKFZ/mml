# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import tempfile
import warnings
from pathlib import Path
from typing import List, Tuple

import omegaconf.listconfig
import orjson
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

import mml
from mml.core.data_loading.file_manager import TASK_PREFIX, MMLFileManager
from mml.core.data_loading.task_description import ALL_TASK_DESCRIPTION_KEYS
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import ask_confirmation

logger = logging.getLogger(__name__)


class UpgradeScheduler(AbstractBaseScheduler):
    """
    AbstractBaseScheduler implementation for the Dataset and MML Upgrade process. Includes the following subroutines:
    - upgrade
    - downgrade
    For upgrade we always assume to upgrade to the currently installed version of MML. For downgrade, we assume that
    previously MML has been upgraded to the currently installed version and is now downgraded to the specified version
    in cfg.mode.version.
    """

    def __init__(self, cfg: DictConfig):
        # make sure to create MMLFileManager beforehand to avoid RunTimeErrors during super.__init__
        MMLFileManager(
            proj_path=Path(tempfile.mkdtemp()),
            data_path=Path(tempfile.mkdtemp()),
            log_path=Path(tempfile.mkdtemp()),
            reuse_cfg=None,  # nothing to reuse here
            remove_cfg=None,  # nothing to remove either
        )
        # assert correct configuration
        if OmegaConf.is_missing(cfg.mode, "version"):
            raise MMLMisconfigurationException(
                "You are up/downgrading the mml setup without specifying a version! "
                "In case of upgrading please provide a source version, in case of "
                "downgrading please provide a target version. If in doubt, please "
                "read the documentation!"
            )
        if (
            not isinstance(cfg.mode.version, omegaconf.listconfig.ListConfig)
            and len(cfg.mode.version) == 3
            and all([isinstance(elem, int) for elem in cfg.mode.version])
        ):
            raise MMLMisconfigurationException("Specify source/target version as list of three int.")
        # tuple variant for better compatibility
        if isinstance(cfg.mode.version, list):
            self.version = tuple(cfg.mode.version)
        elif isinstance(cfg.mode.version, str):
            self.version = tuple([int(x) for x in cfg.mode.version.split(".")])
        else:
            raise MMLMisconfigurationException("Provide mode.version either as list (aka [x,y,z]) or as str (x.y.z).")
        if len(self.version) != 3:
            raise MMLMisconfigurationException("Provide mode.version as major - minor - patch (list or str).")
        if "upgrade" in list(cfg.mode.subroutines) and "downgrade" in list(cfg.mode.subroutines):
            raise MMLMisconfigurationException("Upgrade mode may either be used to upgrade or to downgrade.")
        self.upgrading = "upgrade" in list(cfg.mode.subroutines)
        # initialize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(UpgradeScheduler, self).__init__(cfg=cfg, available_subroutines=["upgrade", "downgrade"])
        # since self.fm does not help with correct paths, we store the correct ones as well for the use in the scheduler
        self.data_path = Path(self.cfg["data_dir"])

    def prepare_exp(self) -> None:
        """
        Prepare experiment expects tasks to be present and loads these into task factory container. Here this should
        be avoided.
        """
        logger.debug("Skipping experiment setup!")

    def create_routine(self) -> None:
        """
        This scheduler implements two subroutines, one for dataset preparation and one for task preparation.

        :return: None
        """
        # determine necessary patches from dict of all available patches
        # pattern: Key=Version of chance - Value: function to up AND downgrade
        patches = {(0, 12, 0): self.upgrade_0_12}
        # sort and filter according to subroutine
        patch_ids: List[Tuple[int, int, int]] = sorted(
            filter(lambda x: self.version < x <= mml.VERSION, patches.keys()), reverse=not self.upgrading
        )
        # -- add commands
        for patch_id in patch_ids:
            self.commands.append(patches[patch_id])
            self.params.append([])
        if len(patch_ids) == 0:
            logger.info("No patches necessary!")
        else:
            # ensure user is aware of implications
            msg = (
                f"You are about to {'upgrade' if self.upgrading else 'downgrading'} your MML environment, "
                f"from version {self.version if self.upgrading else mml.VERSION} to version "
                f"{mml.VERSION if self.upgrading else self.version}. Although the effects should be revertible it "
                f"is recommended to create a backup of your data and/or results! Do you want to continue? "
                f'Please type "y"'
            )
            try:
                confirmed = ask_confirmation(self.highlight_text(msg))
            except TimeoutError:
                logger.error("No input provided for necessary response, will kill this run!")
                raise
            if not confirmed:
                raise RuntimeError('Stopped MML up/downgrade scheduler. To up/downgrade rerun and answer "y".')

    def upgrade_0_12(self) -> None:
        """
        This performs the necessary updates to results and data from 0.11 to 0.12 version of mml-core.
        Iterate over all installed tasks and update the keys: tags, train_tuples, unlabeled_tuples, test_tuples
        """
        logger.info("Now rolling patch 0.12")
        all_task_descriptions = self._get_all_task_descriptions()
        logger.info(f"Found {len(all_task_descriptions)} task descriptions to update.")
        # processing each task description
        for description_path in track(all_task_descriptions, description="Updating task descriptions..."):
            # load
            with open(str(description_path), "rb") as f:
                data_dict = orjson.loads(f.read())
            # replace (format is new : old)
            replacements = {
                "keywords": "tags",
                "train_samples": "train_tuples",
                "name": "alias",
                "unlabeled_samples": "unlabeled_tuples",
                "test_samples": "test_tuples",
            }
            # ensuring correct order for fast loading of header
            new_data_dict = {}
            # some keys got deprecated, will be added when downgrading (this causes information loss during upgrading)
            if not self.upgrading:
                for key in ["task_id", "orig_performance", "top_performance"]:
                    new_data_dict[key] = (
                        description_path.parent.name + "%" + description_path.stem if key == "task_id" else ""
                    )
            # these are all remaining entries (sorted via the new definition)
            for key in ALL_TASK_DESCRIPTION_KEYS:
                # these keys have been renamed
                if key in replacements:
                    current = replacements[key] if self.upgrading else key
                    target = key if self.upgrading else replacements[key]
                else:
                    current = target = key
                if current not in data_dict:
                    raise RuntimeError(f"Did not find {current} in {description_path}.")
                new_data_dict[target] = data_dict[current]
            # write
            with open(str(description_path), "wb") as f:
                f.write(orjson.dumps(new_data_dict))
            print(description_path)
        logger.info("Done rolling patch 0.12.")

    def _get_all_task_descriptions(self) -> List[Path]:
        """
        Helper function ro receive all task descriptions of the installation.

        :return: List of paths to all .json files defining task descriptions.
        """
        # gather all TASK descriptions
        all_task_descriptions = []
        for dataset in (self.data_path / "RAW").iterdir():
            if not dataset.is_dir():
                continue
            all_task_descriptions.extend(list(dataset.glob("".join(["[" + c + "]" for c in TASK_PREFIX]) + "*.json")))
        # now preprocessed folder
        for preprocess in (self.data_path / "PREPROCESSED").iterdir():
            if not preprocess.is_dir():
                continue
            for dataset in preprocess.iterdir():
                if not dataset.is_dir():
                    continue
                all_task_descriptions.extend(
                    list(dataset.glob("".join(["[" + c + "]" for c in TASK_PREFIX]) + "*.json"))
                )
        return all_task_descriptions
