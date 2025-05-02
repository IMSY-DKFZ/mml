# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import os
import shutil
from pathlib import Path

from mml.core.data_preparation.utils import WIPBar
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.schedulers.create_scheduler import CreateScheduler

logger = logging.getLogger(__name__)


def prepare_exp(self: CreateScheduler) -> None:
    """
    New implementation of create schedulers prepare experiment method.

    :param CreateScheduler self: the scheduler instance
    :return:
    """
    try:
        nw_drive = Path(os.getenv("MML_NW_DRIVE", None))
    except TypeError:
        raise MMLMisconfigurationException("MML_NW_DRIVE environment variable is not set. Please add to your mml.env")
    if not nw_drive.exists():
        raise FileNotFoundError(f"MML_NW_DRIVE (={nw_drive}) does not exist")
    nw_downloads = nw_drive / "MedicalMetaLearner" / "DOWNLOADS"
    if not nw_downloads.exists():
        raise RuntimeError("Did someone delete the MML network drive data?")
    data_sets = [args[0] for idx, args in enumerate(self.params) if self.commands[idx].__name__ == "prepare_dataset"]
    available_downloads = [p.name for p in nw_downloads.iterdir() if p.is_dir()]
    for d_set in data_sets:
        if d_set not in available_downloads:
            logger.info(f"Network drive does not offer shortcut for dataset {d_set}")
            continue
        # check if kaggle download, these are not yet compatible with pre-copying
        if (nw_downloads / d_set / "kaggle").exists():
            logger.info(f"Although data for {d_set} exists on the network drive will not download this kaggle dset.")
            continue
        # no we have to copy
        logger.info(f"Found dataset {d_set} on the DKFZ network drive - will try to copy now.")
        target_path = self.fm.get_download_path(dset_name=d_set)
        if len(list(target_path.iterdir())) != 0:
            # already existing downloads, better be careful and not mess something up
            logger.error(
                f"Found existing download data (potentially from previous attempt) at {target_path}. "
                f"This triggers to skip the data copy! If dset creation fails, you might want to "
                f"clean delete {target_path} manually."
            )
            continue
        with WIPBar() as bar:
            bar.desc = "Copying dataset"
            shutil.copytree(src=str(nw_downloads / d_set), dst=str(target_path), dirs_exist_ok=True)
        logger.info(f"Successfully copied dataset {d_set} from network drive to local MML setup at {target_path}.")
