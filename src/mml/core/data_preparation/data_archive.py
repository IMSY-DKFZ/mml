# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mml.core.scripts import utils as core_utils

logger = logging.getLogger(__name__)


class DataKind(core_utils.StrEnum):
    """
    Kinds of data. Used to somehow sort into distinct top level folders. If multiple kinds are mixed, MIXED should be
    used as default. Usage is not enforced but may help to structure any data storage.
    """

    MIXED = "mixed_data"
    UNLABELED_DATA = "unlabeled_data"
    TRAINING_DATA = "training_data"
    TRAINING_LABELS = "training_labels"
    TESTING_DATA = "testing_data"
    TESTING_LABELS = "testing_labels"


@dataclass
class DataArchive:
    """A simple dataclass holding information about an data archive (e.g. a zipfile)."""

    path: Path  # path to the archive
    kind: DataKind = DataKind.MIXED  # datakind
    md5sum: Optional[str] = None  # is there a md5 sum?
    password: Optional[str] = None  # is there a password encryption?
    keep_top_level: bool = False  # should an additional layer be created during extraction?

    def check_hash(self) -> None:
        """
        Checks if the optional md5sum of the DataArchive matches the actual files md5sum.
        """
        if self.md5sum is None:
            return
        block_size = 65536
        hasher = hashlib.md5()
        with open(str(self.path), "rb") as file:
            buf = file.read(block_size)
            while len(buf) > 0:
                hasher.update(buf)
                buf = file.read(block_size)
        if hasher.hexdigest() != self.md5sum:
            raise RuntimeError(
                f"incorrect md5sum for file {self.path.name}, should be {self.md5sum} but is {hasher.hexdigest()}"
            )
        logger.info(f"file {self.path.name} has correct hash!")
