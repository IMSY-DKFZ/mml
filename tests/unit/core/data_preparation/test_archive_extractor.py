# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import pytest

from mml.core.data_preparation.archive_extractors import unpack_files
from mml.core.data_preparation.data_archive import DataArchive


def test_invalid_target_path():
    target_path = Path("~/invalid/path/to/store")
    with pytest.raises(FileNotFoundError):
        unpack_files(archives=[], target=target_path)


def test_invalid_file_path(tmp_path):
    target_path = tmp_path / "DSET_001_name"
    target_path.mkdir()
    archive = DataArchive(path=Path("C:/Users/akrit/DKFZ/invalid.txt"))
    with pytest.raises(FileNotFoundError):
        unpack_files(archives=[archive], target=target_path)
