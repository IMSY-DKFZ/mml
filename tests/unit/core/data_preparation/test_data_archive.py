# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest

from mml.core.data_preparation.data_archive import DataArchive


def test_check_hash_success(tmp_path):
    tmp_path /= "test.txt"
    data = "some sample text to be stored and hashed"
    with open(str(tmp_path), "w") as file:
        file.write(data)
    archive = DataArchive(path=tmp_path, md5sum="307f768a68e5ab1918ece2cb99d72d90")
    archive.check_hash()


def test_check_hash_error(tmp_path):
    tmp_path /= "test_fail.txt"
    data = "some short random text"
    with open(str(tmp_path), "w") as file:
        file.write(data)
    archive = DataArchive(path=tmp_path, md5sum="incorrecthashvalue")
    with pytest.raises(RuntimeError):
        archive.check_hash()
