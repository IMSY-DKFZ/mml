# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest

from mml.core.data_preparation.dset_creator import DSetCreator


def test_data_type(file_manager):
    # dset creator uses file manager instance
    url = "http://test/images.tar"
    file_name = "images.tar"
    data_kind = "invalid"
    dset_name = "DSET001_name"
    creator = DSetCreator(dset_name)
    with pytest.raises(TypeError, match=r"unsupported operand"):
        creator.download(url, file_name, data_kind)
