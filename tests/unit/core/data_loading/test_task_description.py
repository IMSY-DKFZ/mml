# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from mml.core.data_loading.task_description import STRUCT_REQ_HEADER_KEYS
from mml.core.data_loading.task_struct import TaskStruct


def test_required_header_keys():
    struct_init_keys = set(TaskStruct.__init__.__annotations__.keys())
    treated_sep = {"relative_root", "preprocessed", "name"}
    assert set(STRUCT_REQ_HEADER_KEYS).union(treated_sep) == struct_init_keys
