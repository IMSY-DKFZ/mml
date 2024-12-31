# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path


def load_dim(path: Path) -> int:
    """
    Loads the dimensionality of a task, provided the storage path of that information.

    :param Path path:
    :return:
    """
    with open(path, "r") as file:
        lines = file.readlines()
    assert len(lines) == 1
    return int(lines[0])
