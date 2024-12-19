# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path


def show_logo(indent: int = 0):
    """
    Prints the mml logo in ascii art to the terminal.

    :param int indent: specifies the number of blanks prepended
    :return:
    """
    logo_path = Path(__file__).parent / "mml_logo.txt"
    with open(logo_path, "r") as file:
        lines = file.readlines()
    for line in lines:
        print(" " * indent + line.rstrip())
