# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

# enables a special test to check if installed tasks are consistent with their current installation code
def pytest_addoption(parser):
    parser.addoption(
        "--check_tasks",
        action="store_true",
        dest="check_tasks",
        default=False,
        help="single test to check the instantiated task meta information",
    )
