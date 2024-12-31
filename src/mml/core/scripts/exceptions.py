# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#


class MMLMisconfigurationException(Exception):
    """Exception used to inform users of misuse with MML."""


class InvalidTransitionError(Exception):
    """Raised whenever a transition between states of a class is invalid."""


class TaskNotFoundError(Exception):
    """Raised when there is an unsolvable error to find a certain task."""
