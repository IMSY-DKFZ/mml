# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from mml_lsf.workers import check_lsf_workers

from mml.core.scripts.schedulers.base_scheduler import AFTER_SCHEDULER_INIT_HOOKS

AFTER_SCHEDULER_INIT_HOOKS.append(check_lsf_workers)
