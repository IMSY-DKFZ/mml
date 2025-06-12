# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from mml_lsf.workers import check_lsf_workers

from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.scripts.schedulers.base_scheduler import AFTER_SCHEDULER_INIT_HOOKS

AFTER_SCHEDULER_INIT_HOOKS.append(check_lsf_workers)
MMLFileManager.add_assignment_path(
    obj_cls=None,
    key="lsf_jobs",
    path=Path("PROJ_PATH") / "LSF" / "FILE_NAME",
    enable_numbering=False,
    reusable=False,
)
