# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest

from mml.core.scripts.schedulers.info_scheduler import InfoScheduler


@pytest.mark.benchmark(group="scheduler_init")
def test_scheduler_init(mml_config, benchmark, file_manager):
    mml_config["mode"]["subroutines"] = ["tasks"]

    def clean_scheduler_reinit():
        scheduler = InfoScheduler(mml_config)
        scheduler.lock_path.unlink()
        scheduler.planned_schedule.unlink()
        scheduler.status_log.unlink()

    benchmark(clean_scheduler_reinit)
