# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from mml_drive.prepare_hook import prepare_exp

from mml.core.scripts.schedulers.create_scheduler import CreateScheduler

# replace default prepare exp definition
CreateScheduler.prepare_exp = prepare_exp
