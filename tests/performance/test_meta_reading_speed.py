# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest

from mml.core.data_loading.file_manager import MMLFileManager


@pytest.mark.benchmark(group="meta_header_loading")
def test_meta_header_loading_speed(benchmark, dummy_meta_class_path):
    benchmark(MMLFileManager.load_task_description_header, dummy_meta_class_path)


@pytest.mark.benchmark(group="full_header_loading")
def test_full_meta_loading_speed(benchmark, dummy_meta_class_path):
    benchmark(MMLFileManager.load_task_description, dummy_meta_class_path)
