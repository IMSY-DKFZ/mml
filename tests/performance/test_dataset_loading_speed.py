# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import pytest

from mml.core.data_loading.task_dataset import TaskDataset


@pytest.mark.benchmark(group="fake_data_loading")
@pytest.mark.parametrize("caching", [True, False])
def test_fake_dataset_loading_speed(benchmark, fake_task, file_manager, caching):
    # create struct
    root = file_manager.data_path / file_manager.task_index["mml_fake_task"]["none"]
    if caching:
        cache_limit = 10000
    else:
        cache_limit = 0
    ds = TaskDataset(root=root, caching_limit=cache_limit)
    if caching:
        ds.fill_cache(num_workers=1)

    def loading_dummy():
        out = []
        for sample in ds:
            out.append(sample["class"])
        return out

    result = benchmark(loading_dummy)
    assert len(result) == len(ds)
