# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import pytest
import requests

from mml.core.data_preparation.utils import download_file


def test_invalid_path():
    non_existent_path = Path("/DOWNLOADS/DSET")
    file_name_1 = "images.tar"
    download_url = "http://test/images.tar"
    with pytest.raises(ValueError, match=r"Invalid path"):
        download_file(path_to_store=non_existent_path, download_url=download_url, file_name=file_name_1)


class MockResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # prevents http error
    @staticmethod
    def raise_for_status():
        pass

    # necessary for progress bar
    @property
    def headers(self):
        return {"content-length": 10000}

    # nonsense data that is returned
    @staticmethod
    def iter_content(chunk_size):
        for _ in range(10000 // chunk_size):
            yield bytes(chunk_size)
        yield bytes(10000 % chunk_size)


def test_valid_download_path(tmp_path, monkeypatch):
    # going to monkeypatch the http request, giving the mocked response above
    def mock_get(*args, **kwargs):
        return MockResponse()

    # apply the monkeypatch for requests.get to mock_get
    monkeypatch.setattr(requests, "get", mock_get)
    download_file(path_to_store=tmp_path, download_url="http://test/images.tar", file_name="images.tar")
