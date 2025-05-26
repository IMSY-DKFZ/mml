# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from pytorch_lightning import seed_everything

import mml.core.scripts.utils
from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.data_loading.task_attributes import Keyword, Modality, RGBInfo, Sizes, TaskType
from mml.core.data_loading.task_dataset import TaskDataset
from mml.core.data_loading.task_description import TaskDescription
from mml.core.data_loading.task_struct import TaskStruct, TaskStructFactory
from mml.core.data_preparation.fake_task import create_fake_dset, create_fake_task
from mml.core.scripts.utils import throttle_logging


@pytest.fixture(scope="function")
def fake_task(file_manager):
    dset_path = create_fake_dset()
    task_path = create_fake_task(dset_path)
    file_manager.add_to_task_index(task_path)
    yield


@pytest.fixture(autouse=True)
def no_plugins(monkeypatch, request):
    # do not mock in case the test is marked (with "@pytest.mark.plugin")
    if "plugin" in request.keywords:
        yield
        return

    # mock plugin loading (only effective in local test setups)

    def mock_load_plugins(*args, **kwargs):
        pass

    # deactivate any plugin loading during the runs
    monkeypatch.setattr(mml.core.scripts.utils, "load_mml_plugins", mock_load_plugins)
    print("deactivated plugin loading")
    yield


@pytest.fixture(autouse=True)
def env_variables(monkeypatch, tmp_path_factory, request):
    # do not mock in case the test is marked (with "@pytest.mark.env")
    if "env" in request.keywords:
        yield
        return

    # prevents resolving issues with the config
    def mock_load_env():
        pass

    # first deactivate any env loading during the runs
    monkeypatch.setattr(mml.core.scripts.utils, "load_env", mock_load_env)
    # then set env variables accordingly
    monkeypatch.setenv("MML_CONFIGS_PATH", "DEFAULT_CONF_PATH")
    monkeypatch.setenv("MML_CONFIG_NAME", "config_mml")
    monkeypatch.setenv("MML_DATA_PATH", str(tmp_path_factory.mktemp(basename="data")))
    monkeypatch.setenv("MML_RESULTS_PATH", str(tmp_path_factory.mktemp(basename="results")))
    monkeypatch.setenv("MML_LOCAL_WORKERS", "0")  # test everything single threaded by default
    monkeypatch.setenv("MML_MYSQL_USER", "test")
    monkeypatch.setenv("MML_MYSQL_PW", "test")
    monkeypatch.setenv("MML_HOSTNAME_OF_MYSQL_HOST", "test")
    monkeypatch.setenv("MML_MYSQL_DATABASE", "test")
    monkeypatch.setenv("MML_MYSQL_PORT", "test")
    monkeypatch.setenv("MML_CLUSTER_WORKERS", "test")
    monkeypatch.setenv("MML_CLUSTER_DATA_PATH", "test")
    monkeypatch.setenv("MML_CLUSTER_RESULTS_PATH", "test")
    monkeypatch.setenv("KAGGLE_USERNAME", "test")
    monkeypatch.setenv("KAGGLE_KEY", "test")
    print("monkeypatched environment variables")
    yield


@pytest.fixture
def file_manager(tmp_path_factory, monkeypatch):
    # store class attributes
    assignments_backup = MMLFileManager._path_assignments.copy()
    log_path = tmp_path_factory.mktemp(basename="logging")
    results_root = tmp_path_factory.mktemp(basename="results")
    proj_path = results_root / "test_project"
    proj_path.mkdir()
    monkeypatch.chdir(log_path)
    manager = MMLFileManager(
        data_path=tmp_path_factory.mktemp(basename="data"),
        proj_path=proj_path,
        log_path=log_path,
    )
    yield manager
    try:
        MMLFileManager.clear_instance()
    except KeyError:
        # some routines might clear instance by themselves
        pass
    MMLFileManager._path_assignments = assignments_backup


@pytest.fixture
def dummy_meta_class_path():
    yield Path(__file__).parent / "dummy_meta_class.json"


@pytest.fixture
def dummy_meta_seg_path():
    yield Path(__file__).parent / "dummy_meta_seg.json"


@pytest.fixture
def dummy_fake_model_storage_path():
    yield Path(__file__).parent / "dummy_fake_model_storage.json"


@pytest.fixture
def dummy_fake_predictions_path():
    yield Path(__file__).parent / "dummy_fake_preds.pt"


@pytest.fixture
def dummy_fake_pipeline_path():
    yield Path(__file__).parent / "dummy_fake_pipeline.yaml"


@pytest.fixture
def image() -> np.ndarray:
    return np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mask() -> np.ndarray:
    return np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)


@pytest.fixture
def mml_config():
    with initialize_config_module(config_module="mml.configs", version_base=None):
        cfg = compose(
            config_name="config_mml",
            overrides=["mode.subroutines=[test]", "preprocessing=default", "augmentations=default"],
        )
        cfg["num_workers"] = int(cfg["num_workers"])
        cfg.arch.name = "resnet18"
        cfg.trainer.enable_model_summary = False
        cfg.cbs = {
            "stats": {"_target_": "lightning.pytorch.callbacks.DeviceStatsMonitor"},
            "lrm": {"_target_": "lightning.pytorch.callbacks.LearningRateMonitor"},
        }
        cfg.trainer.min_epochs = 1
        cfg.trainer.max_epochs = 1
        cfg.sampling.sample_num = 20
        cfg.sampling.balanced = False
        cfg.sampling.batch_size = 10
        cfg.sampling.enable_caching = False
        cfg.sampling.cache_max_size = 0
        yield cfg


@pytest.fixture(scope="session", autouse=True)
def deactivate_lightning_logging():
    with throttle_logging(level=logging.WARN, package="pytorch_lightning"):
        yield


@pytest.fixture(autouse=True)
def make_deterministic():
    seed_everything(42)
    yield


@pytest.fixture
def test_task_monkeypatch(file_manager, monkeypatch, image, mask):
    # the test struct that will be returned by the task factory
    test_structs = {
        f"test_task_{x}": TaskStruct(
            name=f"test_task_{x}",
            task_type=TaskType.CLASSIFICATION,
            modalities={Modality.CLASS: "", Modality.IMAGE: "test"},
            means=RGBInfo(*[0.5, 0.5, 0.5]),
            stds=RGBInfo(*[0.1, 0.1, 0.1]),
            sizes=Sizes(*[100, 100, 100, 100]),
            relative_root=f"root_{x}",
            class_occ={"zero": 100, "one": 100, "two": 100},
            preprocessed="none",
            keywords=[Keyword.ARTIFICIAL],
            idx_to_class={0: "zero", 1: "one", 2: "two"},
        )
        for x in "abc"
    }
    test_structs["test_task_d"] = TaskStruct(
        name="test_task_d",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        modalities={Modality.MASK: "", Modality.IMAGE: "test"},
        means=RGBInfo(*[0.5, 0.5, 0.5]),
        stds=RGBInfo(*[0.1, 0.1, 0.1]),
        sizes=Sizes(*[100, 100, 100, 100]),
        relative_root="root_d",
        class_occ={"zero": 100, "one": 100, "two": 100},
        preprocessed="none",
        keywords=[Keyword.ARTIFICIAL],
        idx_to_class={0: "zero", 1: "one", 2: "two"},
    )

    def get_test_struct(self, name):
        return test_structs[name]

    monkeypatch.setattr(target=TaskStructFactory, name="get_by_name", value=get_test_struct)

    # the meta information that will be returned by the file manager
    task_description_class = TaskDescription.from_json(
        {
            "task_type": TaskType.CLASSIFICATION,
            "modalities": {Modality.IMAGE: None, Modality.CLASS: None},
            "idx_to_class": {0: "zero", 1: "one", 2: "two"},
            "class_occ": {"zero": 100, "one": 100, "two": 100},
            "train_folds": [[str(x) for x in range(60 * y, 60 * (y + 1))] for y in range(5)],
            "train_samples": {
                str(x): {Modality.IMAGE: "some_path", Modality.CLASS: np.random.randint(low=0, high=3)}
                for x in range(300)
            },
            "test_samples": {
                str(x): {Modality.IMAGE: "some_path", Modality.CLASS: np.random.randint(low=0, high=3)}
                for x in range(50)
            },
            "name": "test_task_class",
        }
    )
    task_description_seg = TaskDescription.from_json(
        {
            "task_type": TaskType.SEMANTIC_SEGMENTATION,
            "modalities": {Modality.IMAGE: None, Modality.MASK: None},
            "idx_to_class": {0: "zero", 1: "one", 2: "two"},
            "class_occ": {"zero": 100, "one": 100, "two": 100},
            "train_folds": [[str(x) for x in range(60 * y, 60 * (y + 1))] for y in range(5)],
            "train_samples": {str(x): {Modality.IMAGE: "some_path", Modality.MASK: "another_path"} for x in range(300)},
            "test_samples": {str(x): {Modality.IMAGE: "some_path", Modality.MASK: "another_path"} for x in range(50)},
            "name": "test_task_seg",
        }
    )

    def get_test_descriptions(self, path=None):
        # this trick covers both use cases as staticmethod and default class method
        if path is None:
            path = self
        task_type = (
            TaskType.CLASSIFICATION if str(path.stem).split("_")[-1] in "abc" else TaskType.SEMANTIC_SEGMENTATION
        )
        return {TaskType.CLASSIFICATION: task_description_class, TaskType.SEMANTIC_SEGMENTATION: task_description_seg}[
            task_type
        ]

    monkeypatch.setattr(target=MMLFileManager, name="load_task_description", value=get_test_descriptions)

    # the data that will be returned when loading a sample

    def get_test_sample(self, index):
        return {
            Modality.IMAGE.value: np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
            Modality.CLASS.value: np.random.randint(low=0, high=3),
            Modality.MASK.value: np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8),
        }

    monkeypatch.setattr(target=TaskDataset, name="load_sample", value=get_test_sample)

    # set the task index of file manager
    for task in test_structs:
        monkeypatch.setitem(file_manager.task_index, name=task, value={"none": f"{task}.json"})
    yield test_structs
