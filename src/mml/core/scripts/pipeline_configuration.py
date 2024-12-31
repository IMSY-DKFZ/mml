# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.data_loading.task_struct import TaskStruct

logger = logging.getLogger(__name__)


class PipelineCfg:
    def __init__(self, pipeline_cfg: DictConfig, restrict_keys: Optional[List[str]] = None) -> None:
        """
        PipelineCfg holds relevant configuration elements of a training pipeline to store, reproduce and leverage
        knowledge at a later point. The intended usage is by invoking :meth:`from_cfg` on the full mml config, which
        will produce a masked copy only focussing on a subset of config keys.

        :param DictConfig pipeline_cfg: a config
        :param Optional[List[str]] restrict_keys: which config keys to focus upon
        """
        self.pipeline_cfg = pipeline_cfg
        self.pipeline_keys = []
        restrict_keys = PIPELINE_CONFIG_PARTS if restrict_keys is None else restrict_keys
        for key in restrict_keys:
            if not isinstance(key, str):
                raise ValueError("provide pipeline keys as strings")
            if not hasattr(self.pipeline_cfg, key):
                warnings.warn(f"requested key {key} not found in pipeline_cfg, will be ignored")
                continue
            self.pipeline_keys.append(key)
        if len(self.pipeline_keys) == 0:
            raise ValueError("(valid) pipeline keys are empty")
        # reduce pipeline_config, this is a fallback if called directly upon a full config, but also in case at some
        # point only a subset of an existing pipeline configuration is intended to be reused
        self.pipeline_cfg = OmegaConf.masked_copy(self.pipeline_cfg, keys=self.pipeline_keys)

    @classmethod
    def from_cfg(cls, current_cfg: DictConfig, restrict_keys: Optional[List[str]] = None) -> "PipelineCfg":
        """
        Extracts relevant pipeline keys from current config determined by restrict_keys.

        :param DictConfig current_cfg: the FULL config to derive the pipeline configuration from
        :param Optional[List[str]] restrict_keys: which config keys to focus upon
        :return:
        """
        pipeline_keys = PIPELINE_CONFIG_PARTS if restrict_keys is None else restrict_keys
        if not all([isinstance(key, str) and hasattr(current_cfg, key) for key in pipeline_keys]):
            raise ValueError(f"keys {pipeline_keys} contains a value, which might be not present in the current config")
        return cls(pipeline_cfg=OmegaConf.masked_copy(current_cfg, keys=pipeline_keys), restrict_keys=pipeline_keys)

    @contextmanager
    def activate(self, current_cfg: DictConfig) -> None:
        """
        To be used as a config manager, activates this pipeline upon the currently active mml config. When the context
        exits, the original configuration is restored.

        :param DictConfig current_cfg: the currently active mml config
        :return: no return value, the mml config is modified in place
        """
        # create backup for later restoration
        old = deepcopy(current_cfg)
        # set config elements based on keys
        for key in self.pipeline_keys:
            if key in self.pipeline_cfg:
                OmegaConf.update(current_cfg, key=key, value=self.pipeline_cfg[key], merge=False, force_add=False)
                logger.debug(f"Activated key {key} from pipeline configuration.")
        # yield to do training etc.
        yield
        # restore old configuration
        for key in self.pipeline_keys:
            OmegaConf.update(current_cfg, key=key, value=old[key], merge=False, force_add=False)
        logger.debug("Deactivated pipeline configuration.")

    def store(self, task_struct: TaskStruct, as_blueprint: bool = False) -> Path:
        """
        Store this pipeline. Requires a task struct to determine task name. If blueprint is set, this will be stored in
        the BLUEPRINTS folder instead of PIPELINES. This allows for easier re-usage.

        :param TaskStruct task_struct: struct of the task this pipeline has or should be applied upon
        :param bool as_blueprint: if true store as blueprint otherwise as pipeline
        :return: the path to the stored file
        """
        # stores pipeline_cfg
        key = "blueprint" if as_blueprint else "pipeline"
        path = MMLFileManager.instance().construct_saving_path(
            obj=self.pipeline_cfg, key=key, task_name=task_struct.name
        )
        # do not resolve to not overwrite
        OmegaConf.save(config=self.pipeline_cfg, f=path, resolve=False)
        return path

    @classmethod
    def load(cls, path: Path, pipeline_keys: Optional[List[str]] = None) -> "PipelineCfg":
        """
        Load a stores pipeline configuration (or blueprint) from path.

        :param Path path: the path to the stored file
        :param Optional[List[str]] pipeline_keys: which config keys to focus upon
        :return: the loaded pipeline configuration (restricted to provided pipeline keys)
        """
        loaded = OmegaConf.load(path)
        logger.debug(f"Loaded pipeline from {path}.")
        return cls(pipeline_cfg=loaded, restrict_keys=pipeline_keys)

    def clone(self) -> "PipelineCfg":
        """
        Convenience method to deepcopy the configuration.

        :return: a deepcopy of the configuration
        """
        return deepcopy(self)


# these are the default keys if none are specified, they comprise all relevant aspects for reproducing a model training
PIPELINE_CONFIG_PARTS = [
    "arch",
    "augmentations",
    "cbs",
    "loss",
    "lr_scheduler",
    "mode",
    "optimizer",
    "preprocessing",
    "sampling",
    "trainer",
    "tta",
    "tune",
]
