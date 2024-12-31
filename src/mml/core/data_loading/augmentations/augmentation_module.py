# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from omegaconf import ListConfig, OmegaConf

from mml.core.data_loading.task_attributes import RGBInfo
from mml.core.scripts.utils import StrEnum

IMAGENET_AA_PATH = Path(__file__).parent / "imagenet.json"
Transform = Callable[[Dict[str, Any]], Dict[str, Any]]


class DataFormat(StrEnum):
    SINGLE_SAMPLE_DICT = "single_sample_dict"  # Dict[str, Union[torch.Tensor, str]] where tensor is single sample
    BATCHED_SAMPLE_DICTS = "batched_sample_dicts"  # Dict[str, Union[torch.Tensor, List[str]]] where tensor is batched
    MULTI_TASK_SAMPLE_DICTS = "multitask_sample_dicts"  # Dict[str, Dict[str ,Union[torch.Tensor, List[str]]]] (batched)


class AugmentationModule(ABC):
    def __init__(
        self,
        device: str,
        cfg: Union[ListConfig, List[Dict[str, Any]]],
        is_first: bool,
        is_last: bool,
        means: Optional[RGBInfo],
        stds: Optional[RGBInfo],
    ):
        self.data_format = None
        self.device = device
        self.cfg = OmegaConf.create(cfg)
        self.is_first = is_first
        self.is_last = is_last
        self.means = means
        self.stds = stds
        self.pipeline = None
        self._build_pipeline()
        if self.pipeline is None:
            raise RuntimeError("AugmentationModule instantiated incorrectly!")

    def __len__(self):
        return len(self.pipeline)

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """The intended way to use AugmentationModule is to call with data argument."""
        if self.device == "cpu":
            inpt = kwargs
        elif self.device == "gpu":
            inpt = args[0]  # batch and dataloader_idx on gpu
        else:
            raise RuntimeError("invalid device!")
        # during first call interpret data format and sanity check
        if self.data_format is None:
            self.data_format = self.get_data_format(inpt)
            self._sanity_check(inpt)
        return self._forward_impl(inpt)

    @abstractmethod
    def _build_pipeline(self):
        """Read in config."""
        pass

    @abstractmethod
    def _forward_impl(self, inpt: Any) -> Any:
        """Apply augmentation on input."""
        pass

    @abstractmethod
    def _sanity_check(self, inputs: Any) -> None:
        """Make sure the input fits to the pipeline."""
        pass

    @staticmethod
    def get_data_format(inpt: Any) -> DataFormat:
        """Checks the input data format."""
        # check if valid format at all
        if not isinstance(inpt, dict):
            raise ValueError(f"Input data format {inpt} is not a dict")
        entry = next(iter(inpt.values()))
        # check if multi level dict for multi task samples
        if isinstance(entry, dict):
            return DataFormat.MULTI_TASK_SAMPLE_DICTS
        # next check if dict contains batched elements
        for vals in inpt.values():
            if isinstance(vals, list):
                return DataFormat.BATCHED_SAMPLE_DICTS
            if isinstance(vals, torch.Tensor):
                if vals.ndim == 4:
                    return DataFormat.BATCHED_SAMPLE_DICTS
        # otherwise this is a single sample dict
        return DataFormat.SINGLE_SAMPLE_DICT


class AugmentationModuleContainer:
    def __init__(self, modules: List[AugmentationModule]):
        self.modules = modules
        if not all(isinstance(mod, AugmentationModule) for mod in self.modules):
            raise ValueError("only AugmentationModules may be passed to AugmentationModuleContainer.")
        if any(mod.device != "cpu" for mod in self.modules):
            raise ValueError("container can only be initialized with cpu device!")
        if len(self.modules) == 0:
            raise ValueError("requires at least one module to be passed!")

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """Pass as kwargs through each module."""
        for mod in self.modules:
            kwargs = mod(**kwargs)
        return kwargs

    def __len__(self) -> int:
        return sum(len(mod) for mod in self.modules)
