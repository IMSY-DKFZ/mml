# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra.utils
import peft
import torch
from humanize import intword
from peft import LoraConfig
from peft.config import PeftConfig
from sqlalchemy.testing.plugin.plugin_base import warnings
from transformers import Conv1D

from mml.core.data_loading.task_attributes import RGBInfo, TaskType
from mml.core.data_loading.task_struct import TaskStruct
from mml.core.scripts.exceptions import MMLMisconfigurationException

logger = logging.getLogger(__name__)


class BaseModel(torch.nn.Module, ABC):
    def __init__(self, **kwargs):
        """
        The base class for MML models. Derived classes must implement the following methods:
         - :meth:`_init_model` - for backbone initialization
         - :meth:`_create_head` - for head creation
         - :meth:`supports` - reporting supported task types
         - :meth:`forward` - the models forward pass through backbone and all heads
         - :meth:`forward_features`  - alternative usage as feature extractor
        """
        super(BaseModel, self).__init__()
        # model requirements
        self.required_mean: Optional[RGBInfo] = None  # mean expected by model
        self.required_std: Optional[RGBInfo] = None  # std expected by model
        self.input_size = (None, None, None)  # channel, height, width - will be defined during init
        # nn modules
        self.backbone: Union[torch.nn.Module, None] = None
        self.heads = torch.nn.ModuleDict({})
        # for freezing functionality
        self._frozen_params: List[str] = []
        # store init kwargs
        self._init_kwargs: Dict[str, Any] = kwargs  # stores stuff that needs to be persistent when re-initializing
        self._head_init_kwargs: List[Dict[str, Any]] = []  # stores init kwargs of heads
        self._peft_kwargs: Dict[str, Any] = {}  # stores any peft kwargs
        # actually init backbone
        self._init_model(**kwargs)
        logger.debug("Model initialised.")

    @abstractmethod
    def _init_model(self, **kwargs: Any) -> None:
        """
        This shall implement the backbone module as well as potentially load pretrained weights thereof.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_head(self, task_type: TaskType, num_classes: int, **kwargs: Any) -> BaseHead:
        """
        This shall implement the creation of heads. Given a certain task type the head must be able to be attached
        to the backbone as implemented by the forward method.
        """
        raise NotImplementedError

    @abstractmethod
    def supports(self, task_type: TaskType) -> bool:
        """
        Whether the model supports a given task type.

        :param TaskType task_type:
        :return: true iff task type is supported by model
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Model forward functionality. Passes input through the backbone once and forwards output through each head.
        :param torch.tensor x: input tensor
        :return: a dictionary, with one entry per head, key is head name and value is head output
        """
        pass

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature extraction functionality. Only forwards features through the backbone and post-processes them to 1D.
        :param  torch.tensor x: input tensor
        :return: a 1D tensor
        """
        pass

    def add_head(self, task_struct: TaskStruct, **kwargs: Any) -> None:
        """
        The functionality used to add heads to a model.

        :param TaskStruct task_struct: struct for the task to add a head for
        :param Any kwargs: additional kwargs that will be forwarded
        """
        if task_struct.name in self.heads:
            raise KeyError(f"You cannot register a head with already present name ({task_struct.name}).")
        if not self.supports(task_type=task_struct.task_type):
            raise RuntimeError(f"Task type {task_struct.task_type} not supported by model.")
        init_kwargs = {"task_type": task_struct.task_type, "num_classes": task_struct.num_classes}
        init_kwargs.update(kwargs)
        self.heads[task_struct.name] = self._create_head(**init_kwargs)
        self._head_init_kwargs.append(init_kwargs)
        logger.debug(
            f"Added head {task_struct.name} of task type {task_struct.task_type} with "
            f"{task_struct.num_classes} classes."
        )

    def count_parameters(self, only_trainable: bool = True) -> Dict[str, int]:
        """
        Gives information on parameter count of the model.

        :param bool only_trainable: if True, only counts parameters that requires_grad.
        :return: a dict with component names as key (backbone or name of heads) and parameter count as value
        """
        info_dict = {}
        for name, module in [("backbone", self.backbone)] + list(self.heads.items()):
            if module is None:
                info_dict[name] = 0
            else:
                info_dict[name] = sum(p.numel() for p in module.parameters() if p.requires_grad or not only_trainable)
        return info_dict

    def freeze_backbone(self) -> None:
        """
        Freezes all backbone parameters.
        """
        for name, par in self.backbone.named_parameters():  # type: ignore[union-attr]
            if par.requires_grad:
                par.requires_grad = False
                self._frozen_params.append(name)
        logger.debug(f"Froze {len(self._frozen_params)} parameters of model.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreezes previously frozen backbone parameters.
        """
        for name, par in self.backbone.named_parameters():  # type: ignore[union-attr]
            if name in self._frozen_params:
                par.requires_grad = True
        logger.debug(f"Unfroze {len(self._frozen_params)} params of model.")
        self._frozen_params = []

    def set_peft(self, peft_cfg: PeftConfig) -> None:
        """
        Applies a PEFT (Parameter Efficient FineTuning) method to the model. Usually this will lead to adapters injected
        to the base model that complement existing weights. The advantage is that the majority of existing weights is
        frozen (the .requires_grad attribute of the tensors is set to false) while only the smaller adapters are kept
        trainable.

        :param PeftConfig peft_cfg: PEFTConfig instance, see
            `huggingface/peft <https://github.com/huggingface/peft/tree/main>`_
        :return: None, since model is mofified in place
        """
        if self._peft_kwargs:
            raise RuntimeError("PEFT already set for this model!")
        if self._frozen_params:
            warnings.warn(
                "Backbone was frozen prior to applying PEFT, will first unfreeze backbone and then apply."
                "You may re-freeze the backbone (i.e. the injected adapters)."
            )
            self.unfreeze_backbone()
        if peft_cfg.is_prompt_learning or peft_cfg.is_adaption_prompt:
            raise MMLMisconfigurationException(f"Applying {peft_cfg.peft_type} is likely an unsupported PEFT type.")
        self._peft_kwargs = peft_cfg.to_dict()
        if isinstance(peft_cfg, LoraConfig) and peft_cfg.target_modules == "auto":
            peft_cfg.target_modules = self.get_lora_compatible_layers(self.backbone)
            logger.info(f"Auto detected {len(peft_cfg.target_modules)} compatible layers for LoRa in model backbone.")
        pre_params = self.count_parameters(only_trainable=True)["backbone"]
        self.backbone = peft.get_peft_model(model=self.backbone, peft_config=peft_cfg)
        post_params = self.count_parameters(only_trainable=True)["backbone"]
        logger.info(
            f"After applying {peft_cfg.peft_type} from {intword(pre_params)} params only {intword(post_params)}"
            f" remain trainable (={post_params / pre_params:.2%})."
        )

    @staticmethod
    def get_lora_compatible_layers(backbone: torch.nn.Module) -> List[str]:
        """
        Helper function to extract all Lora compatible layers (from the peft library).

        :param torch.nn.Module backbone: the model to extract layers from
        :return: list of strings the correspond to the respective layer names
        """
        layer_names = []
        for name, module in backbone.named_modules():
            # these are the currently LORA supported layers
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
                layer_names.append(name)
        return layer_names

    @staticmethod
    def load_checkpoint(param_path: Union[Path, str]) -> "BaseModel":
        """
        Load from a checkpoint. Be aware that MML uses its own checkpoint structure (different from the one in
        `lightning <https://github.com/Lightning-AI/lightning>`_). Detail can be found in
        :meth:`~mml.core.models.torch_base.BaseModel.save_checkpoint`.

        :param Union[Path, str] param_path: path to load checkpoint from
        :return:
        """
        state = torch.load(param_path, weights_only=False)
        model: BaseModel = hydra.utils.instantiate(dict(_target_=state["__target__"], **state["__init_kwargs__"]))
        # for backward compatibility we check whether the keyword is present in the state
        if "__peft_kwargs__" in state and len(state["__peft_kwargs__"]) > 0:
            peft_cfg = PeftConfig.from_peft_type(**state["__peft_kwargs__"])
            model.set_peft(peft_cfg)
        model.backbone.load_state_dict(state["backbone"])  # type: ignore[union-attr]
        model._frozen_params = state["__frozen_params__"]
        for head_name, init_kwargs in zip(state["__head_names__"], state["__head_init_kwargs__"]):
            head = model._create_head(**init_kwargs)
            model.heads[head_name] = head
            head.load_state_dict(state[head_name])
        logger.info("Loaded MML checkpoint!")
        logger.debug(f"@ {param_path}")
        return model

    def save_checkpoint(self, param_path: Union[Path, str]) -> None:
        """
        Save a model checkpoint.

        :param Union[Path, str] param_path: path to store checkpoint
        :return:
        """
        state = {name: head.state_dict() for name, head in self.heads.items()}
        state.update(
            {
                "backbone": self.backbone.state_dict(),  # type: ignore[union-attr]
                "__head_names__": list(self.heads.keys()),
                "__init_kwargs__": self._init_kwargs,
                "__target__": self.__class__,
                "__frozen_params__": self._frozen_params,
                "__head_init_kwargs__": self._head_init_kwargs,
                "__peft_kwargs__": self._peft_kwargs,
            }
        )
        torch.save(state, param_path)
        logger.info("Saved checkpoint!")
        logger.debug(f"@ {param_path}")


class BaseHead(torch.nn.Module, ABC):
    def __init__(self, task_type: TaskType, num_classes: int, **kwargs: Any):
        """The base class for MML model heads."""
        super(BaseHead, self).__init__()
        self.task_type = task_type
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
