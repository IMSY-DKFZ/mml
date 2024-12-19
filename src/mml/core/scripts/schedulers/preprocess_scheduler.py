# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict

import cv2
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from p_tqdm import p_umap

from mml.core.data_loading.augmentations.albumentations import AlbumentationsAugmentationModule
from mml.core.data_loading.modality_loaders import DEFAULT_MODALITY_LOADERS, NonMappingOpenCVMaskLoader
from mml.core.data_loading.task_attributes import EMPTY_MASK_TOKEN, DataSplit, Modality
from mml.core.data_loading.task_dataset import TaskDataset
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import TAG_SEP, multi_threaded_p_tqdm

logger = logging.getLogger(__name__)


class PreprocessScheduler(AbstractBaseScheduler):
    """
    AbstractBaseScheduler implementation for the process of preprocessing data. Includes the following subroutines:
    - preprocess
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(PreprocessScheduler, self).__init__(cfg=cfg, available_subroutines=["preprocess"])
        # store target preprocessing pipeline (will be overwritten later))
        self.pp_id = deepcopy(self.cfg.preprocessing.id)
        self.pp_pipeline = deepcopy(self.cfg.preprocessing.pipeline)

    def create_routine(self):
        """
        This scheduler implements one subroutine, which preprocesses a task's data.

        :return: None
        """
        # -- add preprocess command
        if "preprocess" in self.subroutines:
            if self.pivot:
                logger.info("Preprocess mode with pivot task will only pp this task!")
                self.commands.append(self.preprocess_task)
                self.params.append([self.pivot])
            else:
                for task in self.cfg.task_list:
                    self.commands.append(self.preprocess_task)
                    self.params.append([task])

    def after_preparation_hook(self):
        # delete pp info from config after (!) task structs are created
        self.cfg.preprocessing.id = "none"
        self.cfg.preprocessing.pipeline = {}

    def before_finishing_hook(self):
        pass

    def preprocess_task(self, task_name: str):
        logger.info("Starting preprocessing data for task " + self.highlight_text(task_name))
        task_struct = self.get_struct(task_name)
        # assert preprocessing not already exists
        if task_struct.preprocessed != "none":
            logger.warning(
                f"Task {task_name} is already preprocessed with {task_struct.preprocessed}, will skip "
                f"computations. Delete {self.fm.data_path / task_struct.relative_root} if you want to redo "
                f"the calculations."
            )
            return

        if TAG_SEP in task_name:
            # check if base task has been preprocessed
            base_name = task_name.split(TAG_SEP)[0]
            if self.pp_id not in self.fm.task_index[base_name]:
                msg = (
                    f"Task {task_name} is a tagged task and should not be preprocessed directly! "
                    f"Please preprocess the base task ({base_name}) and rerun preprocessing the tagged version "
                    f"afterwards."
                )
                logger.error(msg)
                return
            logger.info(
                f"{task_name} as a tagged task has already preprocessed base version for preprocessing "
                f"{self.pp_id}. No image preprocessing necessary. To auto create the tagged preprocessing "
                f"simply refer to the tagged task and its preprocessing in any mml call and it will be "
                f"created on the fly."
            )
            return
        # if this is a base task, then the images have to be preprocessed
        # prepare data loading
        warnings.warn("THIS BEHAVIOUR CHANGED: Test data is now also to be preprocessed!")
        pp_transform = AlbumentationsAugmentationModule(
            device="cpu", cfg=self.pp_pipeline, is_first=True, is_last=False, tensorize=False, means=None, stds=None
        )  # need to set is_last=False to prevent float conversion
        loaders = {mod: DEFAULT_MODALITY_LOADERS[mod]() for mod in task_struct.modalities}
        if Modality.MASK in loaders:
            loaders[Modality.MASK] = (NonMappingOpenCVMaskLoader(),)  # prevent mappings
        task_dataset = TaskDataset(
            root=self.fm.data_path / task_struct.relative_root,
            split=DataSplit.FULL_TRAIN,
            transform=None,
            loaders=loaders,
        )
        task_dataset.modalities = {
            k: v for k, v in task_dataset.modalities.items() if k in [Modality.IMAGE, Modality.MASK]
        }  # load at max images and masks
        # prepare storage
        source_base = (self.fm.data_path / task_struct.relative_root).parent
        target_base = self.fm.get_dataset_path(raw_path=source_base, preprocessing=self.pp_id)
        storage_definition_path = self.fm.get_pp_definition(preprocessing=self.pp_id)
        if not storage_definition_path.exists():
            # write pipeline definition to preprocessing folder
            OmegaConf.save(self.pp_pipeline, storage_definition_path, resolve=True)
        # start multithreaded preprocessing
        with multi_threaded_p_tqdm():
            for split in [DataSplit.FULL_TRAIN, DataSplit.TEST, DataSplit.UNLABELLED]:
                logger.info(f"Preprocessing split: {split}")
                task_dataset.select_samples(split=split, fold=0)
                if len(task_dataset) > 0:
                    exist_results = p_umap(
                        partial(preprocess_and_store, ds=task_dataset, target_base=target_base, transform=pp_transform),
                        range(len(task_dataset)),
                        num_cpus=self.cfg.num_workers,
                    )
                    logger.info(f"Existing {split} files found:\n{pd.DataFrame(data=exist_results).sum().to_string()}")
                else:
                    logger.info(f"No samples in split: {split}")
        # adapt task meta info
        task_creator = TaskCreator(dset_path=(self.fm.data_path / task_struct.relative_root).parent)
        task_creator.load_existent(self.fm.data_path / task_struct.relative_root)
        task_creator.dset_path = target_base
        task_creator.infer_stats(device=torch.device("cuda") if self.cfg.allow_gpu else torch.device("cpu"))
        task_creator.protocol(msg=f"Preprocessed with id={self.pp_id}")
        task_creator.push_and_test()
        logger.info("Finished preprocessing the data for task " + self.highlight_text(task_name))


def preprocess_and_store(
    index: int, ds: TaskDataset, target_base: Path, transform: AlbumentationsAugmentationModule
) -> Dict[str, bool]:
    """
    Function to preprocess and store a single data tuple of a dataset.

    :param ds: dataset
    :param target_base: root path to store preprocessed data
    :param index: index of data tuple to be preprocessed
    :param transform: an albumentations transform module to be applied on loaded data
    :return: dict with modality keys and boolean indicating whether a file already existed beforehand
    """
    pp_elements = transform(**ds.load_sample(index))
    exists = {mod: False for mod in pp_elements.keys()}
    path_info = ds.samples[index]
    for mod, pp_item in pp_elements.items():
        if mod in [Modality.CLASS, Modality.SOFT_CLASSES, Modality.CLASSES, Modality.SAMPLE_ID, Modality.TASK]:
            # no preprocessing needed here
            continue
        if mod not in [Modality.IMAGE, Modality.MASK]:
            raise NotImplementedError(f"Preprocessing not supported for modality {mod}.")
        # ignore dummy mask tokens
        if mod == Modality.MASK and path_info[Modality.MASK] == EMPTY_MASK_TOKEN:
            continue
        if not isinstance(path_info[mod], str):
            raise TypeError(f"Only able to process paths, represented as strings, was given {type(path_info[mod])}")
        path = target_base / path_info[mod]
        if path.exists():
            # default is to overwrite existing files
            exists[mod] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        img = pp_elements[mod]
        if len(img.shape) == 3 and img.shape[2] == 3:
            # since loading converts opencv BGR format to RGB we have to undo this here before saving color images
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename=str(path), img=img)
    return exists
