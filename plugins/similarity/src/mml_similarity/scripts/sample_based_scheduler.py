# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging

import numpy as np
import torch
from mml_similarity.scripts.abstract_task_distance_scheduler import AbstractTaskDistanceScheduler
from mml_similarity.scripts.feature_distances import compute_feature_distance, sample_features
from omegaconf import DictConfig
from tqdm import tqdm

from mml.core.scripts.exceptions import MMLMisconfigurationException

logger = logging.getLogger(__name__)
SAMPLE_BASED_MODES = ["mmd", "kld", "emd", "fid"]


class SampleBasedScheduler(AbstractTaskDistanceScheduler):
    """
    AbstractBaseScheduler implementation for the sampling statistics setup. Includes the following subroutines:
    - feature extraction
    - distance computation
    - (plotting, not as full subroutine, but as part of the finish experiment routine)
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(SampleBasedScheduler, self).__init__(cfg=cfg, available_subroutines=["feature", "distance"])
        self.extracted_features = None  # holds the features, used to avoid repeatedly loading
        if len(self.cfg.augmentations.gpu) != 0:
            raise MMLMisconfigurationException(
                f"Distance computations for {self.distance_measure} do not support GPU augmentations!"
            )

    def create_routine(self):
        """
        This scheduler implements 2 subroutines ('feature', 'distance').

        :return: None
        """
        # -- add feature extraction commands
        if "feature" in self.subroutines:
            for task in self.cfg.task_list:
                self.commands.append(self.extract_features)
                self.params.append([task])
        # -- add distance computation commands
        self.create_default_distances_routine()

    # first individual command, routine for feature extraction
    def extract_features(self, task_name: str) -> None:
        """
        Extracts features based on some pretrained encoder model.
        If "features" is already present in the task struct (by setting `reuse.features=PROJECT`) for a project having
        already extracted the features this step is skipped and the present features are used for following
        computations.

        :param str task_name: name of the task to extract features from
        :return: None, will update task_struct.paths["features"]
        """
        task_struct = self.get_struct(task_name)
        if "features" in task_struct.paths:
            logger.info("Found existing features for task " + self.highlight_text(task_name))
            return
        logger.info("Starting feature extraction for dataset " + self.highlight_text(task_name))
        # gather features# feature extraction by pretrained model
        datamodule = self.create_datamodule(task_structs=task_struct)
        datamodule.setup(stage="fit")
        model = self.create_model(task_structs=[task_struct])
        features = sample_features(
            model=model,  # type:ignore
            datamodule=datamodule,
            device=torch.device("cuda") if self.cfg["allow_gpu"] else torch.device("cpu"),
        )
        # save features after extraction
        path = self.fm.construct_saving_path(obj=features, key="features", task_name=task_struct.name)
        np.save(path, features)
        # add reference to dataset
        task_struct.paths["features"] = path
        logger.info("Successfully finished feature extraction for task " + self.highlight_text(task_name))

    # third individual command, routine for computing distances from the reduced information matrix
    def compute_distance_impl(self, source_task_name: str, target_task_name: str) -> float:
        # load extracted features (numpy arrays)
        if self.extracted_features is None:
            self.extracted_features = {}
            with tqdm(total=len(self.cfg.task_list), desc="Loading extracted features") as t:
                for ix, task_name in enumerate(self.cfg.task_list):
                    t.set_postfix(task=task_name)
                    task_struct = self.get_struct(task_name)
                    if "features" not in task_struct.paths or not task_struct.paths["features"].exists():
                        raise FileNotFoundError(
                            f"Task {task_name} does not have a valid feature path! Please ensure "
                            f"to run the feature extraction routine first or set reuse accordingly."
                        )
                    self.extracted_features[task_name] = np.load(task_struct.paths["features"])
                    t.update()
            # check dimensions
            shapes = [v.shape for v in self.extracted_features.values()]
            if any([s != shapes[0] for s in shapes]):
                logger.error(f"Found inconsistent shapes during loading of extracted features: {shapes}.")
            logger.info("Loaded features.")
        # compute distance
        distance = compute_feature_distance(
            features_1=self.extracted_features[source_task_name],
            features_2=self.extracted_features[target_task_name],
            distance_cfg=self.cfg.distance,
            mode_id=self.distance_measure,
        )
        return distance

    def is_sym_distance(self) -> bool:
        return {"mmd": True, "emd": True, "fid": True, "kld": False}[self.distance_measure]
