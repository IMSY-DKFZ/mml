# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging

import torch
import torch.utils.data
from mml_similarity.scripts.abstract_task_distance_scheduler import AbstractTaskDistanceScheduler
from mml_similarity.scripts.fisher import compute_fisher_information_embedding, embedding_to_tensor
from mml_similarity.scripts.vector_distances import compute_task_distance
from omegaconf import DictConfig

from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.utils import LearningPhase

logger = logging.getLogger(__name__)


class FEDScheduler(AbstractTaskDistanceScheduler):
    """
    AbstractBaseScheduler implementation for the FED setup. Includes the following subroutines:
    - initial transfer phase for probe model specification
    - fisher information matrix (diagonal) computation
    - distance computation
    - (plotting, not as full subroutine, but as part of the finish experiment routine)
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(FEDScheduler, self).__init__(cfg=cfg, available_subroutines=["tune", "fim", "distance"])
        if len(self.cfg.augmentations.gpu) != 0:
            raise MMLMisconfigurationException(
                f"Distance computations for {self.distance_measure} do not support GPU" f" augmentations!"
            )

    def create_routine(self) -> None:
        """
        This scheduler implements 3 subroutines (transfer, fim, distance).

        :return: None
        """
        # -- add transfer learning commands
        if "tune" in self.subroutines:
            for task in self.cfg.task_list:
                self.commands.append(self.train_probe_model)
                self.params.append([task])
        # -- add fim computation commands
        if "fim" in self.subroutines:
            for task in self.cfg.task_list:
                self.commands.append(self.compute_fim)
                self.params.append([task])
        # -- add distance computation commands
        self.create_default_distances_routine()

    # first individual command, routine for tuning a probe model
    def train_probe_model(self, task_name: str) -> None:
        """
        Computes a probe model to compute the fisher embedding. Trains the final classifier of a pretrained model.
        If "fc_tuned" is already present in the task struct (by setting reuse.fc_tuned=PROJECT) for a project having
        already computed the probe model this step is skipped and the present probe model is used for following
        computations.

        :param str task_name: name of the task to train the probe model
        :return: None, will update task_struct.paths["fc_tuned"]
        """
        task_struct = self.get_struct(task_name)
        if "fc_tuned" in task_struct.paths:
            logger.info("Found existing probe model for task " + self.highlight_text(task_name))
            return
        logger.info("Starting tuning for task " + self.highlight_text(task_name))
        data_module = self.create_datamodule(task_struct)
        lit_model = self.create_model([task_struct])
        lit_model.model.freeze_backbone()
        trainer = self.create_trainer(metrics_callback=False)
        logger.debug("Tuning trainer...")
        self.lightning_tune(trainer=trainer, model=lit_model, datamodule=data_module)
        logger.debug("Fitting trainer...")
        trainer.fit(model=lit_model, datamodule=data_module)
        logger.debug("Saving model...")
        path = self.fm.construct_saving_path(lit_model.model, key="fc_tuned", task_name=task_struct.name)
        lit_model.model.save_checkpoint(param_path=path)
        task_struct.paths["fc_tuned"] = path
        logger.info("Successfully finished tuning for task " + self.highlight_text(task_name))

    # second individual command, routine for computing the fisher information matrix (or subpart of it)
    def compute_fim(self, task_name: str) -> None:
        """
        Computes the fisher embedding based on a probe model. Requires "fc_tuned" to be present in the task struct.
        If "fim" is already present in the task struct (by setting reuse.fim=PROJECT) for a project having
        already computed the fisher embedding this step is skipped and the present fim is used for following
        computations.

        :param str task_name: name of the task to compute the fim
        :return: None, will update task_struct.paths["fim"]
        """
        task_struct = self.get_struct(task_name)
        if "fim" in task_struct.paths:
            logger.info("Found existing fim for task " + self.highlight_text(task_name))
            return
        logger.info("Starting fim computation for task " + self.highlight_text(task_name))
        if "fc_tuned" not in task_struct.paths:
            raise RuntimeError(
                f"Could not find probe model path in task_struct of task {task_name}. Please ensure to run the tune "
                f"routine beforehand or set reuse accordingly."
            )
        # loading model, need only torch.nn.Module not lightning module -> can drop this information
        logger.debug("Loading model...")
        path = task_struct.paths["fc_tuned"]
        model = self.create_model([task_struct]).model
        model.load_checkpoint(param_path=path)
        logger.debug("Preparing data...")
        data_module = self.create_datamodule(task_structs=task_struct)
        data_module.setup(stage="fit")
        loader_kwargs = data_module.get_loader_kwargs_from_cfg(phase=LearningPhase.TRAIN, task_name=task_name)
        device = torch.device("cuda" if self.cfg.allow_gpu else "cpu")
        embedding = compute_fisher_information_embedding(
            fim_cfg=self.cfg.distance.fim,
            model=model,
            ds=data_module.task_datasets[task_name][LearningPhase.TRAIN],
            loader_kwargs=loader_kwargs,
            device=device,
            task_type=task_struct.task_type,
        )
        embedding_path = self.fm.construct_saving_path(embedding, key="fim", task_name=task_struct.name)
        torch.save(embedding, embedding_path)
        task_struct.paths["fim"] = embedding_path
        logger.info("Successfully finished fim computation for task " + self.highlight_text(task_name))

    # third individual command, routine for computing distances from the fisher information embedding
    def compute_distance_impl(self, source_task_name: str, target_task_name: str) -> float:
        """
        Computes the fisher embedding distance between two tasks. Requires "fim" to be present in both task structs. If
        the fims do not match (e.g. they are based on different probe models) an error will be raised.

        :param str source_task_name: source task name
        :param str target_task_name: target task name
        :return: the fisher embedding distance between source and target task
        """
        # load datasets
        source_struct = self.get_struct(source_task_name)
        target_struct = self.get_struct(target_task_name)
        if any("fim" not in struct.paths for struct in [source_struct, target_struct]):
            raise RuntimeError(
                "Could not find fim path in task_struct of some task. Please ensure to run the fim "
                "routine beforehand or set reuse accordingly."
            )
        # load saved embeddings
        source_embedding = torch.load(source_struct.paths["fim"], weights_only=False)
        target_embedding = torch.load(target_struct.paths["fim"], weights_only=False)
        # filter embeddings for prefix
        source_embedding = {k: v for k, v in source_embedding.items() if k.startswith(self.cfg.distance.prefix)}
        target_embedding = {k: v for k, v in target_embedding.items() if k.startswith(self.cfg.distance.prefix)}
        # assure compatibility
        assert set(source_embedding.keys()) == set(
            target_embedding.keys()
        ), f"Fisher embeddings of task {source_task_name} and task {target_task_name} are not matching!"
        # computing the distance
        distance = compute_task_distance(
            embedding_to_tensor(source_embedding),
            embedding_to_tensor(target_embedding),
            metric=self.cfg.distance.metric,
        )
        return distance

    def is_sym_distance(self) -> bool:
        """
        Helper function used to identify if distances have to be calculated in both directions. For FEDScheduler,
        this is depending on the distance metric applied on the embeddings.

        :return: (bool) True is indicating symmetric behaviour
        """
        return {
            "euclidean": True,
            "cosine": True,
            "normalized_euclidean": True,
            "jensenshannon": True,
        }[self.cfg.distance.metric]
