# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

import numpy as np
import optuna
import torch
from omegaconf import DictConfig
from PIL import Image
from prettytable.colortable import ColorTable, Themes
from torchvision.utils import make_grid
from tqdm import tqdm

from mml.core.data_loading.task_attributes import DataSplit, Modality, TaskType
from mml.core.data_loading.task_dataset import TaskDataset
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import LearningPhase, throttle_logging

logger = logging.getLogger(__name__)


class InfoScheduler(AbstractBaseScheduler):
    """
    AbstractBaseScheduler implementation for receiving information on a project. Includes the following subroutines:
    - tasks  (show information on tasks)
    - hpo  (show information on hpo results)
    - samples  (plots sample images for each task (and/or jointly))
    - models (show information on existing model descriptions)
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(InfoScheduler, self).__init__(cfg=cfg, available_subroutines=["tasks", "hpo", "samples", "models"])
        self.total_sample_sum = 0

    def create_routine(self):
        """
        This scheduler implements two subroutines, one for task information and one for hpo study information.

        :return: None
        """
        # -- add task info commands
        if "tasks" in self.subroutines:
            for task in self.cfg.task_list:
                self.commands.append(self.info_task)
                self.params.append([task])
        # -- add study info commands
        if "hpo" in self.subroutines:
            if self.cfg.mode.study_name:
                # was given a single study
                self.commands.append(self.info_study)
                self.params.append([self.cfg.mode.study_name])
            else:
                # show all available studies for this project
                logger.info("Was given no study name to search for, so showing all studies with project prefix.")
                study_summaries = optuna.study.get_all_study_summaries(storage=self.cfg.hpo.storage)
                logger.debug(f"Found {len(study_summaries)} studies in database.")
                filtered_studies = [
                    study.study_name for study in study_summaries if str(study.study_name).startswith(self.cfg.proj)
                ]
                for study_name in filtered_studies:
                    self.commands.append(self.info_study)
                    self.params.append([study_name])
        # -- add sample plotting command
        if "samples" in self.subroutines:
            self.commands.append(self.plot_samples)
            self.params.append([])
        # -- add MODEL info command
        if "models" in self.subroutines:
            self.commands.append(self.info_models)
            self.params.append([])

    def after_preparation_hook(self):
        pass

    def before_finishing_hook(self):
        logger.info(f"Total number of all samples: {self.total_sample_sum}.")

    def prepare_exp(self) -> None:
        # no need to prepare tasks if only looking at hpo results
        if "tasks" in self.subroutines or "sample_grid" in self.subroutines or "models" in self.subroutines:
            super(InfoScheduler, self).prepare_exp()

    def info_task(self, task_name: str) -> None:
        logger.info("Starting info on task " + self.highlight_text(task_name))
        task_struct = self.get_struct(task_name)
        logger.info(str(task_struct))
        dataset = TaskDataset(
            root=Path(self.cfg.data_dir) / task_struct.relative_root, split=DataSplit.FULL_TRAIN, fold=0, transform=None
        )
        logger.info(f"Num samples (full train set): {len(dataset)}")
        self.total_sample_sum += len(dataset)
        if dataset.task_type == TaskType.CLASSIFICATION:
            dataset.select_samples(split=DataSplit.VAL, fold=0)
            class_occ = {}
            for sample in dataset.samples:
                _cl = sample[Modality.CLASS]
                if _cl in class_occ:
                    class_occ[_cl] += 1
                else:
                    class_occ[_cl] = 1
            logger.info(f"Default validation class occurrences are: {class_occ}")
        logger.info("Finished info on task " + self.highlight_text(task_name))

    def info_study(self, study_name: str) -> None:
        logger.info("Starting info on study " + self.highlight_text(study_name))
        try:
            loaded_study = optuna.load_study(study_name=study_name, storage=self.cfg.hpo.storage)
        except KeyError:
            raise MMLMisconfigurationException(
                f"Study {study_name} not found! Adapt mode.study_name config, "
                f"and check your hpo.storage settings to make studies persistent. "
                f"See  https://docs.sqlalchemy.org/en/20/core/engines.html for details."
                f"Create the study from scratch with mml ... "
                f"hydra.sweeper.study_name=YOUR_STUDY_NAME "
                f"hpo.storage=YOUR_STORAGE_SETTING --multirun."
            )
        df = loaded_study.trials_dataframe()
        logger.info(f"Detailed study info:\n{df.drop(columns='number').to_string()}")
        logger.info(f"Statistics of study:\n{df.drop(columns=['number', 'duration']).describe().T.to_string()}")
        top = (
            df.drop(
                columns=[
                    "number",
                    "duration",
                    "datetime_start",
                    "datetime_complete",
                    "state",
                    "system_attrs_search_space",
                ]
            )
            .sort_values(by="value", ascending=self.cfg.hpo.direction == "minimize", ignore_index=True)
            .head(5)
            .T.to_string()
        )
        logger.info(f"Top entries:\n{top}")
        if len(loaded_study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) > 0:
            best_params = loaded_study.best_params
            best_val = loaded_study.best_value
            logger.info(f"Best params: {best_params}")
            logger.info(f"Best value: {best_val}")
        else:
            logger.info("No complete trial corresponding to this study has been found.")
        logger.info("Finished info on study " + self.highlight_text(study_name))

    def plot_samples(self) -> None:
        logger.info("Starting plotting sample grid of all tasks.")
        if len(self.cfg.task_list) == 0:
            logger.error("No tasks selected to show samples from.")
            return
        img_samples = {}
        # sort by sample size
        task_sizes = {struct.name: sum(struct.class_occ.values()) for struct in self.task_factory.container}
        task_list = sorted(task_sizes.items(), key=lambda x: x[1])
        for task_name in tqdm([t[0] for t in task_list], desc="Loading samples"):
            task_struct = self.get_struct(task_name=task_name)
            datamodule = self.create_datamodule(task_structs=task_struct)
            # suppress warning of non-normalized image usage
            with throttle_logging(level=logging.WARNING, package="mml.core.data_loading.lightning_datamodule"):
                datamodule.setup(stage="fit")
            img_samples[task_name] = datamodule.task_datasets[task_name][LearningPhase.TRAIN][-1][Modality.IMAGE] * 255
        if self.cfg.mode.info_grid:  # gridplot
            proportion = 11 / 5  # width to height proportion
            grid = make_grid(tensor=list(img_samples.values()), nrow=int(np.sqrt(len(img_samples) * proportion)))
            path = self.fm.construct_saving_path(obj=grid, key="sample_grid")
            ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(path)
            logger.info(f"Finished plotting sample grid. Can be found at {path}.")
        if self.cfg.mode.info_individual:  # individual samples plots
            for task_name, img_tensor in img_samples.items():
                path = self.fm.construct_saving_path(obj=img_tensor, key="img_examples", task_name=task_name)
                ndarr = img_tensor.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(path)
            logger.info("Finished plotting individual samples for each task.")

    def info_models(self):
        if len(self.cfg.task_list) == 0:
            logger.error("No tasks selected to show models from.")
            return
        logger.info(
            "Model info shows only loaded models (not all existing!). Use reuse.models=... to select a "
            "project to load models from."
        )
        table = ColorTable(theme=Themes.OCEAN)
        table.field_names = ["task", "created", "fold", "performance", "training (secs)", "params?", "preds?"]
        no_model_tasks = []
        for task in self.cfg.task_list:
            task_models = self.get_struct(task).models
            if len(task_models) == 0:
                no_model_tasks.append(task)
                continue
            for ix, model in enumerate(task_models):
                table.add_row(
                    [
                        model.task,
                        model.created,
                        model.fold,
                        f"{model.performance:.8f}" if model.performance else "x",
                        int(model.training_time),
                        "y" if model.parameters and model.parameters.exists() else "n",
                        len(model.predictions),
                    ],
                    divider=ix + 1 == len(task_models),
                )
        print(table)
        logger.info(f"No models found for {len(no_model_tasks)} tasks ({no_model_tasks}).")
