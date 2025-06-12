# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import random
import warnings
from typing import List, Optional

import lightning
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

from mml.core.data_loading.task_attributes import DataSplit
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.model_storage import ModelStorage
from mml.core.scripts.pipeline_configuration import PipelineCfg
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import ARG_SEP, TAG_SEP, LearningPhase, catch_time

logger = logging.getLogger(__name__)


class TrainingScheduler(AbstractBaseScheduler):
    """
    New version of the former "optimization" scheduler. Supports the following features:
    - model training
    - model prediction
    - model testing

    In addition to the standard hooks (after_preparation_hook, before_finishing_hook) it provides additional hooks that
    may be overridden by inheriting schedulers:
    - before_training_hook
    - after_training_hook

    It further allows for task nesting and cross validation.
    """

    def __init__(self, cfg: DictConfig):
        # check compliance to new train scheduler behaviour
        if not cfg.pivot.name:
            raise MMLMisconfigurationException(
                "Train mode (and all inherited ones) requires a pivot task to be set from mml version 0.12.0 onwards."
            )
        if ("nested" in cfg.pivot.name or "nested" in cfg.pivot.tags) and cfg.mode.nested:
            warnings.warn(
                "TrainingScheduler takes care of task nesting itself. Currently you are introducing DOUBLE "
                "nesting by setting mode.nested=true and choosing an already nested pivot task."
            )
        # initialize
        self.n_folds: int = -1  # will be set during create_routine
        super(TrainingScheduler, self).__init__(cfg=cfg, available_subroutines=["train", "predict", "test"])
        self.monitored_performances: List[float] = []  # final loss values observed during validation
        # interpretation and checking for multitask options
        self.co_tasks: List[str] = []
        if self.cfg.mode.multitask:
            if self.cfg.mode.multitask == 1:
                raise MMLMisconfigurationException(
                    "To enable multitask learning set mode.multitask to the TOTAL number of task to be learned jointly."
                )
            if self.cfg.mode.co_tasks == "random":
                choices = [task for task in self.cfg.task_list if not task.startswith(self.pivot)]
                if len(choices) < self.cfg.mode.multitask - 1:
                    raise MMLMisconfigurationException("Available tasks are not enough to support sufficient co-tasks.")
                self.co_tasks = random.sample(population=choices, k=self.cfg.mode.multitask - 1)
                logger.info(f"Randomly selected co-learning tasks: {self.co_tasks}")
            else:
                # was given explicit co tasks, check if compatible
                if any(task not in self.cfg.task_list for task in self.cfg.mode.co_tasks):
                    raise MMLMisconfigurationException("Co task not present in cfg.task_list!")
                if len(set(self.cfg.mode.co_tasks)) != len(self.cfg.mode.co_tasks):
                    raise MMLMisconfigurationException(
                        "mode.co_tasks supports identical co-tasks only if "
                        "you use the identity tag (task_name+identity) to "
                        "create a virtual duplicate."
                    )
                if any(task.startswith(self.pivot) for task in self.cfg.mode.co_tasks):
                    raise MMLMisconfigurationException("Should not use pivot (or derivative) as co-task.")
                self.co_tasks = self.cfg.mode.co_tasks
                logger.info(f"Configured co-learning tasks: {self.co_tasks}")
        # these are some information leakage backup checks
        if not self.cfg.mode.nested and self.cfg.mode.cv:
            # check lr scheduler leakage
            if (
                self.cfg.lr_scheduler["_target_"] is not None
                and "ReduceLROnPlateau" in self.cfg.lr_scheduler["_target_"]
            ):
                raise MMLMisconfigurationException(
                    "Using ReduceLROnPlateau LR-scheduler without activating mode.nested=true leads to "
                    "information leakage from val split on training and should be avoided. Either activate"
                    " nesting or change the LR-scheduler, e.g. with <lr_scheduler=none>."
                )
            # check early stopping leakage
            for cb in self.cfg.cbs.values():
                if "EarlyStopping" in cb["_target_"] and "val" in cb["monitor"]:
                    raise MMLMisconfigurationException(
                        "Using EarlyStopping callback without activating mode.nested=true leads to"
                        "information leakage from val split on training and should be avoided. Either deactivate this "
                        "callback (e.g. <callbacks=none>), choose a different monitor value (not depending on the val "
                        "split) or activate nesting."
                    )
        # checks regarding the storing of model parameters
        if not self.cfg.mode.store_parameters and "train" in self.subroutines and "predict" in self.subroutines:
            raise MMLMisconfigurationException("Predictions after training require mode.store_parameters=True!")

        if (
            self.cfg.mode.store_parameters
            and "check_val_every_n_epoch" in self.cfg.trainer
            and self.cfg.trainer.check_val_every_n_epoch > self.cfg.trainer.max_epochs
        ):
            raise MMLMisconfigurationException(
                f"It seems like you only validate every {self.cfg.trainer.check_val_every_n_epoch} epochs, but "
                f'only train for max {self.cfg.trainer.max_epochs} although requested "mode.store_parameters".'
            )

        if self.cfg.mode.store_parameters and self.cfg.mode.cv:
            warnings.warn(
                f"Cross-Validation will store {self.n_folds} model parameters. To reduce memory consumption "
                f"you may consider either setting mode.store_parameters=false (which will omit storing the "
                f"model parameters) or reuse.clean_up.parameters=true (which deletes the model parameters "
                f"at the end of the experiment."
            )
        # more checks
        if self.cfg.mode.cv and "test" in self.subroutines:
            warnings.warn(
                "Chose both cross validation and testing (on hold out test set). Note that only one CV model "
                "will be evaluated!"
            )
        if not self.cfg.mode.nested and "test" in self.subroutines:
            warnings.warn(
                "You are testing on the `actual` test set! To ensure unbiased fair evaluation this should only "
                "be done on the very end of model development."
                "You may chose mode.nested=true so the testing subroutine will be performed NOT on the (potential) "
                "official task test split, but on the hold-out fold."
            )
        if self.cfg.mode.eval_on:
            if not (isinstance(self.cfg.mode.eval_on, list) or isinstance(self.cfg.mode.eval_on, ListConfig)):
                raise MMLMisconfigurationException("Must provide mode.eval_on as list of tasks, gave.")
            if any(t not in self.cfg.task_list for t in self.cfg.mode.eval_on):
                raise MMLMisconfigurationException(
                    f"Chose to evaluate on {self.cfg.mode.eval_on} but one of these tasks is not given in the task_list"
                )
        if (
            self.cfg.reuse.models
            and "train" in self.subroutines
            and any(sub in self.subroutines for sub in ["test", "predict"])
        ):
            raise MMLMisconfigurationException(
                "Reusing existing models combined with training. This may lead to undetermined behaviour during "
                "testing/predicting"
            )

    def create_routine(self):
        """
        This scheduler implements three sub-routines, training, testing and prediction.
        The routine takes care of cross validation and nesting.
        """
        # calculate the number of available pivot folds
        try:
            pivot_description = self.fm.load_task_description(
                self.fm.data_path / self.fm.task_index[self.pivot]["none"]
            )
        except KeyError:
            raise RuntimeError(
                f"Task {self.pivot} not found in task index. You may need to call info or create mode before passing "
                f"as pivot task to train."
            )
        self.n_folds = len(pivot_description.train_folds)
        # derive folds to loop over
        if self.cfg.mode.cv:
            folds = list(range(self.n_folds))
        else:
            folds = [0]
        # adapt task list, depending on the nesting and cv mode configurations, happens before prepare_exp struct
        # construction, but after the modifications of tasks and pivot by the auto tagging tasks options
        if self.cfg.mode.nested:
            for fold in folds:
                self.cfg.task_list.append(f"{self.pivot}{TAG_SEP}nested{ARG_SEP}{fold}")

        eval_tasks = self.cfg.mode.eval_on or [None]  # None means predict on self

        # -- add training commands
        if "train" in self.subroutines:
            for fold in folds:
                self.commands.append(self.train_fold)
                if self.cfg.mode.nested:
                    self.params.append([f"{self.pivot}{TAG_SEP}nested{ARG_SEP}{fold}", 0])
                else:
                    self.params.append([self.pivot, fold])
        if "predict" in self.subroutines:
            # predicts on the test split (which is original val split for nested tasks)
            for fold in folds:
                for eval_on in eval_tasks:
                    self.commands.append(self.predict_fold)
                    if self.cfg.mode.nested:
                        self.params.append([f"{self.pivot}{TAG_SEP}nested{ARG_SEP}{fold}", 0, eval_on])
                        # also add predictions on test set of original pivot (not nested) - to be used in postprocessing
                        if eval_on is None:
                            self.commands.append(self.predict_fold)
                            self.params.append([f"{self.pivot}{TAG_SEP}nested{ARG_SEP}{fold}", 0, self.pivot])
                    else:
                        self.params.append([self.pivot, fold, eval_on])
        if "test" in self.subroutines:
            for eval_on in eval_tasks:
                self.commands.append(self.test_task)
                if self.cfg.mode.nested:
                    self.params.append([f"{self.pivot}{TAG_SEP}nested{ARG_SEP}0", eval_on])
                else:
                    self.params.append([self.pivot, eval_on])

    def after_preparation_hook(self):
        if self.cfg.mode.eval_on:
            # compare pivot and eval tasks for compatibility
            for eval_task in self.cfg.mode.eval_on:
                pivot_struct = self.get_struct(self.pivot)
                eval_struct = self.get_struct(eval_task)
                if pivot_struct.task_type != eval_struct.task_type:
                    raise MMLMisconfigurationException(
                        f"Invalid task type for evaluation! Pivot task has type "
                        f"{pivot_struct.task_type} but evaluation task {eval_task} has type "
                        f"{eval_struct.task_type}."
                    )
                if pivot_struct.num_classes != eval_struct.num_classes:
                    raise MMLMisconfigurationException(
                        f"Invalid number of classes for evaluation! Pivot task has "
                        f"{pivot_struct.num_classes} classes but evaluation {eval_task} task has "
                        f"{eval_struct.num_classes} classes."
                    )

    def before_finishing_hook(self):
        # return the task loss (averaged over folds if mode.cv is active)
        self.return_value = np.mean(self.monitored_performances)
        # gather further metrics on training
        with open(self.planned_schedule, "r") as schedule_file:
            planned_schedule = schedule_file.readlines()
        train_runs = [line for line in planned_schedule if self.train_fold.__name__ in line]
        # if neither predict nor test are applied we want to show validation results
        if len(train_runs) > 0 and "test" not in self.subroutines and "predict" not in self.subroutines:
            args = []
            for line in train_runs:
                # store args as tuples task_name, fold in a list for all calls
                args.append(elem.strip(" '") for elem in line.split("/")[-1].strip(" []\n").split(","))
            # for each of the trained models we will evaluate the validation if test
            aggregated_metrics = {}
            logger.info(f"Will try to aggregate validation results over {len(args)} training runs.")
            for task_name, fold in args:
                struct = self.get_struct(task_name)
                model_candidates = [model for model in struct.models if model.fold == int(fold)]
                if len(model_candidates) != 1:
                    # this can happen if reuse was used beforehand
                    logger.error(
                        f"Ambiguous model choices for {task_name} and {fold}! Will skip while aggregating results."
                    )
                    continue
                model = model_candidates[0]
                # check if recorded metrics have validation entry
                val_idxs = [
                    idx
                    for idx, metric_dict in enumerate(model.metrics)
                    if any(k.startswith(LearningPhase.VAL) for k in metric_dict)
                ]
                if len(val_idxs) == 0:
                    logger.error(
                        f"No validation metrics for task {task_name} and fold {fold}! Will skip while "
                        f"aggregating results."
                    )
                    continue
                for metric_name, metric_value in model.metrics[val_idxs[-1]].items():
                    if metric_name in aggregated_metrics:
                        aggregated_metrics[metric_name].append(metric_value)
                    else:
                        aggregated_metrics[metric_name] = [metric_value]
            # compute stats and show
            if len(aggregated_metrics) == 0:
                logger.error("No validation metrics found!")
            else:
                logger.info("Aggregated validation results over training:")
                for metric, values in aggregated_metrics.items():
                    logger.info(f"{metric} : {np.mean(values):.2f} ± {np.std(values):.2f}")

    def before_training_hook(
        self,
        datamodule: lightning.LightningDataModule,
        model: lightning.LightningModule,
        trainer: lightning.Trainer,
        fold: int,
        task_name: str,
    ) -> None:
        """
        This hook allows of setup modification before the model fitting starts (and also before lightning tuning).
        Allows to modify task_weights, data, trainer callbacks, etc. May be overwritten as part of inheriting from
        TrainScheduler.

        :param lightning.LightningDataModule datamodule: the prepared datamodule (no setup run yet)
        :param lightning.LightningModule model: the prepared model
        :param lightning.Trainer trainer: the prepared trainer
        :param int fold: the current fold
        :param str task_name: the current task
        :return: None
        """
        pass

    def after_training_hook(
        self,
        datamodule: lightning.LightningDataModule,
        model: lightning.LightningModule,
        trainer: lightning.Trainer,
        fold: int,
        task_name: str,
    ) -> None:
        """
        This hook allows of setup modification after the model fitting ended (and potential lightning tuning).
        Allows to modify task_weights, data, trainer callbacks, etc. May be overwritten as part of inheriting from
        TrainScheduler.

        :param lightning.LightningDataModule datamodule: the datamodule used
        :param lightning.LightningModule model: the trained model
        :param lightning.Trainer trainer: the used trainer
        :param int fold: the used fold
        :param str task_name: the pivot task
        :return: None
        """
        pass

    def train_fold(self, task_name: str, fold: int) -> None:
        logger.info("Starting training for task " + self.highlight_text(task_name) + f" and fold {fold}.")
        pivot_struct = self.get_struct(task_name)
        co_structs = [self.get_struct(task_name=task_name) for task_name in self.co_tasks]
        if self.cfg.mode.use_blueprint:
            if "blueprint" in pivot_struct.paths:
                pipeline = PipelineCfg.load(
                    path=pivot_struct.paths["blueprint"], pipeline_keys=self.cfg.mode.pipeline_keys
                )
                logger.info(f"Found blueprint pipeline for task {task_name}, will evaluate that.")
            else:
                raise RuntimeError(f"Was not able to find appropriate blueprint pipeline for task {task_name}!")
        else:
            pipeline = PipelineCfg.from_cfg(current_cfg=self.cfg, restrict_keys=self.cfg.mode.pipeline_keys)
        with pipeline.activate(current_cfg=self.cfg):
            # preparation
            datamodule = self.create_datamodule(task_structs=[pivot_struct] + co_structs, fold=fold)
            module = self.create_model(
                task_structs=[pivot_struct] + co_structs, task_weights=self.cfg.mode.task_weights
            )
            module.train()  # see https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.0
            trainer = self.create_trainer(
                monitor=(f"val/{task_name}/loss", "min") if self.cfg.mode.store_best else None, metrics_callback=True
            )
            self.before_training_hook(
                datamodule=datamodule, model=module, trainer=trainer, fold=fold, task_name=task_name
            )
            # tuning and training
            with catch_time() as training_timer:
                self.lightning_tune(trainer=trainer, model=module, datamodule=datamodule)
                trainer.fit(model=module, datamodule=datamodule)
            self.after_training_hook(
                datamodule=datamodule, model=module, trainer=trainer, fold=fold, task_name=task_name
            )
            # create another pipeline from the current one (within blueprint keys activated and without restrictions)
            # to ensures storing the full superset of configuration from a potential partially masked blueprint training
            pipeline_path = PipelineCfg.from_cfg(current_cfg=self.cfg).store(
                task_struct=pivot_struct, as_blueprint=False
            )
        # output processing
        if self.cfg.mode.store_best:
            if self.checkpoint_callback.best_model_score is None:
                best_score = 1000  # catch fast_dev_run
            else:
                best_score = self.checkpoint_callback.best_model_score.item()
        else:
            try:
                best_score = self.metrics_callback.metrics[-2][f"val/{task_name}/loss"]
            except KeyError:
                raise RuntimeError(
                    'Unable to find "val/{task_name}/loss" in recorded metrics of the last epoch,'
                    "make sure to activate validation with lightning trainer."
                )
        self.monitored_performances.append(best_score)
        parameters_path = self.fm.construct_saving_path(module, key="parameters", task_name=pivot_struct.name)
        if self.cfg.mode.store_parameters:
            # determine the correct parameters directory
            cpt_path = (
                self.checkpoint_callback.best_model_path
                if self.cfg.mode.store_best
                else self.checkpoint_callback.last_model_path
            )
            # load these weights
            state_dict = torch.load(cpt_path, map_location=torch.device("cpu"), weights_only=False)["state_dict"]
            module.load_state_dict(state_dict)
            # store model
            module.model.save_checkpoint(param_path=parameters_path)
        else:
            logger.info("mode.store_parameters is set false, so no parameters will be stored!")
        storage = ModelStorage(
            pipeline=pipeline_path,
            parameters=parameters_path,
            fold=fold,
            task=task_name,
            performance=best_score,
            metrics=self.metrics_callback.metrics,
            training_time=training_timer.elapsed,
        )
        storage.store(task_struct=pivot_struct, fold=fold)
        pivot_struct.models.append(storage)
        logger.info("Finished training for task " + self.highlight_text(task_name) + f" and fold {fold}.")

    def predict_fold(self, task_name: str, fold: int, eval_on: Optional[str] = None) -> None:
        logger.info("Starting predicting for task " + self.highlight_text(task_name) + f" and fold {fold}.")
        task_struct = self.get_struct(task_name)
        # find model storage
        choices = [storage for storage in task_struct.models if storage.fold == fold]
        if len(choices) == 0:
            raise RuntimeError(f"Did not find any existing model storage for task {task_name} and fold {fold}.")
        # sort ascending
        choices.sort(key=lambda x: x.created)
        storage = choices[-1]
        logger.info(f"Found {len(choices)} matching model storages, used the latest from {storage.created}.")
        # check preprossing compatibility
        original_pipeline = PipelineCfg.load(path=storage.pipeline)
        if original_pipeline.pipeline_cfg.preprocessing.id != self.cfg.preprocessing.id:
            warnings.warn(
                f"Current preprocessing is {self.cfg.preprocessing.id} but the loaded model was originally "
                f"preprocessed as {original_pipeline.pipeline_cfg.preprocessing.id}. MML will try to continue "
                f"with the given preprocessing pipeline."
            )
        # prepare model
        module = self.create_model(task_structs=[task_struct], load_parameters=storage.parameters)
        if eval_on and eval_on != task_name:
            eval_task = self.get_struct(eval_on)
            eval_task_name = eval_on
            logger.info("Will predict on task " + self.highlight_text(eval_task_name) + "!")
            module.setup_redirection(head=task_name, task=eval_task_name)
        else:
            eval_task_name = task_name
            eval_task = task_struct
        # prepare data and trainer
        datamodule = self.create_datamodule(task_structs=eval_task, fold=fold)
        trainer = self.create_trainer()
        # perform predictions
        split_batched_predictions = {}
        with catch_time() as predict_timer:
            for split in [DataSplit.TEST, DataSplit.VAL, DataSplit.UNLABELLED]:
                # switch prediction split
                logger.info(f"Predicting split {split.name}.")
                datamodule.predict_on = split
                split_batched_predictions[split] = trainer.predict(
                    model=module, dataloaders=datamodule, return_predictions=True
                )
                if split_batched_predictions[split] is not None:
                    logger.info(f"Predicted {len(split_batched_predictions[split])} batches for split {split.name}.")
        logger.debug(f"Prediction time was {predict_timer.pretty_time}.")
        # reformat predictions as dict -> image_id : prediction for each data split and combine them
        split_unbatched_predictions = {}
        for data_split, pred_dict_list in split_batched_predictions.items():
            split_unbatched_predictions[data_split.name] = []
            if pred_dict_list is None:
                warnings.warn(f"No predictions found for {data_split}!")
                continue
            for batch in pred_dict_list:
                for sample_idx in range(batch[eval_task_name]["logits"].size(0)):
                    predict_dict = {"logits": batch[eval_task_name]["logits"][sample_idx]}
                    if batch[eval_task_name]["targets"] is not None:
                        predict_dict.update({"target": batch[eval_task_name]["targets"][sample_idx]})
                    if batch[eval_task_name]["sample_ids"] is not None:
                        predict_dict.update({"sample_id": batch[eval_task_name]["sample_ids"][sample_idx]})
                    split_unbatched_predictions[data_split.name].append(predict_dict)
        preds_path = self.fm.construct_saving_path(
            split_unbatched_predictions, key="predictions", task_name=eval_task.name, file_name=f"preds-fold-{fold}.pt"
        )
        torch.save(split_unbatched_predictions, preds_path)
        storage.predictions[eval_task_name] = preds_path
        storage.store()
        logger.info("Finished predicting for task " + self.highlight_text(task_name) + f" and fold {fold}.")

    def test_task(self, task_name: str, eval_on: Optional[str] = None) -> None:
        logger.info("Starting testing for task " + self.highlight_text(task_name))
        task_struct = self.get_struct(task_name)
        # find model storage
        if len(task_struct.models) == 0:
            raise RuntimeError(f"Did not find any existing model storage for task {task_name}.")
        # sort ascending
        choices = sorted(task_struct.models, key=lambda x: x.created)
        storage = choices[-1]
        logger.info(f"Found {len(choices)} matching model storages, used the latest from {storage.created}.")
        # check preprossing compatibility
        original_pipeline = PipelineCfg.load(path=storage.pipeline)
        if original_pipeline.pipeline_cfg.preprocessing.id != self.cfg.preprocessing.id:
            warnings.warn(
                f"Current preprocessing is {self.cfg.preprocessing.id} but the loaded model was originally "
                f"preprocessed as {original_pipeline.pipeline_cfg.preprocessing.id}. MML will try to continue "
                f"with the given preprocessing pipeline."
            )
        # prepare model
        module = self.create_model(task_structs=[task_struct], load_parameters=storage.parameters)
        if eval_on and eval_on != task_name:
            eval_task = self.get_struct(eval_on)
            eval_task_name = eval_on
            logger.info("Will test on task" + self.highlight_text(eval_task_name) + "!")
            module.setup_redirection(head=task_name, task=eval_task_name)
        else:
            eval_task = task_struct
        # prepare data and trainer
        datamodule = self.create_datamodule(task_structs=eval_task)
        trainer = self.create_trainer(metrics_callback=True)
        # run the testing
        with catch_time() as test_timer:
            trainer.test(model=module, datamodule=datamodule)
        logger.debug(f"Testing time was {test_timer.pretty_time}.")
        storage.metrics += self.metrics_callback.metrics
        storage.store()
        logger.info(f"Results: {self.metrics_callback.metrics}")
        logger.info("Finished testing for task " + self.highlight_text(task_name))
