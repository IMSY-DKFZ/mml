# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import hydra.utils
import lightning
import numpy as np
import torch.nn
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from lightning.pytorch.callbacks import BatchSizeFinder
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.lr_finder import _LRCallback
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.wrappers import BootStrapper

from mml.core.data_loading.augmentations.kornia import KorniaAugmentationModule
from mml.core.data_loading.lightning_datamodule import MultiTaskDataModule
from mml.core.data_loading.task_attributes import Modality, TaskType
from mml.core.data_loading.task_dataset import TaskDataset
from mml.core.data_loading.task_struct import TaskStruct
from mml.core.models.merger import PredictionMerger
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.utils import LearningPhase
from mml.core.visualization.cm import render_confusion_matrix
from mml.core.visualization.predictions import render_predictions
from mml.core.visualization.utils import undo_image_normalization

logger = logging.getLogger(__name__)

# determines the config parts in loss and metrics that correspond to the task types
CONFIGS_ROUTES = {
    TaskType.CLASSIFICATION: "cls",
    TaskType.MULTILABEL_CLASSIFICATION: "mlcls",
    TaskType.SEMANTIC_SEGMENTATION: "seg",
    TaskType.REGRESSION: "reg",
}


class SingleFrameLightningModule(lightning.LightningModule):
    """
    The default MML lightning module supporting frame wise training and inference.
    """

    def __init__(self, task_structs: List[TaskStruct], cfg: DictConfig, weights: Optional[List[float]] = None):
        super(SingleFrameLightningModule, self).__init__()
        # save hyperparameters
        self.save_hyperparameters()
        self.cfg = cfg
        if weights is None:
            weights = [1.0] * len(task_structs)
        if len(task_structs) != len(weights):
            raise ValueError(f"Number of weights ({len(weights)} does not match number of tasks {len(task_structs)}.")
        self.weights = torch.as_tensor(weights)
        self.task_structs = {struct.name: struct for struct in task_structs}
        self.targets: Dict[str, str] = {  # type: ignore
            name: struct.target.value for name, struct in self.task_structs.items() if struct.target is not None
        }
        # construct model
        self.model = hydra.utils.instantiate(self.cfg.arch)
        for struct in self.task_structs.values():
            self.model.add_head(task_struct=struct)
        # construct criterion
        self.criteria = self.get_criteria()
        # construct metrics
        metric_lists = {name: self.get_metrics(struct) for name, struct in self.task_structs.items()}
        metric_collections = {
            name: MetricCollection(metrics).set_dtype(torch.float) for name, metrics in metric_lists.items()
        }
        for metric_collection in metric_collections.values():
            metric_collection.persistent(mode=True)
        self.train_metrics = torch.nn.ModuleDict(
            {
                task_name: metric_collections[task_name].clone(prefix=f"train/{task_name}/")
                for task_name in self.task_structs
            }
        )
        self.val_metrics = torch.nn.ModuleDict(
            {
                task_name: metric_collections[task_name].clone(prefix=f"val/{task_name}/")
                for task_name in self.task_structs
            }
        )
        if self.cfg.metrics.bootstrap:
            # wrap in bootstrapper, needs to proceed with dict to avoid duplicate naming of bootstrapper in collection
            wrapped_metric_lists = {
                name: {met._get_name(): BootStrapper(met, num_bootstraps=self.cfg.metrics.bootstrap) for met in metrics}
                for name, metrics in metric_lists.items()
            }
            wrapped_collections = {name: MetricCollection(metrics) for name, metrics in wrapped_metric_lists.items()}
            for metric_collection in wrapped_collections.values():
                metric_collection.persistent(mode=True)
            self.test_metrics = torch.nn.ModuleDict(
                {
                    task_name: wrapped_collections[task_name].clone(prefix=f"test/{task_name}/")
                    for task_name in self.task_structs
                }
            )
        else:
            self.test_metrics = torch.nn.ModuleDict(
                {
                    task_name: metric_collections[task_name].clone(prefix=f"test/{task_name}/")
                    for task_name in self.task_structs
                }
            )
        # this attribute is used for auto_lr_finder of lightning
        self.lr = None
        # create confusion matrices
        self.train_cms = torch.nn.ModuleDict()
        self.val_cms = torch.nn.ModuleDict()
        self.test_cms = torch.nn.ModuleDict()
        if self.cfg.logging.cm:
            for name, struct in self.task_structs.items():
                if struct.task_type not in [TaskType.SEMANTIC_SEGMENTATION, TaskType.CLASSIFICATION]:
                    warnings.warn(f"{struct.task_type} does not support logging.cm configuration.")
                else:
                    cm = MulticlassConfusionMatrix(num_classes=struct.num_classes, normalize=None)
                    self.train_cms[name] = cm
                    self.val_cms[name] = cm.clone()
                    self.test_cms[name] = cm.clone()
        # mapping of model heads, allows to infer on tasks other than trained for
        self._task_to_head_mapping = {task_name: task_name for task_name in self.task_structs}
        # fix for multi-dataloader loss aggregation
        self.loss_list = []
        # test time augmentation
        if len(self.cfg.tta.variations) > 0:
            self.tta_pipelines = {
                key: KorniaAugmentationModule(
                    device="gpu", cfg=pipeline, is_first=False, is_last=False, means=None, stds=None
                )
                for key, pipeline in self.cfg.tta.variations.items()
            }
        else:
            self.tta_pipelines = {}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Default forward method, this is not used from within pytorch lightning itself. It is provided to the outside
        as inference option.

        :param torch.Tensor x: plain batch or single image (no modality dict!)
        :return: dict with one entry per model head and corresponding prediction logits
        """
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Special forward method generating embeddings for images, this is not used from within pytorch lightning itself.
        It is provided to the outside as embedding generator option.

        :param torch.Tensor x: plain batch or single image (no modality dict!)
        :return: tensor of shape num_samples x num_features # TODO verify
        """
        return self.model.forward_features(x)

    @property
    def is_tuning(self) -> bool:
        """
        Checks if the model is currently being tuned, which allows to modify some operations.
        """
        return bool(
            [cb for cb in self.trainer.callbacks if isinstance(cb, BatchSizeFinder) or isinstance(cb, _LRCallback)]
        )

    def push_and_sort(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        raise_on_error: bool = True,
        perform_tta: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        The lightning internal used "forward" method for dict based dataloaders. It deals with the dict input of the
        combined dataloader in any mode but "sequential" and resolves the modalities as well as tasks.

        :param Dict[str, Dict[str, torch.Tensor]] batch: a batch of format {task_name: {modality_name: tensor_values}}
        :param bool raise_on_error: if False accepts missing targets in the batch (e.g. during test step)
        :param bool perform_tta: if True performs multiple forward passes with augmented batch variants and merges them
        :return: a tuple consisting of logits dict and targets dict which with keys for each task
        :rtype: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        """
        if perform_tta and len(self.tta_pipelines) > 0:
            # only augment images through TTA
            task_imgs = {
                task: {Modality.IMAGE.value: batch[task][Modality.IMAGE.value]} for task in batch if batch[task]
            }
            logits = {}
            for task in task_imgs:
                merger = PredictionMerger(mode=self.cfg.tta.mode, modality=Modality.from_str(self.targets[task]))
                for pipeline in self.tta_pipelines.values():
                    variation = pipeline(task_imgs[task])
                    predicted_logits = self.model(variation[Modality.IMAGE.value])[self._task_to_head_mapping[task]]
                    if self.targets[task] in [Modality.MASK.value, Modality.BBOX.value, Modality.KEYPOINTS.value]:
                        # if necessary undo geometric variation
                        predicted_logits = pipeline.inverse({self.targets[task]: predicted_logits})
                    merger.update(predicted_logits)
                logits[task] = merger.compute()
        else:
            logits = {
                task: self.model(batch[task][Modality.IMAGE.value])[self._task_to_head_mapping[task]]
                for task in batch
                if batch[task]
            }
        try:
            targets = {task: batch[task][self.targets[task]] for task in logits if task in self.targets}
        except KeyError:
            if raise_on_error:
                raise
            else:
                targets = None
        return logits, targets

    def reformat_batch_from_sequential(
        self, batch: Dict[str, torch.Tensor], dataloader_idx: int
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Prepares the batch format of "sequential" mode combined loader to default format.

        :param Dict[str, torch.Tensor] batch: a batch of format {modality_name: tensor_values}
        :param int dataloader_idx: index of the dataloader
        :return: a batch of format {task_name: {modality_name: tensor_values}}
        :rtype:  Dict[str, Dict[str, torch.Tensor]]
        """
        datamodule: MultiTaskDataModule = self.trainer.datamodule
        task_name = datamodule.task_structs[dataloader_idx].name
        return {task_name: batch}

    @rank_zero_only
    def log_images_prediction_reference(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        phase: LearningPhase,
    ) -> None:
        """
        Logging utility for showing image examples together with reference and model predictions.

        :param Dict[str, Dict[str, torch.Tensor]] batch: batch as provided by dataloader (batch[task][modality])
        :param Dict[str, torch.Tensor] logits: logits as provided by model :meth:step
        :param Dict[str, torch.Tensor] targets: targets as provided by :meth:step
        :param LearningPhase phase: may be either train, val or test, used to access underlying
            :class:~mml.core.data_loading:task_dataset:TaskDataset and as a logging prefix
        :return:
        """
        if self.is_tuning:
            return
        datamodule: MultiTaskDataModule = self.trainer.datamodule
        images = {task: batch[task][Modality.IMAGE.value] for task in batch}
        for task in batch:
            # catch empty task batch (e.g. during validation)
            if batch[task] is None:
                continue
            # reduce plotting number
            n = min(self.cfg.logging.samples, images[task].size(dim=0))
            if n <= 0:
                return
            images = images[task][:n]
            logits = logits[task][:n]
            targets = targets[task][:n]
            # undo image normalization
            mean, std = datamodule.get_image_normalization(struct=self.task_structs[task])
            raw_images = undo_image_normalization(images=images, means=mean, stds=std)
            # render figure
            fig = render_predictions(
                raw_images=raw_images,
                logits=logits,
                targets=targets,
                classes=datamodule.task_datasets[task][phase].classes,
                task_type=self.task_structs[task].task_type,
            )
            # log figure
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    tag=f"{phase}/{task}/prediction", figure=fig, global_step=self.trainer.global_step, close=True
                )
            else:
                logger.error(f"Unable to log prediction examples for {type(self.logger)} logger type.")
                break

    def compute_and_log_loss(
        self, logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], phase: LearningPhase
    ) -> torch.Tensor:
        # generate loss tensor hardware-agnostic
        present_task = next(iter(logits.keys()))
        loss = torch.zeros(1).to(logits[present_task])
        # compute loss across tasks, incorporating task weights
        n_tasks = 0
        for task_name, weight in zip(self.task_structs, self.weights):
            if task_name in targets and targets[task_name].size(dim=0) > 0:
                n_tasks += 1
                task_loss = self.criteria[task_name](logits[task_name], targets[task_name])
                loss += weight * task_loss
                self.log(
                    f"{phase.value}/{task_name}/loss",
                    task_loss,
                    batch_size=logits[task_name].size(0),
                    add_dataloader_idx=False,
                )
        loss /= n_tasks  # for fair comparison between train (all tasks present in batch) and val/test (single task)
        if phase == LearningPhase.TRAIN:
            self.log(f"{phase.value}/loss", loss, batch_size=sum([b.size(0) for b in logits.values()]), prog_bar=True)
            # no add_dataloader_idx=False possible here, we log the aggregated loss in on_validation/test_epoch_end
            # see https://github.com/Lightning-AI/pytorch-lightning/issues/11126#issuecomment-1504866597
        else:
            self.loss_list.append(loss.item())
        return loss

    def compute_and_log_metrics(
        self, logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], phase: LearningPhase
    ):
        task_metrics = {
            LearningPhase.TRAIN: self.train_metrics,
            LearningPhase.VAL: self.val_metrics,
            LearningPhase.TEST: self.test_metrics,
        }[phase]
        for task in logits:
            if self.cfg.metrics.bootstrap and phase == LearningPhase.TEST:
                # logging of bootstrapped metrics requires distinction between mean and std, only updating here and
                # logging on test epoch end
                task_metrics[task].update(logits[task], targets[task])
            else:
                # default metric logging
                task_metrics[task](logits[task], targets[task])
                task_metrics[task](logits[task], targets[task])
                self.log_dict(task_metrics[task], batch_size=logits[task].size(0), add_dataloader_idx=False)

    def _generic_step(
        self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int, phase: LearningPhase
    ) -> torch.Tensor:
        """
        High level computational procedures during a batch passing.

        :param Dict[str, Dict[str, torch.Tensor]] batch: a batch as provided by `MultiTaskDataModule`
        :param int batch_idx: the index of the batch within one epoch
        :param LearningPhase phase: the current phase of the step
        :return: the loss computed on the batch
        """
        # pass forward, do TTA only during testing
        logits, targets = self.push_and_sort(batch, raise_on_error=True, perform_tta=phase == LearningPhase.TEST)
        # compute loss
        loss = self.compute_and_log_loss(logits, targets, phase=phase)
        # compute metrics
        self.compute_and_log_metrics(logits=logits, targets=targets, phase=phase)
        # log predictions of first batch from each epoch
        if batch_idx == 0:
            self.log_images_prediction_reference(batch=batch, logits=logits, targets=targets, phase=phase)
        phase_cms = {
            LearningPhase.TRAIN: self.train_cms,
            LearningPhase.VAL: self.val_cms,
            LearningPhase.TEST: self.test_cms,
        }[phase]
        for task in self.task_structs:
            if task in self.train_cms:
                phase_cms[task].update(logits[task], targets[task])
        # return loss to pass backward by lightning
        return loss

    def training_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self._generic_step(batch=batch, batch_idx=batch_idx, phase=LearningPhase.TRAIN)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        batch = self.reformat_batch_from_sequential(batch=batch, dataloader_idx=dataloader_idx)
        return self._generic_step(batch=batch, batch_idx=batch_idx, phase=LearningPhase.VAL)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        batch = self.reformat_batch_from_sequential(batch=batch, dataloader_idx=dataloader_idx)
        return self._generic_step(batch=batch, batch_idx=batch_idx, phase=LearningPhase.TEST)

    def log_confusion_matrix(self, phase: LearningPhase) -> None:
        """
        Logging utility for showing the confusion matrix of each epoch. Each logging also resets the cm in preparation
        for the next epoch.

        :param LeaningPhase phase: currently active learning phase to separate train, val and test
        :return:
        """
        # check config
        if not self.cfg.logging.cm or self.is_tuning:
            return

        phase_cms = {
            LearningPhase.TRAIN: self.train_cms,
            LearningPhase.VAL: self.val_cms,
            LearningPhase.TEST: self.test_cms,
        }[phase]
        for task in self.task_structs:
            if task not in phase_cms:
                continue
            # compute and reset cm
            cm = phase_cms[task].compute()
            phase_cms[task].reset()
            # render figure
            datamodule: MultiTaskDataModule = self.trainer.datamodule
            fig = render_confusion_matrix(
                cm=cm.detach().cpu().numpy(), classes=datamodule.task_datasets[task][phase].classes
            )
            # log figure
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    tag=f"{phase}/{task}/cm", figure=fig, global_step=self.trainer.global_step, close=True
                )
            else:
                logger.error(f"Unable to log prediction examples for {type(self.logger)} logger type.")
                break

    def on_train_epoch_end(self) -> None:
        self.log_confusion_matrix(phase=LearningPhase.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self.log_confusion_matrix(phase=LearningPhase.VAL)
        avg_loss = np.mean(self.loss_list)  # we ignore weighing by batch_size for now
        self.log("val/loss", avg_loss, add_dataloader_idx=False)
        if isinstance(self.logger, TensorBoardLogger):
            # for hparams view in tensorboard
            # (see also https://lightning.ai/docs/pytorch/stable/extensions/logging.html#logging-hyperparameters)
            self.log("hp_metric", avg_loss, add_dataloader_idx=False)
        self.loss_list.clear()

    def on_test_epoch_end(self) -> None:
        self.log_confusion_matrix(phase=LearningPhase.TEST)
        avg_loss = np.mean(self.loss_list)  # we ignore weighing by batch_size for now
        self.log("test/loss", avg_loss, add_dataloader_idx=False)
        self.loss_list.clear()
        if self.cfg.metrics.bootstrap:
            # actual logging of values
            for task, collection in self.test_metrics.items():
                try:
                    self.log_dict(collection.compute(), add_dataloader_idx=False)
                    collection.reset()
                except ValueError:
                    logger.info(
                        f"No metrics available for {task}. This may be due to evaluation on a different task "
                        f"or previous multitask model being evaluated solely on one task"
                    )

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        # if only a single task is used for sequential loading no dataloader_idx will be passed, so default is required
        batch = self.reformat_batch_from_sequential(batch=batch, dataloader_idx=dataloader_idx)
        logits, targets = self.push_and_sort(batch, raise_on_error=False, perform_tta=True)
        try:
            sample_ids = {task: batch[task]["sample_id"] for task in batch}
        except KeyError:
            warnings.warn("No image ids found during prediction")
            sample_ids = None
        return {
            task: {
                "logits": logits[task],
                "targets": targets[task] if targets is not None else None,
                "sample_ids": sample_ids[task],
            }
            for task in batch
            if batch[task]
        }

    def configure_optimizers(self):
        # docstring is provided by lighting
        # instantiate optimizer (and lr_scheduler if present) from cfg
        if self.lr:
            logger.info(f"Using learning rate {self.lr}.")
            optim = instantiate(self.cfg.optimizer, lr=self.lr, params=self.parameters())
        else:
            optim = instantiate(self.cfg.optimizer, params=self.parameters())
        if self.cfg.lr_scheduler["_target_"]:
            lr_scheduler = instantiate(self.cfg.lr_scheduler, optimizer=optim)
            return {"optimizer": optim, "lr_scheduler": lr_scheduler, "monitor": "val/loss"}
        return optim

    def get_metrics(self, struct: TaskStruct) -> List[Metric]:
        """
        Generates a collection of metrics, suited for the given task, based on the configs.

        :param TaskStruct struct: struct of the task
        :return: a list of torchmetrics metrics
        :rtype: List[torchmetrics.Metric]
        """
        # check task type
        route = CONFIGS_ROUTES[struct.task_type]
        # create Metrics
        mets = []
        for entry in self.cfg.metrics[route]:
            if "num_classes" in entry:
                mets.append(hydra.utils.instantiate(entry, num_classes=struct.num_classes))
            else:
                mets.append(hydra.utils.instantiate(entry))
        return mets

    def get_criteria(self) -> torch.nn.ModuleDict:
        """
        Generates the criteria modules. These correspond to the loss functions of each task. This is run once at the
        initialisation of the lightning module.

        :return: a dict of task to loss module
        """
        criteria = {}
        for name, struct in self.task_structs.items():
            criterion_cfg = self.cfg.loss[CONFIGS_ROUTES[struct.task_type]]
            if self.cfg.loss.class_weights:
                if self.cfg.sampling.balanced:
                    warnings.warn(
                        "provided criterion class weights but balanced sampling is activated please ensure"
                        " this behaviour is intended!"
                    )
                if self.cfg.loss.auto_activate_weighing:
                    raise MMLMisconfigurationException(
                        "provided criterion class weights but auto_activate_weighing is enabled."
                    )
                criteria[name] = instantiate(criterion_cfg, weight=torch.tensor(self.cfg.loss.class_weights))
            elif not self.cfg.sampling.balanced and self.cfg.loss.auto_activate_weighing:
                logger.info("Since sampling is unbalanced will try to auto activate loss weights for classes.")
                classes = TaskDataset.get_classes_from_idx_dict(struct.idx_to_class)
                class_weights = torch.tensor(max(struct.class_occ.values())) / torch.tensor(
                    [struct.class_occ[cl] for cl in classes], dtype=torch.float
                )
                try:
                    criteria[name] = instantiate(criterion_cfg, weight=class_weights)
                except InstantiationException as err:
                    logger.debug(err)
                    logger.warning(
                        f"Criterion {criterion_cfg['_target_']} does not accept weights. Will "
                        f"fall back to ignoring class imbalances."
                    )
                    criteria[name] = instantiate(criterion_cfg)
            else:
                criteria[name] = instantiate(criterion_cfg)
        return torch.nn.ModuleDict(criteria)

    @staticmethod
    def get_monitor_metric() -> Tuple[str, str]:
        """Returns the monitoring metric. This is used by Lightning to determine best model after training."""
        return "val/loss", "min"

    def setup_redirection(self, head: str, task: str) -> None:
        """
        Sets up a redirection to use model head "old" for data from task "new". This also includes preparation to
        use metrics and cm logging with "new" task name.

        :param str head: the existing model head name (likely learned before)
        :param str task: the new task that shall be passed through the old head
        :return:
        """
        logger.info(f"Redirecting data of task {task} through model head {head}.")
        if head not in self.task_structs:
            raise ValueError(f"Task {head} has no existing head.")
        if task in self.train_metrics:
            raise ValueError(f"Task {task} has already been redirected!")
        # data redirection
        self._task_to_head_mapping[task] = head
        # setup cm
        for phase_cms in [self.train_cms, self.val_cms, self.test_cms]:
            if head in phase_cms:
                phase_cms[task] = phase_cms[head].clone()
        # setup metric
        for phase_metrics in [self.train_metrics, self.val_metrics, self.test_metrics]:
            if head in phase_metrics:
                phase_metrics[task] = phase_metrics[head].clone()
                phase_metrics[task].prefix = phase_metrics[task].prefix.replace(f"/{head}/", f"/{task}/")
        # setup target
        self.targets[task] = self.targets[head]
