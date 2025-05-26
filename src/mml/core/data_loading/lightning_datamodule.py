# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra.utils
import lightning
import torch
from lightning.pytorch.accelerators.cuda import CUDAAccelerator
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from mml.core.data_loading.augmentations.albumentations import AlbumentationsAugmentationModule
from mml.core.data_loading.augmentations.augmentation_module import AugmentationModule, AugmentationModuleContainer
from mml.core.data_loading.augmentations.kornia import KorniaAugmentationModule
from mml.core.data_loading.augmentations.torchvision import TorchvisionAugmentationModule
from mml.core.data_loading.modality_loaders import ModalityLoader
from mml.core.data_loading.task_attributes import IMAGENET_MEAN, IMAGENET_STD, DataSplit, Modality, RGBInfo, TaskType
from mml.core.data_loading.task_dataset import TaskDataset
from mml.core.data_loading.task_struct import TaskStruct
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.utils import LearningPhase, catch_time

logger = logging.getLogger(__name__)


class MultiTaskDataModule(lightning.LightningDataModule):
    """
    This class wraps one or multiple :class:`~mml.core.data_loading.task_dataset.TaskDataset`s for lightning.
    Given the respective :class:`~mml.core.data_loading.task_structs.TaskStruct`s it takes care of setting up all
    correct data splits. It particularly interprets the following elements of the config:

     * `loaders`: the :class:`~mml.core.data_loading.modality_loaders.ModalityLoader`s
     * `preprocessing`: the :class:`~mml.core.data_loading.augmentations.augmentation_module.AugmentationModule`
     * `augmentations`: also :class:`~mml.core.data_loading.augmentations.augmentation_module.AugmentationModule`
     * all aspects with respect to sampling and the dataloader

    Importantly it provides the necessary lightning interface (e.g., :meth:`setup`, :meth:`train_dataloader`, etc.).
    """
    def __init__(self, task_structs: List[TaskStruct], cfg: DictConfig, fold: int = 0):
        logger.debug("Initializing Lightning datamodule.")
        super().__init__()
        self.task_structs: List[TaskStruct] = task_structs
        self.cfg = cfg
        self.roots: Dict[str, Path] = {
            struct.name: Path(self.cfg.data_dir) / struct.relative_root for struct in self.task_structs
        }
        self.fold: int = fold
        # attach batch size as attribute, this allows lightning to tune
        self.batch_size: Optional[int] = self.cfg.sampling.batch_size
        # variable storing the datasets after setup
        self.task_datasets: Dict[str, Dict[LearningPhase, TaskDataset]] = {
            struct.name: {} for struct in self.task_structs
        }
        # check that preprocessing matches caching option
        self.do_cache = {struct.name: self.cfg.sampling.enable_caching for struct in self.task_structs}
        _cache_counter = 0
        for struct in self.task_structs:
            if self.do_cache[struct.name]:
                if struct.preprocessed != self.cfg.preprocessing.id:
                    logger.error(
                        f"Requested caching on dataset {struct.name} but data is not preprocessed! Will deactivate"
                    )
                    self.do_cache[struct.name] = False
                else:
                    if _cache_counter + struct.num_samples > self.cfg.sampling.cache_max_size:
                        logger.info(
                            f"No caching for {struct.name}, since cache would likely exceed sampling.cache_max_size."
                        )
                        self.do_cache[struct.name] = False
                    else:
                        _cache_counter += struct.num_samples
        logger.debug(f"Cache information: {self.do_cache}")
        # new backend functionality
        self.has_gpu_augs = len(self.cfg.augmentations.gpu) > 0
        if self.has_gpu_augs:
            if self.cfg.trainer.accelerator == "cpu":
                raise MMLMisconfigurationException("Set trainer.accelerator to cpu but provided gpu augmentations.")
            if not self.cfg.allow_gpu:
                warnings.warn(
                    "Provided gpu augmentations but set allow_gpu=False. Please note that this config option"
                    " is intended for any non-lightning computations. This means to properly deactivate "
                    "gpu usage modify trainer.accelerator (e.g., set to cpu). Will continue with gpu "
                    "augmentations."
                )
        # will store gpu augmentations per device
        self.gpu_train_augs: Optional[AugmentationModule] = None
        self.gpu_test_augs: Optional[AugmentationModule] = None
        # control predict split
        self.predict_on = DataSplit.TEST

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: str) -> None:
        """
        Implements the lightning interface to prepare the datamodule. In particular sets up the
        :class:`~mml.core.data_loading.task_dataset.TaskDataset`s

        :param stage:
        :return:
        """
        logger.debug("Datamodule setup started")
        with catch_time() as timer:
            if stage == "fit":
                # prepare train and validation splits
                for struct in self.task_structs:
                    self.task_datasets[struct.name][LearningPhase.TRAIN] = TaskDataset(
                        root=self.roots[struct.name],
                        split=DataSplit.TRAIN,
                        fold=self.fold,
                        transform=self.get_cpu_transforms(struct=struct, phase=LearningPhase.TRAIN),
                        caching_limit=self.cfg.sampling.cache_max_size
                        if (self.do_cache[struct.name] and self.cfg.sampling.enable_caching)
                        else 0,
                        loaders=self.get_modality_loaders(),
                    )
                    self.task_datasets[struct.name][LearningPhase.VAL] = TaskDataset(
                        root=self.roots[struct.name],
                        split=DataSplit.VAL,
                        fold=self.fold,
                        transform=self.get_cpu_transforms(struct=struct, phase=LearningPhase.VAL),
                        caching_limit=self.cfg.sampling.cache_max_size
                        if (self.do_cache[struct.name] and self.cfg.sampling.enable_caching)
                        else 0,
                        loaders=self.get_modality_loaders(),
                    )
                    # if requested cache datasets
                    if self.do_cache[struct.name] and self.cfg.sampling.enable_caching:
                        # fill initial cache
                        for phase in [LearningPhase.TRAIN, LearningPhase.VAL]:
                            self.task_datasets[struct.name][phase].fill_cache(num_workers=self.cfg.num_workers)
            # prepare test split
            elif stage == "test":
                for struct in self.task_structs:
                    # no caching for test dataset!
                    self.task_datasets[struct.name][LearningPhase.TEST] = TaskDataset(
                        root=self.roots[struct.name],
                        split=DataSplit.TEST,
                        fold=0,
                        transform=self.get_cpu_transforms(struct=struct, phase=LearningPhase.TEST),
                        loaders=self.get_modality_loaders(),
                    )
            elif stage == "predict":
                for struct in self.task_structs:
                    # no caching for predict dataset!
                    self.task_datasets[struct.name][LearningPhase.TEST] = TaskDataset(
                        root=self.roots[struct.name],
                        split=self.predict_on,
                        fold=self.fold,
                        # always use test pipeline during predictions
                        transform=self.get_cpu_transforms(struct=struct, phase=LearningPhase.TEST),
                        loaders=self.get_modality_loaders(),
                    )
            # prepare gpu augmentations (per GPU device)
            if self.has_gpu_augs:
                self._set_gpu_transforms()
        logger.debug(f"MultiTaskDataModule setup time: {timer.minutes:.0f}m {timer.seconds:.2f}s")

    def train_dataloader(self, *args, **kwargs) -> CombinedLoader:
        return self._get_dataloader(phase=LearningPhase.TRAIN)

    def val_dataloader(self, *args, **kwargs) -> CombinedLoader:
        return self._get_dataloader(phase=LearningPhase.VAL)

    def test_dataloader(self, *args, **kwargs) -> CombinedLoader:
        return self._get_dataloader(phase=LearningPhase.TEST)

    def predict_dataloader(self, *args, **kwargs) -> CombinedLoader:
        return self._get_dataloader(phase=None, predict=True)

    def _get_dataloader(self, phase: Optional[LearningPhase], predict: bool = False) -> CombinedLoader:
        """
        Unifies the pytorch lightning dataloader functionalities. Produces a CombinedLoader.

        :param phase phase: learning phase to construct loader for, ignored if predict is true
        :param bool predict: if true phase is ignored and self.predict_on is used to deduce the loader
        """
        if phase is None and predict is False:
            raise ValueError("Must provide phase if not in predict mode")
        if predict:
            if phase is not None:
                warnings.warn("Calling _get_dataloader with predict=True will ignore phase kwarg and use TEST phase.")
            phase = LearningPhase.TEST
        loader_kwargs = {
            struct.name: self.get_loader_kwargs_from_cfg(phase=phase, task_name=struct.name)
            for struct in self.task_structs
        }
        return CombinedLoader(
            {
                struct.name: DataLoader(self.task_datasets[struct.name][phase], **loader_kwargs[struct.name])
                for struct in self.task_structs
            },
            mode="max_size_cycle" if (phase == LearningPhase.TRAIN) else "sequential",
        )

    def get_cpu_transforms(
        self, struct: TaskStruct, phase: LearningPhase = LearningPhase.TRAIN
    ) -> Union[AugmentationModule, AugmentationModuleContainer]:
        """
        Returns the necessary
        :param struct:
        :param phase:
        :return:
        """
        # preprocessing is always an albumentations pipeline
        pp_tfs = []
        if self.cfg.preprocessing.id != struct.preprocessed:
            if struct.preprocessed != "none":
                raise RuntimeError("Preprocessing on the fly only available from none (raw).")
            pp_tfs = self.cfg.preprocessing.pipeline
            warnings.warn(
                f"Task {struct.name} not yet preprocessed. Pipeline contains {len(pp_tfs)}"
                f" transforms. If you want to speed up training, preprocess this task beforehand."
            )
        means, stds = self.get_image_normalization(struct=struct)
        if "backend" not in self.cfg.augmentations.cpu:
            # no cpu augmentations requested, only pp augmentations
            return AlbumentationsAugmentationModule(
                cfg=pp_tfs,
                device="cpu",
                is_first=True,
                is_last=not self.has_gpu_augs,
                means=None if self.has_gpu_augs else means,
                stds=None if self.has_gpu_augs else stds,
                floatify=True,
                tensorize=True,
            )
            # otherwise there are cpu augmentations (at least for training)
        if phase != LearningPhase.TRAIN:
            aug_tfs = []
        else:
            aug_tfs = self.cfg.augmentations.cpu.pipeline
        # check if cpu backend matches pp backend
        if self.cfg.augmentations.cpu.backend == "albumentations":
            # now we can merge pp and aug transforms into one composed albumentations pipeline
            return AlbumentationsAugmentationModule(
                cfg=pp_tfs + aug_tfs,
                device="cpu",
                is_first=True,
                is_last=not self.has_gpu_augs,
                means=None if self.has_gpu_augs else means,
                stds=None if self.has_gpu_augs else stds,
                floatify=True,
                tensorize=True,
            )
        module_class = {
            "torchvision": TorchvisionAugmentationModule,
        }[self.cfg.augmentations.cpu.backend]

        kwargs = {
            "means": None if self.has_gpu_augs else means,
            "stds": None if self.has_gpu_augs else stds,
            "is_first": len(pp_tfs) == 0,
            "is_last": not self.has_gpu_augs,
            "device": "cpu",
        }
        if "backend" in self.cfg.augmentations.cpu and self.cfg.augmentations.cpu.backend == "torchvision":
            if "MixUp" in aug_tfs or "CutMix" in aug_tfs:
                if any(task.task_type != TaskType.CLASSIFICATION for task in self.task_structs):
                    raise MMLMisconfigurationException("MixUp and CutMix only applicable for classification tasks.")
                if len(self.task_structs) > 1:
                    raise MMLMisconfigurationException("MixUp and CutMix only applicable for single tasks.")
                kwargs["num_classes"] = self.task_structs[0].num_classes
        aug_module = module_class(cfg=aug_tfs, **kwargs)
        # no preprocessing necessary
        if len(pp_tfs) == 0:
            return aug_module
        # last scenario, we have to combine preprocessing with a non-albumentations cpu backend
        pp_module = AlbumentationsAugmentationModule(
            cfg=pp_tfs,
            device="cpu",
            is_first=True,
            is_last=False,
            means=None,
            stds=None,
            floatify=False,
            tensorize=True,
        )
        return AugmentationModuleContainer(modules=[pp_module, aug_module])

    def _set_gpu_transforms(self):
        """Might be called during setup (for each device individually) to generate gpu transforms."""
        mean, std = self.get_image_normalization(struct=self.task_structs[0])
        module_class = {"kornia": KorniaAugmentationModule, "torchvision": TorchvisionAugmentationModule}[
            self.cfg.augmentations.gpu.backend
        ]
        kwargs = {"means": mean, "stds": std, "is_first": False, "is_last": True, "device": "gpu"}
        if self.cfg.augmentations.gpu.backend == "torchvision":
            if "MixUp" in self.cfg.augmentations.gpu.pipeline or "CutMix" in self.cfg.augmentations.gpu.pipeline:
                if any(task.task_type != TaskType.CLASSIFICATION for task in self.task_structs):
                    raise MMLMisconfigurationException("MixUp and CutMix only applicable for classification tasks.")
                if len(self.task_structs) > 1:
                    raise MMLMisconfigurationException("MixUp and CutMix only applicable for single tasks.")
                kwargs["num_classes"] = self.task_structs[0].num_classes
        self.gpu_train_augs = module_class(cfg=self.cfg.augmentations.gpu.pipeline, **kwargs)
        self.gpu_test_augs = module_class(cfg={}, **kwargs)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Enables gpu augmentations after batch has been transferred to device."""
        if self.has_gpu_augs:
            if self.trainer.training:
                batch = self.gpu_train_augs(batch)
            else:
                batch = self.gpu_test_augs(batch)
        return batch

    def get_image_normalization(self, struct: TaskStruct) -> Tuple[Optional[RGBInfo], Optional[RGBInfo]]:
        """
        Returns the applied / required image normalization information.

        :return: tuple of means and stds for each of the channels, in case no normalization is applied returns None
            for both
        :rtype: Tuple[Optional[~mml.core.data_loading.task_attributes.RGBInfo],
            Optional[~mml.core.data_loading.task_attributes.RGBInfo]]
        """
        if self.cfg.augmentations.normalization == "imagenet":
            means = IMAGENET_MEAN
            stds = IMAGENET_STD
        elif self.cfg.augmentations.normalization == "task":
            if self.has_gpu_augs:
                raise MMLMisconfigurationException("GPU Augmentations require uniform normalization across tasks.")
            means = struct.means
            stds = struct.stds
        elif self.cfg.augmentations.normalization == "pretraining":
            # check model availability
            if not self.trainer.lightning_module:
                raise RuntimeError(
                    "Model not yet available for normalization config pretraining. Please open an issue."
                )
            means = self.trainer.lightning_module.model.required_mean
            stds = self.trainer.lightning_module.model.required_std
        elif self.cfg.augmentations.normalization is None:
            means = None
            stds = None
            warnings.warn(f"Deactivated normalization for task {struct.name}.", UserWarning)
        else:
            raise MMLMisconfigurationException(
                f"Config value {self.cfg.augmentations.normalization} unknown for "
                f"<augmentations.normalization>. Valid values are `imagenet`, "
                f"`task` and `null` (for no normalization)."
            )
        return means, stds

    def get_loader_kwargs_from_cfg(self, task_name: str, phase: LearningPhase = LearningPhase.TRAIN) -> Dict[str, Any]:
        try:
            # if we are in a lightning context, we assure the accelerator type
            gpu_acc = isinstance(self.trainer.accelerator, CUDAAccelerator)
        except AttributeError:
            # otherwise we fall back to the main config option
            gpu_acc = self.cfg.allow_gpu
        kwargs = {
            "num_workers": self.cfg.num_workers // len(self.task_structs),
            "pin_memory": gpu_acc,
            "drop_last": self.cfg.sampling.drop_last,
        }
        if kwargs["num_workers"] > 0:
            kwargs["persistent_workers"] = (
                True
                # it seems like there is some issue with deleting registered open files https://github.com/pytorch/pytorch/issues/91252
            )
        # during training
        if phase == LearningPhase.TRAIN:
            ds: TaskDataset = self.task_datasets[task_name][LearningPhase.TRAIN]
            if self.cfg.sampling.balanced:
                weights = self.get_dataset_balancing_weights(ds)
                num_samples = len(ds) if self.cfg.sampling.sample_num == 0 else self.cfg.sampling.sample_num
                kwargs["sampler"] = torch.utils.data.WeightedRandomSampler(weights, num_samples=num_samples)
            elif self.cfg.sampling.sample_num != 0:
                kwargs["sampler"] = torch.utils.data.RandomSampler(
                    ds, replacement=True, num_samples=self.cfg.sampling.sample_num
                )
            else:
                kwargs["shuffle"] = True
        # do not rely on config batch size, since lightning tuner might have determined datamodule attribute
        kwargs["batch_size"] = self.batch_size // len(self.task_structs)
        # if self.v2_gpu_train_augs is not None:
        #     kwargs['collate_fn'] = tv_tensor_collate
        return kwargs

    @staticmethod
    def get_dataset_balancing_weights(ds: TaskDataset) -> torch.Tensor:
        class_weights = torch.tensor(1.0) / torch.tensor([ds.class_occ[cl] for cl in ds.classes], dtype=torch.float)
        if ds.task_type == TaskType.CLASSIFICATION:
            # calc weights
            all_mapped_classes = [ds.loaders[Modality.CLASS].load(entry=s[Modality.CLASS]) for s in ds.samples]
            extracted_labels = torch.tensor(all_mapped_classes)
            weights = class_weights[extracted_labels]
        elif ds.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            modality_kind = Modality.CLASSES if Modality.CLASSES in ds.modalities else Modality.SOFT_CLASSES
            extracted_labels = [ds.loaders[modality_kind].load(entry=s[modality_kind]) for s in ds.samples]
            if modality_kind == Modality.CLASSES:
                empty_occ = sum([frame_labels.sum() == 0 for frame_labels in extracted_labels])
                empty_weight = torch.tensor(1.0) / torch.tensor(empty_occ, dtype=torch.float)
                weights = torch.tensor(
                    [class_weights[elem].mean() if elem.sum() > 0 else empty_weight for elem in extracted_labels]
                )
            elif modality_kind == Modality.SOFT_CLASSES:
                extracted_labels = torch.stack(extracted_labels).float()
                # replace missing classes with at least the value one (simulating a single occurrence)
                class_weights = torch.nan_to_num(class_weights, nan=None, posinf=1.0)
                weights = torch.matmul(extracted_labels, class_weights)
        else:
            raise RuntimeError("Balanced sampling only supported for (multilabel) classification tasks!")
        return weights

    def get_modality_loaders(self) -> Dict[Modality, ModalityLoader]:
        """Creates ModalityLoader instances from the config."""
        loader_dict = {}
        for k, v in self.cfg.loaders.items():
            loader_dict[Modality(k)] = hydra.utils.instantiate(v)
        return loader_dict

    def teardown(self, stage: Optional[str] = None) -> None:
        # unset potentially set torchvision setting
        tv_tensors.set_return_type("Tensor")
