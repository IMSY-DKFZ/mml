# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import importlib
import itertools
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import omegaconf
import quapy
import quapy.data.datasets
import sklearn.base
import torch
import torch.nn as nn
from omegaconf import DictConfig
from psrcal.calibration import AffineCalLogLoss, calibrate
from quapy.method.aggregative import ACC
from torchmetrics import BootStrapper, MetricCollection
from tqdm import tqdm

from mml.core.data_loading.task_attributes import DataSplit, TaskType
from mml.core.scripts.decorators import beta
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.model_storage import EnsembleStorage, ModelStorage
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import ARG_SEP, TAG_SEP, LearningPhase

importlib.reload(warnings)

logger = logging.getLogger(__name__)


@beta("Postprocessing mode is in beta.")
class PostprocessScheduler(AbstractBaseScheduler):
    """
    Scheduler for post-processing predictions, supports the following sub-routines:
    - calibrate
    - ensemble
    """

    def __init__(self, cfg: DictConfig):
        if not cfg.pivot.name:
            raise MMLMisconfigurationException("Must provide a pivot task for post-processing!")
        self.models: Dict[str, List[ModelStorage]] = {}  # will hold relevant model storages
        self.eval_logits: Dict[Tuple[str, int], torch.Tensor] = {}  # will hold loaded predictions for speed
        self.eval_labels: Optional[torch.Tensor] = None  # will hold evaluation labels
        super(PostprocessScheduler, self).__init__(cfg=cfg, available_subroutines=["calibrate", "ensemble"])
        if self.cfg.mode.eval_frac < 0 or self.cfg.mode.eval_frac >= 1:
            raise MMLMisconfigurationException(
                f"mode.eval_frac={self.cfg.mode.eval_frac} is not valid! Should be strictly between 0 and 1!"
            )
        if self.cfg.mode.temperature >= 1 or self.cfg.mode.weights_temperature >= 1:
            raise MMLMisconfigurationException("ensemble search temperatures must be below 1!")
        if self.cfg.mode.temperature < 0 or self.cfg.mode.weights_temperature < 0:
            raise MMLMisconfigurationException("ensemble search temperatures must be at least 0!")

    def after_preparation_hook(self):
        # checks: only support classification pivots for now - not even multiclass or regression
        struct = self.get_struct(self.pivot)
        if struct.task_type != TaskType.CLASSIFICATION:
            raise MMLMisconfigurationException("Only classification tasks are supported for post-processing!")

    def create_routine(self):
        """
        This scheduler implements two sub-routines, calibration as well as combined selection and prediction for
        ensembles. The routine takes care of finding any relevant previous predictions in this project.
        """
        # predictions on the pivot task may be done from models trained upon any other task (!)
        all_models = {
            task: self.fm.reusables[task]["models"] for task in self.fm.reusables if "models" in self.fm.reusables[task]
        }
        # models must have a corresponding prediction entry to match the current pivot
        for task in all_models:
            relevant_models = [m for m in all_models[task] if self.pivot in m.predictions]
            if len(relevant_models) > 0:
                self.models[task] = sorted(relevant_models, key=lambda m: m.created)
        if len(self.models) == 0:
            raise RuntimeError(
                "No models available for post-processing! You may reuse existing models via "
                "reuse.models=SOME_PROJECT or short reuse=current for reusing models from the current "
                "project. To train models run mml train with your current pivot task."
            )
        if "calibrate" in self.subroutines:
            for task in self.models:
                for model_idx, model in enumerate(self.models[task]):
                    self.commands.append(self.calibrate_predictions)
                    self.params.append([task, model_idx])
        if "ensemble" in self.subroutines:
            self.commands.append(self.select_ensemble)
            self.params.append([])

    def calibrate_predictions(self, task: str, model_index: int):
        logger.info(f"Calibrating predictions {model_index} for task " + self.highlight_text(task))
        model = self.models[task][model_index]
        # also consider nested predictions
        predictions = [
            pred
            for pred in model.predictions
            if pred == self.pivot or pred.startswith(f"{self.pivot}{TAG_SEP}nested{ARG_SEP}")
        ]
        for pred in predictions:
            # load prediction
            all_splits_prediction = torch.load(model.predictions[pred])
            # we use the validation split as base for inferring calibration parameters
            if DataSplit.VAL.value not in all_splits_prediction:
                raise RuntimeError(
                    f"No predictions have been made on validation data for model @ {model._stored}"
                    f"and prediction on {pred} (@ {model.predictions[pred]})."
                )
            all_logits = {}
            all_labels = {}
            for split in [DataSplit.VAL, DataSplit.TEST, DataSplit.UNLABELLED]:
                if split.value not in all_splits_prediction or len(all_splits_prediction[split.value]) == 0:
                    continue
                all_logits[split] = torch.stack([item["logits"] for item in all_splits_prediction[split.value]]).float()
                if split != DataSplit.UNLABELLED:
                    all_labels[split] = torch.tensor([item["target"] for item in all_splits_prediction[split.value]])
            if self.cfg.mode.prior == "test":
                # derive optimal priors from test data
                prior = torch.bincount(all_labels[DataSplit.TEST])
                # to avoid any zero division we assume a minimum probability
                prior = np.clip(prior, a_min=1e-8, a_max=None)
                # scaling for convergence stability
                prior = prior / prior.sum()
            elif self.cfg.mode.prior == "quantify":
                use_as_train_split = DataSplit.TEST
                if DataSplit.UNLABELLED not in all_logits:
                    warnings.warn(
                        "Calibration prior quantify expects predictions on unlabeled split! Any prediction "
                        "without unlabeled split (e.g. from nested training) will fall back to calibrate"
                        "according to the validation split prevalences"
                    )
                    prior = None
                else:
                    if DataSplit.TEST not in all_logits:
                        warnings.warn(
                            "No test set predictions found - will fall back to val set in order to quantify "
                            "unlabeled data prevalences."
                        )
                        use_as_train_split = DataSplit.VAL
                    # convert data to qp format
                    qp_train_data = quapy.data.LabelledCollection(
                        torch.softmax(all_logits[use_as_train_split], dim=1),
                        all_labels[use_as_train_split],
                        classes=list(range(self.get_struct(self.pivot).num_classes)),
                    )
                    qp_test_data = quapy.data.LabelledCollection(
                        [], [], classes=list(range(self.get_struct(self.pivot).num_classes))
                    )
                    qp_dset = quapy.data.base.Dataset(training=qp_train_data, test=qp_test_data)

                    # define dummy classifier to be used with quapy
                    class IdentityClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
                        def __init__(self, n_classes=2):
                            self.n_classes = n_classes
                            self.classes_ = np.arange(n_classes)

                        def fit(self, X, y=None):
                            return self

                        def predict_proba(self, X):
                            return X

                        def predict(self, X):
                            probas = self.predict_proba(X)
                            return np.argmax(probas, axis=1)

                    identity_class = IdentityClassifier(self.get_struct(self.pivot).num_classes)
                    quapy_model = ACC(identity_class)
                    quapy_model.fit(qp_dset.training)
                    prior = quapy_model.quantify(torch.softmax(all_logits[DataSplit.UNLABELLED], dim=1))
                    # to avoid any zero division we assume a minimum probability
                    prior = np.clip(prior, a_min=1e-8, a_max=None)
                    prior = prior / np.sum(prior)
            elif isinstance(self.cfg.mode.prior, list):
                # use given priors
                prior = self.cfg.mode.prior
            elif self.cfg.mode.prior == "val":
                # infer priors on validation data, psrcal infers them
                prior = None
            else:
                raise MMLMisconfigurationException(f"Invalid prior strategy {self.cfg.mode.prior}")
            # update both unlabelled, and test predictions if existing
            calibrated_splits = [split for split in all_logits if split != DataSplit.VAL]
            if len(calibrated_splits) > 1:
                stacked_test_logits = torch.cat([all_logits[split] for split in calibrated_splits])
            else:
                stacked_test_logits = all_logits[calibrated_splits[0]]
            # calibrate and return updated predictions
            cal_logits, _ = calibrate(
                trnscores=all_logits[DataSplit.VAL],
                trnlabels=all_labels[DataSplit.VAL],
                tstscores=stacked_test_logits,
                calclass=AffineCalLogLoss,
                bias=True,
                priors=prior,
                quiet=True,
            )
            # de-stack returned logits
            if len(calibrated_splits) > 1:
                split_calibrated_logits = torch.split(
                    cal_logits.detach(),
                    split_size_or_sections=[all_logits[split].size(0) for split in calibrated_splits],
                )
                cal_logits = {split: logits for split, logits in zip(calibrated_splits, split_calibrated_logits)}
            else:
                cal_logits = {calibrated_splits[0]: cal_logits.detach()}
            # store updated predictions
            for split, logits in cal_logits.items():
                for case_idx, case_logits in enumerate(logits):
                    all_splits_prediction[split.value][case_idx]["calibrated"] = case_logits
            torch.save(obj=all_splits_prediction, f=model.predictions[pred])

    def select_ensemble(self):
        pivot_struct = self.get_struct(self.pivot)
        # determine evaluation samples
        datamodule = self.create_datamodule(task_structs=pivot_struct)
        datamodule.setup(stage="test")
        task_dataset = datamodule.task_datasets[self.pivot][LearningPhase.TEST]
        all_test_ids = task_dataset._sample_ids
        num_eval_samples = int(self.cfg.mode.eval_frac * len(all_test_ids))
        assert 0 < num_eval_samples < len(all_test_ids), "Rounding lead to invalid number of eval samples"
        rng = np.random.default_rng(seed=self.cfg.seed)
        eval_ids = rng.choice(all_test_ids, size=num_eval_samples, replace=False)
        logger.info(
            f"Will perform ensemble selection based on {len(eval_ids)} samples from test split "
            f"({len(eval_ids) / len(all_test_ids):.2%})."
        )
        # use the loss function with optional class weights, but need to ensure no unintended weighting is performed
        with omegaconf.open_dict(self.cfg):
            self.cfg.sampling.balanced = False
            self.cfg.loss.auto_activate_weighing = False
        lightning_module = self.create_model(task_structs=[pivot_struct])
        evaluate_criteria = lightning_module.get_criteria()[self.pivot]
        logger.info(f"Will perform ensemble selection based on {evaluate_criteria}.")
        single_results = {}
        with tqdm(total=sum(map(len, self.models.values())), desc="Evaluating single models:") as pbar:
            for task in self.models:
                for model_idx, model in enumerate(self.models[task]):
                    preds = torch.load(model.predictions[self.pivot])
                    # re-arrange test split based on ID
                    test_preds = {item["sample_id"]: item for item in preds[DataSplit.TEST.value]}
                    try:
                        # test for calibrated logits
                        logits = torch.stack([test_preds[sample_id]["calibrated"] for sample_id in eval_ids]).float()
                    except KeyError:
                        warnings.warn(
                            "Found uncalibrated predictions. It is recommended to perform post-hoc re-calibration."
                        )
                        logits = torch.stack([test_preds[sample_id]["logits"] for sample_id in eval_ids]).float()
                    labels = torch.tensor([test_preds[sample_id]["target"] for sample_id in eval_ids])
                    # store extracted predictions for later re-use
                    self.eval_logits[(task, model_idx)] = logits
                    if self.eval_labels is None:
                        self.eval_labels = labels
                    else:
                        if not torch.equal(labels, self.eval_labels):
                            raise RuntimeError("Difference in reference detected!")
                    single_results[(task, model_idx)] = evaluate_criteria(logits, labels).item()
                    pbar.update()
        logger.info(f"Evaluated {len(single_results)} individual models.")
        # these will keep the best result in memory
        best_ensemble = None
        best_performance = None
        best_weights = None
        # determine budget and search space
        remaining_budget = self.cfg.mode.budget
        sorted_task_idx_pairs = sorted(single_results.keys(), key=lambda x: single_results[x])
        search_space = {
            ensemble_size: list(itertools.combinations(sorted_task_idx_pairs, ensemble_size))
            for ensemble_size in range(2, self.cfg.mode.max_ensemble_size + 1)
        }
        # perform ensemble search
        with tqdm(total=self.cfg.mode.budget, desc="Testing ensembles") as bar:
            while remaining_budget > 0:
                # select sub search space
                remaining_ensemble_sizes = [
                    ensemble_size for ensemble_size in search_space if len(search_space[ensemble_size]) > 0
                ]
                if len(remaining_ensemble_sizes) == 0:
                    logger.info(
                        "No more ensemble combinations available for searching - you may increase "
                        "mode.weights_budget to search more weighted combinations of extisting models or "
                        "train and predict more."
                    )
                    break
                ensemble_size = rng.choice(remaining_ensemble_sizes)
                # select models
                if rng.random() > self.cfg.mode.temperature:
                    # keep the order of single model testing
                    ensemble_models = search_space[ensemble_size].pop(0)
                else:
                    # do a "wild" guess
                    ensemble_models = search_space[ensemble_size].pop(rng.integers(len(search_space[ensemble_size])))
                # in case of weight optimization we sample multiple times
                if self.cfg.mode.weights_budget > 0:
                    # give default preference to better models
                    alpha = np.asarray([1 / single_results[(task, model_idx)] for task, model_idx in ensemble_models])
                    weight_variations = rng.dirichlet(
                        alpha=alpha * (1 - self.cfg.mode.weights_temperature), size=self.cfg.mode.weights_budget
                    ).tolist()
                else:
                    weight_variations = [1 / ensemble_size] * ensemble_size
                for weights in weight_variations:
                    # mix predictions
                    mixed_logits = torch.sum(
                        torch.stack(
                            [
                                weight * self.eval_logits[(task, model_idx)]
                                for weight, (task, model_idx) in zip(weights, ensemble_models)
                            ]
                        ),
                        dim=0,
                    )
                    performance = evaluate_criteria(mixed_logits, self.eval_labels).item()
                    # create ensemble
                    if best_performance is None or best_performance > performance:  # lower is better
                        best_performance = performance
                        best_weights = weights
                        best_ensemble = ensemble_models
                    remaining_budget -= 1
                    bar.update()
        logger.info("Done with ensemble search. Will perform full prediction.")
        # now we generate the full predictions on test and unlabeled, need to load these first
        full_test_labels = None
        all_final_logits = {DataSplit.TEST: [], DataSplit.UNLABELLED: []}
        unlabeles_sample_ids = None
        for task, model_idx in best_ensemble:
            preds = torch.load(self.models[task][model_idx].predictions[self.pivot])
            for split in [DataSplit.TEST, DataSplit.UNLABELLED]:
                # re-arrange split based on ID
                try:
                    split_preds = {item["sample_id"]: item for item in preds[split.value]}
                except KeyError:
                    warnings.warn(
                        f"No predictions for split {split} found for model @ {model._stored}"
                        f"and prediction on {self.pivot} (@ {model.predictions[self.pivot]})."
                    )
                    continue
                if len(split_preds) == 0:
                    warnings.warn(
                        f"No predictions for split {split} found for model @ {model._stored}"
                        f"and prediction on {self.pivot} (@ {model.predictions[self.pivot]})."
                    )
                    continue
                if split == DataSplit.UNLABELLED and unlabeles_sample_ids is None:
                    unlabeles_sample_ids = list(split_preds.keys())
                sample_id_order = {DataSplit.UNLABELLED: unlabeles_sample_ids, DataSplit.TEST: all_test_ids}[split]
                try:
                    # test for calibrated logits
                    logits = torch.stack(
                        [split_preds[sample_id]["calibrated"] for sample_id in sample_id_order]
                    ).float()
                except KeyError:
                    logits = torch.stack([split_preds[sample_id]["logits"] for sample_id in sample_id_order]).float()
                # gather logits
                all_final_logits[split].append(logits)
                # ensure test set labels match
                if split == DataSplit.TEST:
                    labels = torch.tensor([split_preds[sample_id]["target"] for sample_id in all_test_ids])
                    # compare labels
                    if full_test_labels is None:
                        full_test_labels = labels
                    else:
                        if not torch.equal(labels, full_test_labels):
                            raise RuntimeError("Difference in reference detected!")
        assert all(
            len(all_final_logits[split]) in [0, len(best_ensemble)] for split in [DataSplit.TEST, DataSplit.UNLABELLED]
        ), "Some models make predictions that other do not!"
        # merge predictions
        mixed_logits = {}
        for split in [DataSplit.TEST, DataSplit.UNLABELLED]:
            if len(all_final_logits[split]) == 0:
                continue
            mixed_logits[split] = torch.sum(
                torch.stack([weight * logits for weight, logits in zip(best_weights, all_final_logits[split])]), dim=0
            )
        if DataSplit.TEST in mixed_logits:
            # assess performance on remaining test data
            indxs = [all_test_ids.index(elem) for elem in all_test_ids if elem not in eval_ids]
            assess_logits = mixed_logits[DataSplit.TEST][indxs]
            assess_labels = full_test_labels[indxs]
            # construct metrics
            evaluate_metrics = lightning_module.get_metrics(pivot_struct)
            if self.cfg.metrics.bootstrap:
                # wrap in bootstrapper, needs to proceed with dict to avoid duplicate naming of bootstrapper in collection
                evaluate_metrics = {
                    met._get_name(): BootStrapper(met, num_bootstraps=self.cfg.metrics.bootstrap)
                    for met in evaluate_metrics
                }
            evaluate_metrics = MetricCollection(evaluate_metrics)
            evaluation = {k: v.item() for k, v in evaluate_metrics(assess_logits, assess_labels).items()}
            logger.info(
                f"Evaluation on {len(all_test_ids) - len(eval_ids)} samples from test split "
                f"({(len(all_test_ids) - len(eval_ids)) / len(all_test_ids):.2%})."
            )
            if self.cfg.metrics.bootstrap:
                for metric in evaluate_metrics.keys():
                    logger.info(
                        f"{metric} = {evaluation[str(metric) + '_mean']:.2f} ± {evaluation[str(metric) + '_std']:.2f}"
                    )
            else:
                for metric, value in evaluation.items():
                    logger.info(f"{metric} = {value:.2f}")
        else:
            evaluation = {}
        # correct prediction format
        ensemble_prediction = {}
        for split in [DataSplit.TEST, DataSplit.UNLABELLED]:
            if split not in mixed_logits:
                # no predictions here
                continue
            probabilities = nn.functional.softmax(mixed_logits[split], dim=1)
            ensemble_prediction[split] = []
            sample_id_order = {DataSplit.UNLABELLED: unlabeles_sample_ids, DataSplit.TEST: all_test_ids}
            for sample_idx, sample_id in enumerate(sample_id_order):
                sample = {
                    "logits": mixed_logits[split][sample_idx],
                    "sample_id": sample_id,
                    "probabilities": probabilities[sample_idx],
                }
                if split == DataSplit.TEST:
                    sample["target"] = full_test_labels[sample_idx]
                ensemble_prediction[split].append(sample)
        # store prediction
        pred_path = self.fm.construct_saving_path(
            obj=ensemble_prediction, key="predictions", task_name=self.pivot, file_name="preds-ensembled.pt"
        )
        torch.save(ensemble_prediction, pred_path)
        logger.info(f"Predictions are stored @ {pred_path}.")
        # store ensemble
        search_params = omegaconf.OmegaConf.to_container(self.cfg.mode, resolve=True)
        search_params.pop("scheduler")
        search_params.pop("subroutines")
        search_params.pop("prior")
        search_params["seed"] = self.cfg.seed
        storage = EnsembleStorage(
            performance=best_performance,
            weights=best_weights,
            members=[self.models[task][model_idx]._stored for task, model_idx in best_ensemble],
            predictions={self.pivot: pred_path},
            metrics=evaluation,
            search_params=search_params,
        )
        ensemble_path = storage.store(task_struct=pivot_struct)
        logger.info(f"Ensemble storage stored @ {ensemble_path}.")
