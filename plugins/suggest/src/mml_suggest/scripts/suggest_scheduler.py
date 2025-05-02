# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import itertools
import logging
import warnings
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scipy
from omegaconf import DictConfig, OmegaConf

from mml.core.models.lightning_single_frame import CONFIGS_ROUTES
from mml.core.scripts.decorators import beta
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.pipeline_configuration import PipelineCfg
from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler
from mml.core.scripts.utils import ARG_SEP, TAG_SEP

logger = logging.getLogger(__name__)


@beta("Suggest mode is in beta.")
class SuggestScheduler(AbstractBaseScheduler):
    """
    AbstractBaseScheduler implementation for transferring task specific knowledge. Includes the following subroutines:
    - suggest
    """

    def __init__(self, cfg: DictConfig):
        # initialize
        super(SuggestScheduler, self).__init__(cfg=cfg, available_subroutines=["suggest"])
        # some checks
        if not self.pivot:
            raise MMLMisconfigurationException("Suggest mode requires a pivot task")
        # load distance files
        self.distances: Dict[str, pd.DataFrame] = {}
        for dist in [self.cfg.mode.distances] if isinstance(self.cfg.mode.distances, str) else self.cfg.mode.distances:
            try:
                distance_file = self.fm.global_reusables[dist]
            except KeyError:
                raise MMLMisconfigurationException(
                    f"Distance {dist} is not present in global artefacts. Make sure "
                    f"to set reuse.{dist}=SOME_PROJECT properly."
                )
            df = pd.read_csv(distance_file, index_col=0)
            if self.pivot not in df.columns:
                raise RuntimeError(f"pivot task {self.pivot} not in distance {dist} (@ {distance_file})")
            # normalize distances
            _data = scipy.stats.zscore(df.to_numpy().astype(float), axis=None, nan_policy="omit")
            if np.isnan(_data).all():
                warnings.warn(
                    f"Something went wrong while normalising distance {dist}. Will fall back to normalized "
                    f"instead of standardized transformation."
                )
                _data = df.to_numpy().astype(float) / df.abs().max(axis=None)
                if np.isnan(_data).all():
                    raise RuntimeError(
                        f"Something went wrong while normalising distance {dist} - normalization failed."
                    )
            self.distances[dist] = pd.DataFrame(_data, columns=df.columns, index=df.index)
        if len(self.distances) == 0:
            raise MMLMisconfigurationException("Must provide at least one valid value for mode.distances")
        # assert source tasks are present in all distance files
        if any([src not in dist.index for src in self.cfg.task_list for dist in self.distances.values()]):
            raise RuntimeError("All source tasks need to be present in all distances.")
        # auto-infer source tasks if not given
        if len(self.cfg.task_list) == 1 and not (self.cfg.tagging.all or self.cfg.tagging.variants):
            logger.info("No task_list given, will include all tasks from (all) distances.")
            sources = None
            for distance_df in self.distances.values():
                if sources is None:
                    sources = set(distance_df.index.tolist())
                else:
                    sources.intersection_update(distance_df.index.tolist())
            counter = 0
            for source in sources:
                if source.split(TAG_SEP)[0] == self.pivot:  # exclude pivot and pivot variants
                    continue
                counter += 1
                self.cfg.task_list.extend(self.get_nested_variants(source))
            logger.info(f"Added all {counter} found tasks as potential source tasks (plus nested variants).")

    def get_nested_variants(self, task: str) -> List[str]:
        """Checks for existing nested variants in the file manager index and adds them to a plain task."""
        variants = [t for t in self.fm.task_index if t.startswith(f"{task}{TAG_SEP}nested{ARG_SEP}")]
        # we assume a single nesting does not "change" the task similarity, but exclude multi level tagging
        variants = [t for t in variants if TAG_SEP not in t[t.find(f"{t}{TAG_SEP}nested{ARG_SEP}") :]]
        variants.append(task)
        return variants

    @staticmethod
    def get_unnested_variant(task) -> str:
        """Returns the unnested variant of a task. ALso works for non-nested tasks."""
        if f"{TAG_SEP}nested{ARG_SEP}" not in task:
            return task
        return "".join(task.split(f"{TAG_SEP}nested{ARG_SEP}")[:-1])

    def create_routine(self):
        """
        This scheduler implements only a single subroutine.

        :return: None
        """
        if "suggest" in self.subroutines:
            # -- add suggestion command
            self.commands.append(self.suggest)
            self.params.append([])

    def suggest(self):
        target_struct = self.get_struct(self.pivot)
        sources_with_models = [s for s in self.cfg.task_list if len(self.get_struct(s).models) > 0 and s != self.pivot]
        source_structs = {s: self.get_struct(s) for s in sources_with_models}
        if len(source_structs) == 0:
            raise RuntimeError(
                "No source task with existing ModelStorage found. Have you set reuse.models=SOME_PROJECT?"
            )
        # make distance suggestions more precise
        all_unnested_sources = set(self.get_unnested_variant(src) for src in sources_with_models)
        quantiles = {dist: np.nanquantile(df.values, q=self.cfg.mode.sim_cutoff) for dist, df in self.distances.items()}
        distance_series = {dist: df[self.pivot] for dist, df in self.distances.items()}
        sources_within_quantiles = {dist: series[series <= quantiles[dist]] for dist, series in distance_series.items()}
        valid_source_pool = set(
            itertools.chain(*[series.index.tolist() for series in sources_within_quantiles.values()])
        )
        if len(valid_source_pool) == 0:
            raise RuntimeError(
                "No source task is within reach. Consider adding more source tasks, lowering "
                "cfg.mode.sim_cutoff and also make sure distance computations are completed."
            )
        # prepare pipelines for each available model storage (for each of the source tasks, merging nested variants)
        all_pipelines = {self.get_unnested_variant(src): {} for src in source_structs}
        for src in source_structs:
            unnested_src = self.get_unnested_variant(src)
            for model_storage in source_structs[src].models:
                pipeline = PipelineCfg.load(path=model_storage.pipeline, pipeline_keys=self.cfg.mode.pipeline_keys)
                # loss interpretation is depending on some factors - we hash those and treat them individually
                loss_config_str = OmegaConf.to_yaml(
                    pipeline.pipeline_cfg.loss[CONFIGS_ROUTES[source_structs[src].task_type]]
                )
                if pipeline.pipeline_cfg.loss.class_weights:
                    loss_config_str += str(pipeline.pipeline_cfg.loss.class_weights)
                elif pipeline.pipeline_cfg.loss.auto_activate_weighing and not pipeline.pipeline_cfg.sampling.balanced:
                    loss_config_str += "balanced_class_weights"
                loss_id = hash(loss_config_str)
                if loss_id not in all_pipelines[unnested_src]:
                    all_pipelines[unnested_src][loss_id] = []
                all_pipelines[unnested_src][loss_id].append((model_storage.performance, pipeline))
            for loss_id in all_pipelines[unnested_src]:
                # only use best performing models
                quantile_cutoff = np.nanquantile(
                    [elem[0] for elem in all_pipelines[unnested_src][loss_id]], q=self.cfg.mode.perf_cutoff
                )
                all_pipelines[unnested_src][loss_id] = [
                    (perf, pipeline)
                    for perf, pipeline in all_pipelines[unnested_src][loss_id]
                    if perf <= quantile_cutoff
                ]
                # sort ascending, such that best pipelines are positioned first
                all_pipelines[unnested_src][loss_id].sort(key=lambda x: x[0])
        # now iterate over all pipeline keys and determine (1) which distance to use (2) which sources to use
        sources_per_key = {key: {} for key in self.cfg.mode.pipeline_keys}
        for pipeline_key in self.cfg.mode.pipeline_keys:
            # set default preferences for this pipeline key
            relation = {dist: 1.0 for dist in self.distances}
            # load preferred relations
            if pipeline_key in self.cfg.mode.relation:
                for dist in self.cfg.mode.relation[pipeline_key]:
                    if dist in relation:  # ignore preferences not available in current distances
                        relation[dist] = self.cfg.mode.relation[pipeline_key][dist]
            # normalize to one
            rel_sum = sum(relation.values())
            relation = {dist: val / rel_sum for dist, val in relation.items()}
            # for each distance sample some source tasks
            for dist, series in sources_within_quantiles.items():
                good_keys = series.index.intersection(all_unnested_sources)
                for src, val in series.loc[good_keys].to_dict().items():
                    # turn distance into likelihood by inverting
                    if src not in sources_per_key[pipeline_key]:
                        sources_per_key[pipeline_key][src] = -val * relation[dist]
                    else:
                        sources_per_key[pipeline_key][src] -= val * relation[dist]
        # determine for each pipeline key the previous pipeline entries to use
        pipelines_per_key = {key: [] for key in self.cfg.mode.pipeline_keys}
        for pipeline_key in self.cfg.mode.pipeline_keys:
            for source, weight in sources_per_key[pipeline_key].items():
                n_loss_ids = len(all_pipelines[source])
                for loss_id_pipeline_list in all_pipelines[source].values():
                    list_length = len(loss_id_pipeline_list)
                    for list_idx, (perf, pipeline) in enumerate(loss_id_pipeline_list):
                        # position interpolates linearly between 1 and 0.1
                        if list_length == 1:
                            weight_by_position = 0.5
                        else:
                            weight_by_position = 1 - list_idx * (0.9 / (list_length - 1))
                        pipelines_per_key[pipeline_key].append((weight_by_position * weight / n_loss_ids, pipeline))
            # normalize and sort per entry
            _sum_weights = sum(elem[0] for elem in pipelines_per_key[pipeline_key])
            pipelines_per_key[pipeline_key] = [
                (w / _sum_weights, pipeline) for w, pipeline in pipelines_per_key[pipeline_key]
            ]
            pipelines_per_key[pipeline_key].sort(key=lambda x: x[0], reverse=True)  # descending order
        # find a "pivot" pipeline with the highest weight, this will be the starting point for merging
        highest_prob_key = max(self.cfg.mode.pipeline_keys, key=lambda key: pipelines_per_key[key][0])
        prim_pipeline = pipelines_per_key[highest_prob_key][0][1]
        for _ in range(self.cfg.mode.n):
            new_pipeline = prim_pipeline.clone()
            # start merging the pipeline - iterate over config elements in a stack
            stack = self.cfg.mode.pipeline_keys
            rng = np.random.default_rng()
            while len(stack) > 0:
                path = stack.pop()
                # in case this is a dictionary we attach all sub-entry paths to the stack
                if OmegaConf.is_dict(OmegaConf.select(new_pipeline.pipeline_cfg, path)):
                    stack.extend(
                        [path + "." + str(key) for key in OmegaConf.select(new_pipeline.pipeline_cfg, path).keys()]
                    )
                    continue
                # if no dict, the entries should be either lists or primitives, we extract the respective entries
                entry_options = {}
                for prob, pipeline in pipelines_per_key[path.split(".")[0]]:
                    val = OmegaConf.select(pipeline.pipeline_cfg, path, default=None)
                    if val in entry_options:
                        entry_options[val] += prob
                    else:
                        entry_options[val] = prob
                if rng.random() < self.cfg.mode.temperature:
                    # trigger uniform sampling
                    entry_options = {entry: 1 / len(entry_options) for entry in entry_options}
                sampled_entry = self._merge_values(entries=entry_options)
                logging.debug(f"Updating {path} to {sampled_entry} (merged from {entry_options}).")
                OmegaConf.update(new_pipeline.pipeline_cfg, key=path, value=sampled_entry)
            # finally store blueprint
            new_pipeline.store(task_struct=target_struct, as_blueprint=True)
        logger.info(f"Successfully compiled {self.cfg.mode.n} pipeline blueprint(s).")

    @staticmethod
    def _merge_values(entries: Dict[Union[None, str, float, int, list], float]) -> Union[None, str, float, int]:
        """
        Merges potential entries within a pipeline. Expects all possible entry values with attached probability value.
        Probabilities mus sum to one!

        :param Dict[Union[None, str, float, int, list], float] entries: dictionary with all possible entries as keys and
          their respective probability as values. It is important that only a single type of entries is allowed (e.g.
          all boolean, or all int)
        :return: the chosen entry
        """
        if not np.isclose(sum(entries.values()), 1):
            raise ValueError("Probabilities do not sum to one!")
        # remove Nones
        sum_non_none = sum([prob for val, prob in entries.items() if val is not None])
        entries = {val: prob / sum_non_none for val, prob in entries.items() if val is not None}
        types = [type(entry) for entry in entries.keys()]
        # simple cases
        if len(entries) == 0:
            return None
        if len(entries) == 1:
            return next(iter(entries.keys()))
        # for multiple values check consistency and chose a primary
        assert all([t == types[0] for t in types]), f"inconsistent types {types}"
        primary = max(entries.items(), key=lambda x: x[1])[0]
        if isinstance(primary, bool):
            # treated similarly to int, but converted back to boolean
            return bool(sum([k * v for k, v in entries.items()]) > 0.5)
        if isinstance(primary, float):
            # float is returned as weighted mean
            return sum([k * v for k, v in entries.items()])
        if isinstance(primary, int):
            # int is also a weighted mean
            return int(sum([k * v for k, v in entries.items()]))
        if isinstance(primary, str) or isinstance(primary, list):
            # str and list are sampled randomly based on weights
            return np.random.default_rng().choice(list(entries.keys()), p=list(entries.values()))
        raise NotImplementedError(f"did not implement merging for type {type(primary)}")
