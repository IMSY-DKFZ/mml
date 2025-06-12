# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import abc
import copy
import datetime
import logging
import os
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import lightning
import torch
from colorama import Back, Fore, Style
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import DictConfig, OmegaConf

from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.data_loading.lightning_datamodule import MultiTaskDataModule
from mml.core.data_loading.task_attributes import Modality
from mml.core.data_loading.task_struct import TaskStruct, TaskStructFactory
from mml.core.models.lightning_single_frame import SingleFrameLightningModule
from mml.core.scripts.callbacks import (
    MetricsTrackerCallback,
    MMLModelCheckpoint,
    MMLRichProgressBar,
    MMLTQDMProgressBar,
    StopAfterKeyboardInterrupt,
)
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.utils import ARG_SEP, TAG_SEP, catch_time, throttle_logging

logger = logging.getLogger(__name__)


class AbstractBaseScheduler(metaclass=abc.ABCMeta):
    """
    This is the base class of a scheduler for a possible series of experiments. Based on a special order of routines
    one can implement a derived scheduler class for an own setup. The scheduler itself keeps track of the status,
    datasets, manages file savings and loading, provides routines for the inclusion of dataloaders & models.
    """

    def __init__(self, cfg: DictConfig, available_subroutines: List[str]):
        """
        Creates the schedule. Can be started afterward with the .run() method.

        :param cfg: configs of the current run
        :param available_subroutines: available subroutines of inherited scheduler
        """
        self.cfg = cfg
        logger.debug("Creating schedule...")
        subroutines = list(self.cfg.mode.subroutines)
        if not isinstance(subroutines, list):
            raise TypeError(
                f"Please hand in subroutines for the scheduler as a list type. You gave type {type(subroutines)}."
            )
        if len(subroutines) == 0:
            raise ValueError("Please hand in non-empty subroutines list for the scheduler.")
        if not set(subroutines).issubset(set(available_subroutines)):
            raise MMLMisconfigurationException(
                f"Allowed subroutines for Scheduler are only {available_subroutines}, but gave {subroutines}."
            )
        self.subroutines = subroutines
        # the active step naming variable will always be updated to the actual scheduler step and used for logging
        self.active_step_naming = "init"
        # try to acquire running lock
        self.lock_path = Path(os.getcwd()) / "lock.tmp"
        if self.lock_path.exists():
            msg = (
                f"Was not able to acquire the lock {self.lock_path}. This might be due to either a currently running "
                f"instance on that path or some ungraceful disruption on that run. If you want to continue exactly "
                f"this experiment run, make sure to avoid running conditions with any other scheduler on that "
                f"folder and manually delete the lock file ({self.lock_path}). If you do not insist on this specific "
                f"run folder just start MML again with the same config options as before (a new run folder will "
                f"be created automatically)."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        self.lock_path.touch(exist_ok=False)
        # file management and continue status (if continue the internal logs will
        self.continue_status = bool(self.cfg["continue"])
        # be aware to call this FileManager before any other file related classes (it is a singleton class)
        if MMLFileManager.exists():
            warnings.warn(
                "MMLFileManager was not created by BaseScheduler, but existed previously. In case of "
                "running multiple schedulers, make sure to correctly tear down the file manager ("
                "usually during finish_exp) by calling clear_instance() on the file manager."
            )
        self.fm = MMLFileManager.instance(
            proj_path=Path(self.cfg["proj_path"]),
            data_path=Path(self.cfg["data_dir"]),
            log_path=Path(os.getcwd()),
            reuse_cfg=self.cfg.reuse,
            remove_cfg=self.cfg.remove,
        )
        # the return value will be returned by the 'run' method, allowing for blackbox optimisation, it is recommended
        # to set this in the subroutine finishing instructions
        self.return_value = None
        # apply tagging.all and tagging.variants to tasks:
        if self.cfg.tagging.all:
            if not self.cfg.tagging.all.startswith(TAG_SEP):
                raise MMLMisconfigurationException(
                    f'tagging.all="{self.cfg.tagging.all}" does not start with "{TAG_SEP}".'
                )
            self.cfg.task_list = [task + self.cfg.tagging.all for task in self.cfg.task_list]
            logger.debug(f"Tagged all tasks with {self.cfg.tagging.all}.")
        if self.cfg.tagging.variants:
            all_tasks = []
            for variant in self.cfg.tagging.variants:
                if not variant.startswith(TAG_SEP):
                    raise MMLMisconfigurationException(
                        f'tagging.variants entry "{variant}" does not start with "{TAG_SEP}".'
                    )
                # identity tag does not need to be fed forward
                if variant == f"{TAG_SEP}identity":
                    all_tasks.extend(self.cfg.task_list)
                    continue
                all_tasks.extend([task + variant for task in self.cfg.task_list])
                logger.debug(f"Created task variant {variant} for all tasks.")
            self.cfg.task_list = all_tasks
        # check if tasks contain duplicates and guarantee different namings of tasks
        tmp_tasklist = self.cfg.task_list.copy()
        if len(set(tmp_tasklist)) < len(tmp_tasklist):
            mod_dic = {x: 0 for x in tmp_tasklist}
            for ix in range(len(tmp_tasklist)):
                mod_dic[tmp_tasklist[ix]] += 1
                if tmp_tasklist.count(tmp_tasklist[ix]) > 1:
                    tmp_tasklist[ix] += f"{TAG_SEP}duplicate{ARG_SEP}" + str(mod_dic[tmp_tasklist[ix]])
            self.cfg.task_list = tmp_tasklist
            logger.info(f"Found {sum(mod_dic.values())} duplicates in task list and modified their names.")
        # setting of pivot dataset
        self.pivot = self.cfg.pivot.name
        if self.pivot:
            # handle pivot specific tags
            new_name = (self.pivot + self.cfg.pivot.tags).strip()
            # replace tags in tasks / add to tasks
            if self.pivot not in self.cfg.task_list:
                self.cfg.task_list.append(new_name)
                logger.info(f"Added pivot task {new_name} to task_list.")
            else:
                warnings.warn(
                    "Pivot has also been found in task_list, this avoids any tagging.all and "
                    "tagging.variants configuration. But it applies pivot.tags."
                )
                self.cfg.task_list[self.cfg.task_list.index(self.pivot)] = new_name
            self.pivot = new_name
            logger.info("Pivot task is " + self.highlight_text(self.pivot) + ".")

        for task in self.cfg.task_list:
            if " " in task:
                raise MMLMisconfigurationException(
                    f"Tagging syntax has changed. Avoid whitespace inside tags and use "
                    f"{TAG_SEP} to separate tags, as well as {ARG_SEP} to seperate "
                    f"arguments, e.g. task_name{TAG_SEP}tag1{TAG_SEP}tag2{ARG_SEP}"
                    f"arg1oftag2{ARG_SEP}arg2oftag2{TAG_SEP}tag3."
                )
        # create TaskStructFactory
        self.task_factory = TaskStructFactory(self.cfg, load=False)

        # managing the scheduler
        self.commands: List[Callable[[...], None]] = []
        self.params: List[List[...]] = []
        self.planned_schedule = self.fm.log_path / "scheduler_plan.txt"
        self.status_log = self.fm.log_path / "scheduler_log.txt"
        # create commands and params
        # -- prepare experiment
        self.commands.append(self.prepare_exp)
        self.params.append([])
        # -- scheduler specific commands
        self.create_routine()
        # -- finish experiment
        self.commands.append(self.finish_exp)
        self.params.append([])

        if len(self.commands) != len(self.params):
            raise RuntimeError(
                "Commands and Params length do not match in schedule creation. Please check your "
                "create_routine implementation."
            )
        # create string version of schedule
        coms = [command.__name__ for command in self.commands]
        pars = [str(param) for param in self.params]
        schedule_lines = ["method: " + coms[ix] + " / " + pars[ix] + "\n" for ix in range(len(coms))]

        # if not continue - give warning and overwrite old scheduler plan and add to log (with marker)
        if not self.continue_status:
            # write out schedule
            with open(self.planned_schedule, "w") as file:
                file.writelines(schedule_lines)
            # append to status log
            with open(self.status_log, "a") as file:
                file.writelines(
                    [
                        "HEADER\n",
                        "Timepoint of beginning\n",
                        datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S") + "\n",
                        "START\n",
                    ]
                )
        else:
            # this is "continue" mode, first check if there has been a previous run of the experiment
            if not self.planned_schedule.exists():
                raise FileNotFoundError(
                    f"Did not find any planned schedule (should be at {self.planned_schedule}). "
                    f"Has this run finished already?"
                )
            # load previous schedule
            with open(self.planned_schedule, "r") as file:
                previous_lines = file.readlines()
            # compare schedules - first the lengths
            if len(previous_lines) != len(schedule_lines):
                msg = (
                    f"Continue mode failed: Old schedule has length {len(previous_lines)} but actual settings "
                    f"require schedule of length {len(schedule_lines)}."
                )
                logger.error(msg)
                raise ValueError(msg)
            # next compare content
            unmatching = [
                self.compare_schedule_entries(previous_lines[ix], schedule_lines[ix])
                for ix in range(len(previous_lines))
            ]
            if any(unmatching):
                dif_ix = unmatching.index(True)
                msg = (
                    f"Content of previous schedule and actual schedule differ at {unmatching.count(True)} places. "
                    f"First difference is {previous_lines[dif_ix]} (previous) versus {schedule_lines[dif_ix]} "
                    f"(now) in line {dif_ix}."
                )
                logger.error(msg)
                raise ValueError(msg)
            # we will not need previous schedule anymore
            del previous_lines
            logger.info("Previously canceled schedule matches current one!")
            # schedules seem to match, find correct position in schedule, start with loading status log
            with open(self.status_log, "r") as file:
                status_lines = file.readlines()
            status_lines = [line.strip() for line in status_lines]
            # calculate already processed steps
            counter = 0
            runtime_counter = 1
            for ix in range(1, len(status_lines)):
                if "method:" == status_lines[ix][:7]:
                    counter += 1
                elif "CONTINUE" == status_lines[ix][:8]:
                    # if already (successfully) continued, be aware that the initial experiment preparation is added
                    if len(status_lines) > ix + 1:
                        runtime_counter += 1
            logger.info(
                f"Evaluated existing previous runs. Found {runtime_counter} previous runs and {counter}/"
                f"{len(self.commands)} commands completed so far."
            )
            # skip already executed commands (and corresponding params)
            self.commands = self.commands[counter:]
            self.params = self.params[counter:]
            # add preparation at the beginning of the experiment
            self.commands = [self.prepare_exp] + self.commands
            self.params = [[]] + self.params
            # append continuation to status log
            with open(self.status_log, "a") as file:
                file.writelines(
                    [
                        "HEADER\n",
                        "Timepoint of continuation\n",
                        datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S") + "\n",
                        "CONTINUE\n",
                    ]
                )
        # hold callback references
        self.metrics_callback: Optional[MetricsTrackerCallback] = None
        self.checkpoint_callback: Optional[ModelCheckpoint] = None
        # finalize initialisation
        self._run_after_init_hooks()
        self._run_checks()
        logger.debug("Finished initialization of scheduler...")

    def _run_after_init_hooks(self):
        """
        Runs some global hooks. These can be set by plugins to modify default behaviour of any scheduler.

        .. code-block:: python

            from mml.core.script.base_scheduler import AFTER_SCHEDULER_INIT_HOOKS

            def my_hook(scheduler: AbstractBaseScheduler) -> None:
                print(scheduler.cfg)

            AFTER_SCHEDULER_INIT_HOOKS.append(my_hook)

        :return:
        """
        for hook in AFTER_SCHEDULER_INIT_HOOKS:
            logger.info(f"Executing after init hook: {hook.__name__}")
            hook(self)

    def _run_checks(self):
        """
        This is where some basic checks are made if configs / setup make sense.
        :return:
        """
        # check if preprocessing id is set correctly (only necessary if started via hydra)
        try:
            hydra_cfg = HydraConfig.get()
        except ValueError:
            hydra_cfg = None
        # check preprocessing ID (only available if not in continue mode as runtime choices may deviate)
        if hydra_cfg and not self.continue_status:
            choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
            if Path(choices["preprocessing"]).stem != self.cfg.preprocessing.id:
                raise MMLMisconfigurationException(
                    f"Preprocessing config id {self.cfg.preprocessing.id} does not match"
                    f" config file name {choices['preprocessing']}!"
                )
        # check if preprocessing pipeline matches
        if self.cfg.preprocessing.id != "none":
            storage_definition_path = self.fm.get_pp_definition(preprocessing=self.cfg.preprocessing.id)
            if storage_definition_path.exists():
                storage_pipeline = OmegaConf.load(storage_definition_path)
                if storage_pipeline != self.cfg.preprocessing.pipeline:
                    raise MMLMisconfigurationException(
                        f"Found a missmatch in preprocessing configurations.\n"
                        f"Preprocessing ID is : {self.cfg.preprocessing.id}.\n"
                        f"Existing preprocessing folder defines this pipeline as:\n"
                        f"{storage_pipeline}\n"
                        f"Current preprocessing config defines pipeline as:"
                        f"{self.cfg.preprocessing.pipeline}."
                    )
        # ensure torch.compile is not used in conjunction witch learning rate tuning
        if self.cfg.tune.lr and self.cfg.compile.enable:
            raise MMLMisconfigurationException(
                f"Tune lr {self.cfg.tune.lr} currently not supported with compile enable"
                f" {self.cfg.compile.enable} due to torch compile checkpointing issue."
                f" To be  resolved in a future version!"
            )

    def set_active_naming(self, command_ix) -> None:
        """
        Defines the active_step_naming attribute for the given command index.

        :param command_ix: index of the command
        :return: None
        """
        prefix = self.commands[command_ix].__name__ + "--" + "_".join([str(param) for param in self.params[command_ix]])
        suffix = "_" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        if self.continue_status and self.fm.checkpoint_path.exists():
            # in case of continuing we need to reuse the old timestamp (for loading last model checkpoint)
            previous_active_steps = sorted([p for p in self.fm.checkpoint_path.iterdir() if p.name.startswith(prefix)])
            if len(previous_active_steps) > 0 and (previous_active_steps[-1] / "last.ckpt").exists():
                suffix = "_" + previous_active_steps[-1].name.split("_")[-1]
                assert (self.fm.checkpoint_path / (prefix + suffix)).exists()
            else:
                warnings.warn(f"Not successful to find a model checkpoint at {self.fm.checkpoint_path} with {prefix=}.")
        self.active_step_naming = prefix + suffix

    # next there are some virtual methods that must/may be overwritten
    @abc.abstractmethod
    def create_routine(self) -> None:
        """
        Adds commands and parameters to the schedule. May e.g. be in the form of:

        .. code-block:: python

            if 'xyz' in self.subroutines:
                for task in self.cfg.task_list:
                    self.commands.append(self.MY_IMPLEMENTED_ROUTINE)
                    self.params.append([task])

        :return: None
        """
        pass

    def after_preparation_hook(self) -> None:
        """
        This hook will be called at the end of the :meth:`prepare_exp` step. That step basically prepares the task
        structs accordingly. Once this is done, this hook can be used for remaining setups necessary that rely on
        task structs (e.g. compatibility checks).

        In contrast to the other scheduler steps in the schedule, the :meth:`prepare_exp` is also performed in
        CONTINUE mode. This hook can be used to ensure continue capability of the scheduler, e.g., by loading
        additional ressources from previous runs. More on CONTINUE mode: :ref:`continue-option`

        Example usages:

         * :meth:`~mml.core.scripts.schedulers.postprocess_scheduler.PostprocessScheduler.after_preparation_hook`
         * :meth:`~mml.core.scripts.schedulers.preprocess_scheduler.PreprocessScheduler.after_preparation_hook`
         * :meth:`~mml_similarity.scripts.abstract_task_distance_scheduler.AbstractTaskDistanceScheduler.after_preparation_hook`
        """
        pass

    def before_finishing_hook(self):
        """
        This hook will be called at the beginning of the :meth:`finish_exp` step. That step performs dumping and clean
        up of intermediates. As such this hook is the perfect opportunity to aggregate results or other intermediates
        from the previous computing steps (e.g., some results plotting).

        It is also ideally placed to set the :attr:`return_value` of the scheduler, which will be returned by the
        :meth:`run` method and can be used for experiment evaluation and in hyperparameter optimization (see
        :doc:`/hpo`).

        Example usages:

         * :meth:`~mml.core.scripts.schedulers.postprocess_scheduler.PostprocessScheduler.after_preparation_hook`
         * :meth:`~mml.core.scripts.schedulers.preprocess_scheduler.PreprocessScheduler.after_preparation_hook`
         * :meth:`~mml_similarity.scripts.abstract_task_distance_scheduler.AbstractTaskDistanceScheduler.after_preparation_hook`
        """
        pass

    # main routine
    def run(self) -> float:
        """
        The run routine starts the schedule and logs the process (within a file at self.status_log).

        :return: self.return_value (which might be set during runtime)
        """
        for ix, command in enumerate(self.commands):
            # undo continue status after preparation
            if self.continue_status and ix >= 2:
                self.continue_status = False
            # set active_step_naming (used for logging)
            self.set_active_naming(ix)
            # seed the step
            if self.cfg.seed:
                with throttle_logging(logging.INFO):
                    seed_everything(self.cfg.seed, workers=True)
                logger.debug(f"Random seeding with seed {self.cfg.seed} performed.")
            # run the command with parameters
            logger.debug(
                f"Trying to run command ({ix + 1}/{len(self.commands)}): {command.__name__} with params: "
                f"{self.params[ix]}"
            )
            with catch_time() as timer:
                command(*self.params[ix])
            logger.debug(f"Command run successfully within {timer.pretty_time}.")
            # reset callback references
            self.metrics_callback = None
            self.checkpoint_callback = None
            # backup all current tasks
            if command.__name__ != "finish_exp":
                self.task_factory.dump(clear_container=False)
            # log after successful command (except for continued exp_prep command)
            if command.__name__ == "prepare_exp" and self.continue_status:
                logger.debug("Skipping status logging of continued experiment preparation!")
            else:
                with open(self.status_log, "a") as file:
                    file.write("method: " + command.__name__ + " / " + str(self.params[ix]) + "\n")
        return self.return_value

    def prepare_exp(self) -> None:
        """
        First command of any experiment. Mainly handles loading of task structs and seeding of experiment. Specific
        preparation might also be done with the >additional_preparation_instructions<. USE THAT INSTEAD AND DO NOT
        OVERWRITE THIS FUNCTION UNLESS YOU KNOW WHAT YOU DO.

        :return: None
        """
        # prepare TaskStructFactory (only if unloaded)
        assert len(self.task_factory.container) == 0, "TaskFactory should be empty prior to loading tasks!"
        if self.continue_status:
            logger.info("Trying to prepare loading to continue experiment...")
            # restore previous task structs
            self.task_factory.loading_old_dump()
            # assert compliance
            assert set([task.name for task in self.task_factory.container]) == set(self.cfg.task_list), (
                f"Loading of {len(self.cfg.task_list)} tasks failed. Inconsistent tasks from loading path {os.getcwd()}!"
            )
        else:
            logger.info("Preparing experiment ...")
            # create task struct for every task in task list
            for task in self.cfg.task_list:
                self.task_factory.create_task_struct(name=task, return_ref=False)
        # loading additional resources dependent on routines
        self.after_preparation_hook()
        logger.info("Starting experiment!")

    def finish_exp(self) -> None:
        """
        Last command of any experiment, this is how every experiment finishes. Ensures dumping of task factory,
        unlinks the planned schedule, removes intermediate results if specified in config and allows also for
        specific instructions of any subclass via the >additional_finishing_instructions< interface. USE THAT INSTEAD
        AND DO NOT OVERWRITE THIS FUNCTION UNLESS YOU KNOW WHAT YOU DO.

        :return: None
        """
        # call finishing instructions, e.g. plotting of results or deleting artifacts
        self.before_finishing_hook()
        # tear down the file manager
        self.fm.remove_intermediates()
        self.fm.clear_instance()
        # clear the planed schedule file
        self.planned_schedule.unlink()
        logger.info("Successfully finished all experiments!")

    def create_trainer(
        self, monitor: Optional[Tuple[str, str]] = None, metrics_callback: bool = False
    ) -> lightning.Trainer:
        """
        Creates a trainer from `cfg.trainer` with callbacks from `cfg.cbs`. By default,
        uses two :class:`~mml.core.scripts.callbacks.MMLModelCheckpoint` callbacks that behave as follows:

        * at least every 30 minutes a checkpoint is stored to ensure resume compatibility,
        * if monitor is given will keep the best model stored based thereof, regularly checking at the end of
          each epochs validation
        * if monitor is None only the very last epoch will be stored (besides the temporal check)

        The non-time based checkpoint may be accessed through :attr:`checkpoint_callback`.

        :param Optional[Tuple[str, str]] monitor: (optional) a tuple of metric name and mode (min or max) to be
            monitored by model checkpoint (saves best model) and early stopping callback (if activated in cfg)
        :param bool metrics_callback: (optional) if true creates and also a metric callback
        :return: trainer instance, the callbacks can be accessed through the scheduler attributes
            :attr:`metrics_callback` and :attr:`checkpoint_callback`
        :rtype: Union[Tuple[pl.Trainer, ModelCheckpoint], Tuple[pl.Trainer, ModelCheckpoint, MetricsTrackerCallback]]
        """
        cbs = []
        # if not monitoring a specific metric, only save last epoch
        cpt_cb = MMLModelCheckpoint(
            monitor=monitor[0] if monitor else None,
            dirpath=self.get_checkpoints_dir(),
            filename="epoch{epoch:02d}-val_loss{val/loss:.2f}",
            auto_insert_metric_name=False,
            save_last="link",
            mode=monitor[1] if monitor else "min",
            save_top_k=1,
            save_on_train_epoch_end=False,
            enable_version_counter=False,
        )
        if self.checkpoint_callback:
            logger.error(
                "Checkpoint callback already initiated! You will only be able to access the latest "
                "ModelCheckpoint through scheduler.checkpoint_callback!"
            )
        self.checkpoint_callback = cpt_cb
        cbs.append(cpt_cb)
        # always ensure storing every 30 minutes
        time_ckpt_cb = MMLModelCheckpoint(
            dirpath=self.get_checkpoints_dir(),
            filename="temp_backup",
            save_last="link",
            train_time_interval=datetime.timedelta(minutes=30),
            enable_version_counter=False,
        )
        cbs.append(time_ckpt_cb)
        # handle interruptions gracefully
        cbs.append(StopAfterKeyboardInterrupt())
        if (
            "enable_progress_bar" in self.cfg.trainer and self.cfg.trainer.enable_progress_bar
        ) or "enable_progress_bar" not in self.cfg.trainer:
            if self.cfg.logging.render.rich:
                cbs.append(MMLRichProgressBar())
            else:
                cbs.append(MMLTQDMProgressBar())
        if metrics_callback:
            if self.metrics_callback:
                logger.error(
                    "Metrics callback already initiated! You will only be able to access the latest "
                    "MetricsTrackerCallback through scheduler.metrics_callback!"
                )
            met_cb = MetricsTrackerCallback()
            self.metrics_callback = met_cb
            cbs.append(met_cb)
        for cb_id in self.cfg.cbs:
            cb_conf = self.cfg.cbs[cb_id]
            if "_target_" in cb_conf:
                logger.debug(f"Instantiating callback <{cb_conf._target_}>")
                cbs.append(instantiate(cb_conf))
            else:
                logger.error(f"Invalid callback configuration: <{cb_conf}> for callback {cb_id}.")
        if self.cfg.hpo.pruning:
            # TODO
            # this will be possible as soon as pruning is supported by hydra optuna sweeper, see
            # https://github.com/facebookresearch/hydra/issues/1954
            # from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
            # prun_cb = PyTorchLightningPruningCallback(trial=None, monitor='val/loss')
            # cbs.append(prun_cb)
            warnings.warn("Pruning not yet supported by optuna hpo.", UserWarning)
        if self.continue_status:
            resume = self.get_checkpoints_dir() / "last.ckpt"
            if not resume.exists():
                # seems like no checkpoint was saved
                resume = None
                logger.warning("Resuming from checkpoint not possible, since no training checkpoint was found!")
        else:
            resume = None
        # set up logging
        if "_target_" in self.cfg.logging.exp_logger:
            if "TensorBoardLogger" in self.cfg.logging.exp_logger["_target_"]:
                # set version for tensorboard logger
                exp_logger = instantiate(self.cfg.logging.exp_logger, version=self.active_step_naming)
            else:
                # else only instantiate
                exp_logger = instantiate(self.cfg.logging.exp_logger)
        else:
            # no exp logger specified
            exp_logger = None
        trainer = instantiate(self.cfg.trainer, logger=exp_logger, callbacks=cbs, enable_checkpointing=True)
        if resume:
            trainer.ckpt_path = resume
        return trainer

    def lightning_tune(
        self,
        trainer: lightning.Trainer,
        model: lightning.LightningModule,
        datamodule: Optional[lightning.LightningDataModule],
        train_dataloaders=None,
    ) -> None:
        """
        Tune a model / datamodule based on configs.tune setting.

        :param trainer: the lightning trainer
        :param model: the lightning model
        :param datamodule: the lightning datamodule
        :param train_dataloaders: alternative method to provide the data, set datamodule to None in this case
        :return: none, tuned values are stored inside model / datamodule
        """
        if self.continue_status and (self.get_checkpoints_dir() / "last.ckpt").exists():
            # this assumes that there is at least ONE checkpoint available, which has tuning results stored
            # if cancelling happened during first epoch we do not have this information
            logger.info("Tuning skipped for continue mode.")
            return
        if self.cfg.tune.lr or self.cfg.tune.bs:
            tuner = Tuner(trainer=trainer)
            # disable caching
            _old_caching = self.cfg.sampling.enable_caching
            self.cfg.sampling.enable_caching = False
            if _old_caching:
                logger.info("Caching disabled while tuning.")
            if self.cfg.tune.bs:
                logger.info("Starting batch size optimization.")
                tuner.scale_batch_size(
                    model=model, datamodule=datamodule, train_dataloaders=train_dataloaders, **self.cfg.tune.bs_kwargs
                )
            if self.cfg.tune.lr:
                logger.info("Starting learning rate optimization.")
                tuner.lr_find(
                    model=model, datamodule=datamodule, train_dataloaders=train_dataloaders, **self.cfg.tune.lr_kwargs
                )
            # restore caching state
            self.cfg.sampling.enable_caching = _old_caching

    def get_checkpoints_dir(self):
        """
        Path to store checkpoints currently.

        :return: Path to a folder to store training checkpoints
        """
        return self.fm.checkpoint_path / self.active_step_naming

    def create_model(
        self,
        task_structs: List[TaskStruct],
        task_weights: Optional[List[float]] = None,
        load_parameters: Optional[Path] = None,
    ) -> lightning.LightningModule:
        """
        Creates a pytorch lightning module.

        :param List[TaskStruct] task_structs: list of task structs to construct lightning module
        :param Optional[List[float]] task_weights: (optional) list of task weights to weigh loss
        :param Optional[Path] load_parameters: (optional) path to load model weights
        :return: LightningModule instance
        """
        if any([Modality.IMAGE not in struct.modalities for struct in task_structs]):
            raise NotImplementedError(
                f"For now mml-core only supports single frame modules. Support of {Modality.VIDEO_CLIP} is planned."
            )
        duplicate_structs = [copy.deepcopy(struct) for struct in task_structs]
        for struct in duplicate_structs:
            struct.models = []  # models might cause hparams saving issues with pytorch lightning
        model = SingleFrameLightningModule(
            task_structs=duplicate_structs, cfg=self.cfg, task_weights=task_weights, load_parameters=load_parameters
        )
        if self.cfg.compile.enable:
            model.model = torch.compile(model.model, **self.cfg.compile.kwargs)
        # deactivate strict loading for more compatibility
        model.strict_loading = False
        return model

    def get_struct(self, task_name: str) -> TaskStruct:
        """
        Convenience function to access a task struct.

        :param str task_name: name of the task
        :return: the corresponding task struct
        """
        return self.task_factory.get_by_name(task_name)

    def create_datamodule(
        self, task_structs: Union[TaskStruct, List[TaskStruct]], fold: int = 0
    ) -> MultiTaskDataModule:
        """
        Creates a pytorch lightning datamodule.

        :param Union[TaskStruct, List[TaskStruct]] task_structs: task struct(s) to create datamodule from
        :param int fold: fold to be used
        :return: datamodule instance
        """
        if isinstance(task_structs, TaskStruct):
            task_structs = [task_structs]
        return MultiTaskDataModule(task_structs=task_structs, cfg=self.cfg, fold=fold)

    def highlight_text(self, text: str) -> str:
        """
        Helper function in highlighting text within terminal. May be turned of by the logging.highlight_task_names
        config option.

        :param text: text to be highlighted
        :return: modified text if highlighting is active, else plain input text
        """
        if self.cfg.logging.highlight_text and not self.cfg.logging.render.rich:
            return Fore.YELLOW + Back.CYAN + Style.BRIGHT + text + Style.RESET_ALL
        else:
            return text

    @staticmethod
    def compare_schedule_entries(entry_1: str, entry_2: str) -> bool:
        """
        Helper function in comparsion of schedules.

        :param entry_1: line of a schedule (command and args)
        :param entry_2: line of a schedule (command and args)
        :return: true if lines are compatible, else false
        """
        pos_1 = entry_1.find("object at")
        pos_2 = entry_1.find(" / ")
        return entry_1[:pos_1] != entry_2[:pos_1] or entry_1[pos_2:] != entry_2[pos_2:]


# these hooks can be accessed by plugins to modify default scheduler behaviour
AFTER_SCHEDULER_INIT_HOOKS: List[Callable[[AbstractBaseScheduler], None]] = []
