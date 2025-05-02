# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import os
import platform
import shutil
import sys
import warnings
from pathlib import Path
from typing import List

import cv2
import hydra
import lightning  # noqa
import matplotlib
import optuna
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from omegaconf import DictConfig, OmegaConf

import mml
import mml.core.scripts.utils as script_utils  # we need to import the module to be able to monkeypatch in tests
from mml.core.scripts.exceptions import MMLMisconfigurationException
from mml.core.scripts.notifier import BaseNotifier
from mml.core.visualization.logo import show_logo

# prevent lightning to add console logger and prevent propagation
logging.getLogger().addHandler(logging.NullHandler())
pl_logger = logging.getLogger("lightning")
if pl_logger.handlers:
    pl_logger.removeHandler(pl_logger.handlers[0])
pl_logger.setLevel(logging.NOTSET)

logger = logging.getLogger("mml")


def wrapped_mml() -> float:
    """
    Wraps the mml main function with environmental loading. So first sets constants of third party libraries and loads
    MML env file as well as potential plugins.

    :return: forwards the return value of the scheduler, see :meth:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.run`
    """
    # ╔════════════════════════════╗
    # ║ ENVIRONMENT INITIALIZATION ║
    # ╚════════════════════════════╝
    # trade in precision for some speedup, see
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("high")
    # choose always non-interactive backend
    matplotlib.use("Agg")
    # Limit the number of threads when using OpenCV to avoid CPU bottlenecks on shared hardware
    cv2.setNumThreads(1)
    # Limit the number of threads when using pytorch to avoid CPU bottlenecks on shared hardware
    torch.set_num_threads(1)
    # load environment variables, must be done before decorating hydra
    script_utils.load_env()
    # load mml.plugins, this allows for added schedulers, tasks, ...
    script_utils.load_mml_plugins()
    # formatting of warnings, omit the newline and source code line in message
    formatwarning_orig = warnings.formatwarning
    warnings.formatwarning = lambda message, category, filename, lineno, line=None: formatwarning_orig(
        message, category, filename, lineno, line=""
    ).strip()
    # load the correct root folder of the configs which is determined as variable in the env file (if not using default
    # config folder from within mml)
    if os.environ["MML_CONFIGS_PATH"] == "DEFAULT_CONF_PATH":
        rel_config_path = "configs"
    else:
        from hydra.core.config_search_path import ConfigSearchPath
        from hydra.core.plugins import Plugins
        from hydra.plugins.search_path_plugin import SearchPathPlugin

        class MMLCoreConfigsSearchPathPlugin(SearchPathPlugin):
            def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
                # Sets the search path for mml with copied config files
                search_path.append(provider="mml-conf-copy", path=f"file://{os.environ['MML_CONFIGS_PATH']}")

        Plugins.instance().register(MMLCoreConfigsSearchPathPlugin)
        rel_config_path = None

    # ╔═══════════════════╗
    # ║ SPECIAL CLI CASES ║
    # ╚═══════════════════╝

    # catch corner case of --version call (must be done before running hydra)
    if "--version" in sys.argv:
        print(f"mml-core {mml.__version__}")
        print(" -------------")
        for plugin, version in sorted(mml.core.scripts.utils.MML_PLUGINS_LOADED.items(), key=lambda x: x[0]):
            print(f"{plugin} {version}")
        sys.exit()

    # catch corner case of plain mml call without mode or other information
    if len(sys.argv) == 1:
        show_logo(indent=8)
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║ For usage details you may                                 ║")
        print("║     * visit the docs (https://mml.readthedocs.io)         ║")
        print('║     * call "mml --help"                                   ║')
        print("╚═══════════════════════════════════════════════════════════╝")
        if os.getenv("MML_ENV_PATH", None):
            print(f"MML_ENV_PATH has been set to {os.getenv('MML_ENV_PATH')}")
        else:
            print("MML_ENV_PATH has not been set")
        print('For a list of installed plugins call "mml --version".')
        sys.exit()

    # argv modification making "mml anything ARGS" -> "mml mode=anything ARGS"
    if "mode=" in sys.argv[1]:
        print('\nWARNING:\nYou may drop the "mode=" string as long as you place the value as first mml argument.\n\n')
    elif len(sys.argv) >= 2 and any(arg.startswith("mode=") for arg in sys.argv[2:]):
        print(
            "\nWARNING:\nProviding mode as non first argument is deprecated and the api might remove support in "
            'the future. Please run for example as "mml info OVERRIDES" or "mml train OVERRIDES"\n\n'
        )
    elif sys.argv[1].startswith("--") or sys.argv[1].startswith("-"):
        # catch other corner cases e.g. mml --help, we do not want to raise awareness in these cases
        pass
    elif "=" in sys.argv[1]:
        # options are given, but no mode has been provided
        print(
            "\nWARNING:\nYou did not provide a mode to mml. Will fallback to info mode. Please use "
            '"mml info KWARGS" instead.\n\n'
        )
    else:
        # now we can expect the user to have intended that the argv[1] should be extended with mode=
        sys.argv[1] = "mode=" + sys.argv[1]

    # ╔══════════════════════════════════╗
    # ║ INVOKE HYDRA FOR CFG COMPILATION ║
    # ╚══════════════════════════════════╝

    # define main function, wrapped by hydra, which loads the config files and handles overwrites
    @hydra.main(version_base=None, config_path=rel_config_path, config_name=os.environ["MML_CONFIG_NAME"])
    def mml_main(cfg: DictConfig) -> float:
        """
        Hydra-wrapped main function. This function will be repeated in "multirun" scenarios. It's return value will
        be used by hydra sweepers (e.g. in hyperparameter optimization scenarios). See
        `hydra multirun <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run>`_ and
        `hydra sweepers <https://hydra.cc/docs/plugins/optuna_sweeper/>`_ for more on this.

        :param cfg: the omegaconf configuration compiled by hydra
        :return: the return value of the scheduler, see :meth:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.run`
        """
        # ╔════════════════════╗
        # ║ MML INITIALIZATION ║
        # ╚════════════════════╝
        if cfg["use_best_params"] and cfg["continue"]:
            raise MMLMisconfigurationException("use_best_params and continue functionality do not work together!")
        # reuse optimized hyperparameters directly
        if cfg["use_best_params"]:
            # indicates to use existing hyperparameter optimization results, first try to load a summary yaml
            hpo_path = Path(cfg["proj_path"]) / "hpo" / cfg["use_best_params"] / "optimization_results.yaml"
            if hpo_path.exists():
                logger.debug(f"Found existing hpo results at {hpo_path}. Will load.")
                hpo_results = OmegaConf.load(hpo_path)
                best_params = hpo_results["best_params"]
            else:
                # summary yaml not found, study might not have finished, check whether a persistent storage exists
                try:
                    loaded_study = optuna.load_study(study_name=cfg["use_best_params"], storage=cfg.hpo.storage)
                except KeyError:
                    raise MMLMisconfigurationException(
                        "use_best_params was not able to load study. Here are some potential reasons this might have "
                        "failed:\n - study was interrupted, without persistent storage (see e.g. mml-sql) no "
                        "intermediate results are kept alive\n - without persistent storage no cross-project loading is "
                        "supported for now, is proj set correctly?\n - in case you tried a persistent storage, have you "
                        "configured hpo.storage correctly AND provided it to this call?\n - also consider typos in your "
                        "arg, you need either to name the exact optuna study OR the hpo folder of THIS project"
                    )
                best_params = loaded_study.best_params
            # re-compose the config, we provide the optimized overrides first and then add the ones directly provided
            # this ensures the latter outrule the former and allow modification of optimized param results
            provided_overrides = OmegaConf.to_container(HydraConfig.get().overrides.task)
            overrides = [key + "=" + val for key, val in best_params.items()] + provided_overrides
            updated_cfg = hydra.compose(
                config_name=os.environ["MML_CONFIG_NAME"], overrides=overrides, return_hydra_config=False
            )
            cfg = updated_cfg
            logger.info("-------------------------------------------------------------------------------------")
            logger.info(
                f"Loaded hpo results from study {cfg['use_best_params']} and merged {len(best_params)} params into config."
            )
            for k, v in best_params.items():
                logger.info(f"  > {k}={v}")
            logger.info("-------------------------------------------------------------------------------------")

        # implements the continue functionality, despite the continue flag and project all other config
        # specifications will be disregarded and instead overwritten by the loaded config
        if cfg["continue"]:
            if "hpo" in Path(os.getcwd()).parts:
                raise ValueError("cannot continue in hyper-parameter-optimization (--multirun) mode")
            # delete already created run folder
            exp_path = Path(os.getcwd())
            single_day_event = len([p for p in exp_path.parent.iterdir() if p.is_dir()]) == 1
            del_path = exp_path.parent if single_day_event else exp_path
            shutil.rmtree(del_path)
            # if continuing latest detect that folder
            if cfg["continue"] == "latest":
                day_path = sorted([date for date in (Path(cfg.proj_path) / "runs").iterdir() if date.is_dir()])[-1]
                time_path = sorted([time for time in day_path.iterdir() if time.is_dir()])[-1]
                cfg["continue"] = day_path.name + "/" + time_path.name
            else:
                selected_path = Path(cfg["proj_path"]) / "runs" / cfg["continue"]
                if not selected_path.exists():
                    msg = f"Chosen value <continue={cfg['continue']}> results in not existent path {selected_path}."
                    logger.error(msg)
                    raise ValueError(msg)
            # change cwd to selected run
            os.chdir(str(Path(cfg["proj_path"]) / "runs" / cfg["continue"]))
            # load stored cfg (but keep continue variable!)
            cfg_path = Path(os.getcwd()) / ".hydra" / "config.yaml"
            assert cfg_path.exists()
            old_cfg = cfg.copy()
            cfg = OmegaConf.load(cfg_path)
            cfg["continue"] = old_cfg["continue"]
            cfg["data_dir"] = old_cfg["data_dir"]
            cfg["out_dir"] = old_cfg["out_dir"]
            cfg["num_workers"] = old_cfg["num_workers"]
            hydra_cfg = HydraConfig.get()
            # this is the NEW hydra config, interestingly this allows to enable verbose mode afterward
            configure_log(DictConfig(hydra_cfg.job_logging), hydra_cfg.verbose)
            logger.info("-------------------------------------------------------------------------------------")
            logger.info(f"Continuing from {os.getcwd()}.")
            logger.info("-------------------------------------------------------------------------------------")
        # resolve type
        cfg["num_workers"] = int(cfg["num_workers"])
        # activate warnings logging (if configured)
        logging.captureWarnings(cfg.logging.capture_warnings)
        try:
            hydra_cfg = HydraConfig.get()
            choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
            mode = choices["mode"]
        except ValueError:
            mode = "unknown"
        logger.info(
            f"Started MML {mml.__version__} on Python {platform.python_version()} with mode {str(mode).upper()}."
        )
        logger.info(f"Plugins loaded: {list(mml.core.scripts.utils.MML_PLUGINS_LOADED.keys())}")
        # instantiate notifiers
        all_notifiers: List[BaseNotifier] = [
            hydra.utils.instantiate(cfg.logging.notifier[elem]) for elem in cfg.logging.notifier if elem != "dummy"
        ]
        logger.debug(f"Instantiated {len(all_notifiers)} notifiers.")
        for notifier in all_notifiers:
            notifier.notify_on_start()
        with script_utils.catch_time() as init_timer:
            try:
                scheduler = hydra.utils.instantiate(cfg.mode.scheduler, cfg)
            except Exception as e:
                logger.exception("MML failed during initialization!")
                if isinstance(e, KeyboardInterrupt):
                    logger.info(f"Omitted notification for class {e.__class__}.")
                else:
                    for notifier in all_notifiers:
                        notifier.notify_on_failure(error=e)
                raise e
        logger.info(f"MML init time was {init_timer.pretty_time}.")
        # ╔═════════════╗
        # ║ MML RUNTIME ║
        # ╚═════════════╝
        # start scheduler
        with script_utils.catch_time() as run_timer:
            try:
                try:
                    val = scheduler.run()
                except OSError as e:
                    # Exception.add_note() is only available from python 3.11 so we have to raise a nested error
                    if isinstance(e, OSError) and "Too many open files:" in str(e):
                        raise OSError(
                            "Too many open files! Try increasing your open file limit (check 'ulimit -n' on "
                            "UNIX systems)."
                        ) from e
                    else:
                        raise
            except Exception as e:
                logger.exception("MML failed during runtime!")
                if isinstance(e, KeyboardInterrupt) or isinstance(e, InterruptedError):
                    logger.info(f"Omitted notification for class {e.__class__}.")
                else:
                    for notifier in all_notifiers:
                        notifier.notify_on_failure(error=e)
                raise e
            finally:
                if scheduler.lock_path.exists():
                    scheduler.lock_path.unlink()
                    logger.debug(f"Deleted run path lock at {scheduler.lock_path}.")
                else:
                    warnings.warn(UserWarning(f"No lock path found at {scheduler.lock_path}."))
        logger.info(f"MML run time was {run_timer.pretty_time}.")
        # check and report return value
        if val is not None:
            # optuna has issues with inf value to store in SQL database, replace with very large float if necessary
            if val == float("inf"):
                response = float("1.e5")
                logger.error(f"Was returned {val}, will convert to {response} to ensure optuna usage.")
                val = response
            # log final return value of the scheduler
            return_val_path = Path(os.getcwd()) / "return_val.txt"
            with open(return_val_path, "w") as f:
                f.write(str(val))
            logger.info(f"Return value is {val}.")
        for notifier in all_notifiers:
            notifier.notify_on_end(return_value=val)
        return val

    # run main
    return mml_main()


if __name__ == "__main__":
    """
    This code part will be triggered if calling 'python -m mml' instead of cli interface 'mml'. Compare with 'cli.py'.
    """
    wrapped_mml()
