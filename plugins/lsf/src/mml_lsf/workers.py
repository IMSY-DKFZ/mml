# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import os
import socket
import warnings

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from mml.core.scripts.schedulers.base_scheduler import AbstractBaseScheduler

logger = logging.getLogger(__name__)

DEFAULT_CPU_CAP = 12


def get_allowed_n_proc():
    """
    Taken from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/default_n_proc_DA.py

    This function is used to find the max number of processes allowed on different Systems. It is specific to
    our cluster infrastructure at DKFZ.

    Interpret the output as the number of processes used for data augmentation PER GPU.
    The way it is implemented here is simply a look-up table. We know the hostnames, CPU and GPU configurations of our
    systems and set the numbers accordingly. For example, a system with 4 GPUs and 48 threads can use 12 threads per
    GPU without overloading the CPU (technically 11 because we have a main process as well), so that's what we use.
    """
    hostname = socket.gethostname()
    logger.info(f"Hostname: {hostname}.")
    if hostname in ["hdf19-gpu16", "hdf19-gpu17", "hdf19-gpu18", "hdf19-gpu19", "e230-AMDworkstation"]:
        use_this = 16
    elif hostname.startswith("e230-dgx1"):
        use_this = 10
    elif hostname.startswith("hdf18-gpu") or hostname.startswith("e132-comp"):
        use_this = 16
    elif hostname.startswith("e230-dgx2"):
        use_this = 6
    elif hostname.startswith("e230-dgxa100-"):
        use_this = 28
    elif hostname.startswith("lsf22-gpu"):
        use_this = 28
    elif hostname.startswith("hdf19-gpu") or hostname.startswith("e071-gpu"):
        use_this = 12
    else:
        use_this = DEFAULT_CPU_CAP

    use_this = min(use_this, os.cpu_count())
    return use_this


def check_lsf_workers(scheduler: AbstractBaseScheduler) -> None:
    """
    Looks up the available workers for DKF LSF cluster nodes and sets accordingly.

    :param scheduler: the scheduler the hook runs upon
    :return:
    """
    if scheduler.continue_status:
        warnings.warn(
            f"LSF cluster plugin detected CONTINUE run. This means system check will be skipped and CPUs will "
            f"be capped by at least {DEFAULT_CPU_CAP}."
        )
    else:
        try:
            hydra_cfg = HydraConfig.get()
        except ValueError:
            hydra_cfg = None
        if hydra_cfg:
            choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
            # if any(part.startswith('num_workers=') for part in hydra_cfg.overrides.task):
            #     logger.info('LSF cluster plugin detected CLI override for num_workers.')
            if choices["sys"] == "local":
                logger.info("LSF cluster plugin detected local system, no changes made to the number of workers.")
                return
            logger.info(f"LSF cluster plugin detected system: {choices['sys']}.")
    configured = scheduler.cfg.num_workers
    allowed = get_allowed_n_proc()
    if configured > allowed:
        scheduler.cfg.num_workers = allowed
        logger.info(f"LSF cluster plugin set CPU workers to {allowed} (previously: {configured}).")
    else:
        logger.info(f"No need to update CPU workers. Will stick to {configured}.")
