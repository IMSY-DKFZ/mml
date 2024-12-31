# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import dataclasses
from typing import List, Optional

from mml.interactive.planning import DefaultRequirements, JobPrefixRequirements


@dataclasses.dataclass
class LSFSubmissionRequirements(JobPrefixRequirements):
    """
    If starting an MML run on an LSF cluster system, use these requirements.
    """

    num_gpus: int
    vram_per_gpu: float  # in GB
    queue: str
    special_requirements: List[str] = dataclasses.field(default_factory=list)
    undesired_hosts: List[str] = dataclasses.field(default_factory=list)
    mail: Optional[str] = None
    interactive: bool = False
    script_name: Optional[str] = None  # if mml is called via a script use ./scriptname instead of mml CLI
    job_group: Optional[str] = None

    def get_prefix(self) -> str:
        parts = ["bsub"]
        for itm in self.special_requirements:
            parts.append(f'-R "{itm}"')
        for itm in self.undesired_hosts:
            parts.append(f"-R \"select[hname!='{itm}']\"")
        parts.append(f"-gpu num={self.num_gpus}:j_exclusive=yes:gmem={self.vram_per_gpu:.1f}G")
        parts.append(f"-q {self.queue}")
        if self.mail:
            parts.append(f'-u "{self.mail}" -B -N')
        if self.interactive:
            parts.append("-I")
        if self.job_group:
            parts.append(f"-g {self.job_group}")
        if self.script_name:
            parts.append(f"./{self.script_name}")
        else:
            parts.append(DefaultRequirements().get_prefix())
        return " ".join(parts)
