# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px

from mml.core.data_loading.task_attributes import Keyword
from mml.core.data_loading.task_struct import TaskStruct
from mml.core.scripts.model_storage import ModelStorage


def plot_dimensions_versus_baseline(all_tasks: List[TaskStruct], store_path: Path) -> None:
    def load_dim(path: Path) -> int:
        with open(path, "r") as file:
            lines = file.readlines()
        assert len(lines) == 1
        return int(lines[0])

    def load_performance(models: List[ModelStorage]) -> float:
        metric = "val/Recall"
        return max([m.metrics[-1][metric] for m in models])

    domain_list = [
        Keyword.DERMATOSCOPY,
        Keyword.LARYNGOSCOPY,
        Keyword.GASTROSCOPY_COLONOSCOPY,
        Keyword.LAPAROSCOPY,
        Keyword.NATURAL_OBJECTS,
        Keyword.HANDWRITINGS,
        Keyword.CATARACT_SURGERY,
        Keyword.FUNDUS_PHOTOGRAPHY,
        Keyword.MRI_SCAN,
        Keyword.X_RAY,
        Keyword.CT_SCAN,
        Keyword.CLE,
        Keyword.CAPSULE_ENDOSCOPY,
        Keyword.ULTRASOUND,
    ]

    def get_domain(task: TaskStruct) -> Keyword:
        domain_candidates = [d for d in task.keywords if d in domain_list]
        # TODO this is a hacky solution only so far
        if task.name == "svhn":
            domain_candidates.append(Keyword.NATURAL_OBJECTS)
        if task.name == "mnist_digit_classification" or task.name == "emnist_digit_classification":
            domain_candidates.append(Keyword.HANDWRITINGS)
        if len(domain_candidates) != 1:
            raise RuntimeError(f"{task} with {domain_candidates}")
        return domain_candidates[0]

    task_infos = [
        {
            "name": task.name,
            "dimension": load_dim(task.paths["dimension"]),
            "performance": load_performance(task.models),
            "domain": get_domain(task).value,
        }
        for task in all_tasks
    ]
    df = pd.DataFrame(data=task_infos)
    fig = px.scatter(data_frame=df, x="dimension", y="performance", color="domain", hover_name="name", trendline="ols")
    fig.write_html(store_path)
