# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import numpy as np
import pandas as pd

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import create_creator_func, register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@article{LIU2022100512,
title = {DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge},
journal = {Patterns},
pages = {100512},
year = {2022},
issn = {2666-3899},
doi = {https://doi.org/10.1016/j.patter.2022.100512},
url = {https://www.sciencedirect.com/science/article/pii/S2666389922001040},
author = {Ruhan Liu and Xiangning Wang and Qiang Wu and Ling Dai and Xi Fang and Tao Yan and Jaemin Son and Shiqi Tang 
          and Jiang Li and Zijian Gao and Adrian Galdran and J.M. Poorneshwaran and Hao Liu and Jie Wang and Yerui Chen 
          and Prasanna Porwal and Gavin Siew {Wei Tan} and Xiaokang Yang and Chao Dai and Haitao Song and Mingang Chen 
          and Huating Li and Weiping Jia and Dinggang Shen and Bin Sheng and Ping Zhang},
keywords = {diabetic retinopathy, screening, deep learning, artificial intelligence, challenge, retinal image, 
            image quality analysis, ultra-widefield, fundus image},
}
"""  # noqa W291

dset_name = "deep_drid"

ALL_TASKS = ["quality", "dr_level", "clarity", "field", "artifact"]


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    instructions = f"""
                 Download link is not available. Please download the dataset by clicking on the Code -> Download ZIP 
                 functionality of Github manually at https://github.com/deepdrdoc/DeepDRiD. 
                 Once the download is complete place the downloaded folder 'DeepDRiD-master.zip' in
                 <MML_DATA_ROOT>/DOWNLOADS/{dset_name}
            """  # noqa W291
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.verify_pre_download(
        file_name="DeepDRiD-master.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


def create_deep_drid_subtask(task: str, dset_path: Path, alias: str) -> None:
    tags = [Keyword.MEDICAL, Keyword.FUNDUS_PHOTOGRAPHY, Keyword.EYE]
    if task == "dr_level":
        tags.append(Keyword.TISSUE_PATHOLOGY)
    elif task in ["quality", "clarity", "artifact"]:
        tags.append(Keyword.IMAGE_ARTEFACTS)
    creator = TaskCreator(
        dset_path=dset_path,
        name=alias,
        task_type=TaskType.CLASSIFICATION,
        desc="TThe aim of this challenge is to evaluate algorithms for automated fundus image "
        "quality estimation and grading of diabetic retinopathy.",
        ref=REFERENCE,
        url="https://isbi.deepdr.org/data.html",
        instr="download via https://github.com/deepdrdoc/DeepDRiD",
        lic=License.CC_BY_4_0,
        release="2021",
        keywords=tags,
    )
    if task == "quality":
        idx_to_class = {0: "insufficient", 1: "sufficient"}
        columns = ["Overall quality"]
    elif task == "dr_level":
        idx_to_class = {0: "no retinopathy", 1: "mild NPDR", 2: "moderate NPDR", 3: "severe NPDR", 4: "PDR"}
        columns = ["left_eye_DR_Level", "right_eye_DR_Level"]
    elif task == "clarity":
        idx_to_class = {1: "lev 1", 4: "lev 2", 6: "lev 3", 8: "lev 3 +", 10: "lev 3 ++"}
        columns = ["Clarity"]
    elif task == "field":
        idx_to_class = {
            1: "no od and mac",
            4: "either od or mac",
            6: "od and mac",
            8: "od and mac +",
            10: "od and mac ++",
        }
        columns = ["Field definition"]
    elif task == "artifact":
        idx_to_class = {
            0: "no artifacts",
            1: "aortic not affected",
            4: "mac not affected",
            6: "between forth and half",
            8: "between half and full",
            10: "entire pp",
        }
        columns = ["Artifact"]
    else:
        raise ValueError(f"Task {task} not valid!")

    # read in meta data
    train_df = pd.read_csv(
        dset_path
        / DataKind.TRAINING_DATA
        / "DeepDRiD-master"
        / "regular_fundus_images"
        / "regular-fundus-training"
        / "regular-fundus-training.csv"
    )
    test_df = pd.read_csv(
        dset_path
        / DataKind.TRAINING_DATA
        / "DeepDRiD-master"
        / "regular_fundus_images"
        / "regular-fundus-validation"
        / "regular-fundus-validation.csv"
    )
    # iterate over dfs
    train_iterator, test_iterator = [], []
    for iterator, df, path_dir in [
        (train_iterator, train_df, "regular-fundus-training"),
        (test_iterator, test_df, "regular-fundus-validation"),
    ]:
        for index, row in df.iterrows():
            vals = [row[column] for column in columns]
            if np.isnan(vals[0]):
                vals.pop(0)
            val = vals[0]
            assert not np.isnan(val)
            iterator.append(
                {
                    Modality.SAMPLE_ID: row["image_id"],
                    Modality.IMAGE: (
                        dset_path
                        / DataKind.TRAINING_DATA
                        / "DeepDRiD-master"
                        / "regular_fundus_images"
                        / path_dir
                        / "Images"
                        / str(row["patient_id"])
                        / f"{row['image_id']}.jpg"
                    ),
                    Modality.CLASS: int(val),
                }
            )
    creator.find_data(train_iterator=train_iterator, test_iterator=test_iterator, idx_to_class=idx_to_class)
    creator.split_folds(n_folds=5, ensure_balancing=True)
    creator.infer_stats()
    creator.push_and_test()


for task in ALL_TASKS:
    alias = f"{dset_name}_{task}"
    creator_func = create_creator_func(create_func=create_deep_drid_subtask, task=task, alias=alias)
    register_taskcreator(task_name=alias, dset_name=dset_name)(creator_func)
