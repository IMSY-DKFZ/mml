# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

import pandas as pd

from mml.core.data_loading.task_attributes import Keyword, License, Modality, RGBInfo, Sizes, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

logger = logging.getLogger(__name__)

REFERENCE = """
@article{Rotemberg2021APD,
  title={A patient-centric dataset of images and metadata for identifying melanomas using clinical context},
  author={Veronica Rotemberg and Nicholas R. Kurtansky and Brigid Betz-Stablein and Liam J. Caffery and Emmanouil 
        Chousakos and Noel C. F. Codella and Marc Combalia and Stephen W. Dusza and Pascale Guitera and David 
        Gutman and Allan C. Halpern and Harald Kittler and Kivanç K{\"o}se and Steve G. Langer and Konstantinos 
        Liopryis and Josep Malvehy and Shenara Musthaq and Jabpani Nanda and Ofer Reiter and George Shih and 
        Alexander J. Stratigos and Philipp Tschandl and Jochen Weber and Hans Peter Soyer},
  journal={Scientific Data},
  year={2021},
  volume={8}
}
"""  # noqa W291

dset_name = "isic20"
task_name = "isic20_melanoma_classification"


@register_dsetcreator(dset_name=dset_name)
def create_isic20() -> Path:
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.download(
        url="https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv",
        file_name="ISIC_2020_Training_Duplicates.csv",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_creator.download(
        url="https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip",
        file_name="ISIC_2020_Training_JPEG.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_creator.download(
        url="https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv",
        file_name="ISIC_2020_Training_GroundTruth.csv",
        data_kind=DataKind.TRAINING_LABELS,
    )
    dset_creator.download(
        url="https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip",
        file_name="ISIC_2020_Test_JPEG.zip",
        data_kind=DataKind.TESTING_DATA,
    )  # -> no ground truth yet
    dset_path = dset_creator.unpack_and_store()
    # remove duplicates
    duplicates = pd.read_csv(dset_path / DataKind.TRAINING_DATA / "ISIC_2020_Training_Duplicates.csv")["image_name_2"]
    counter = 0
    for dup in duplicates:
        p = dset_path / DataKind.TRAINING_DATA / "train" / f"{dup}.jpg"
        if p.exists():
            p.unlink()
            counter += 1
    logger.info(f"Removed {counter} duplicates.")
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="The dataset contains 33,126 dermoscopic training images of unique benign and malignant "
        "skin lesions from over 2,000 patients. Each image is associated with one of these "
        "individuals using a unique patient identifier. All malignant diagnoses have been "
        "confirmed via histopathology, and benign diagnoses have been confirmed using either "
        "expert agreement, longitudinal follow-up, or histopathology.",
        ref=REFERENCE,
        url="https://challenge2020.isic-archive.com/",
        instr="download via kaggle (https://www.kaggle.com/c/siim-isic-melanoma-classification/data) "
        "or homepage(https://challenge2020.isic-archive.com/)",
        lic=License.CC_BY_NC_4_0,
        release="2020",
        keywords=[Keyword.MEDICAL, Keyword.DERMATOSCOPY, Keyword.TISSUE_PATHOLOGY],
    )
    idx_to_class = {0: "benign", 1: "malignant"}
    train_annotation = pd.read_csv(
        dset_path / DataKind.TRAINING_LABELS / "ISIC_2020_Training_GroundTruth.csv", index_col=0
    )["target"]
    train_iterator = []
    for img_path in (dset_path / DataKind.TRAINING_DATA / "train").iterdir():
        train_iterator.append(
            {
                Modality.SAMPLE_ID: img_path.stem,
                Modality.IMAGE: img_path,
                Modality.CLASS: int(train_annotation.at[img_path.stem]),
            }
        )
    test_iterator = []
    for img_path in (dset_path / DataKind.TESTING_DATA / "ISIC_2020_Test_Input").iterdir():
        if img_path.suffix != ".jpg":
            continue
        test_iterator.append({Modality.SAMPLE_ID: img_path.stem, Modality.IMAGE: img_path})
    task.find_data(train_iterator=train_iterator, test_iterator=test_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    # because of varying sizes and high resolution inference takes multiple hours, set stats instead
    task.set_stats(
        means=RGBInfo(*[0.8050940632820129, 0.6202492117881775, 0.5902113318443298]),
        stds=RGBInfo(*[0.15143747627735138, 0.177970752120018, 0.20396198332309723]),
        sizes=Sizes(*[480, 6000, 640, 6000]),
    )
    task.push_and_test()
