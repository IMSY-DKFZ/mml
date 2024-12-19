# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import create_creator_func, register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

# TODO refactor as kaggle download, also check reference 3
# TODO check for ICC problems of some images
#  -> solved manually by "find . -type f -name '*.png' - exec mogrify \{\} \;"
#  -> mogrify is part of ImageMagick

REFERENCE = """
@Article{sym12040651,
AUTHOR = {Loey, Mohamed and Smarandache, Florentin and M. Khalifa, Nour Eldeen},
TITLE = {Within the Lack of Chest COVID-19 X-ray Dataset: A Novel Detection Model Based on GAN and Deep Transfer Learning},
JOURNAL = {Symmetry},
VOLUME = {12},
YEAR = {2020},
NUMBER = {4},
ARTICLE-NUMBER = {651},
URL = {https://www.mdpi.com/2073-8994/12/4/651},
ISSN = {2073-8994},
DOI = {10.3390/sym12040651}
}
"""
REFERENCE_2 = """
@article{article,
author = {Eldeen, Nour and Smarandache, Florentin and Loey, Mohamed},
year = {2020},
month = {08},
pages = {},
title = {A Study of the Neutrosophic Set Significance on Deep Transfer Learning Models: an Experimental Case on a Limited COVID-19 Chest X-ray Dataset},
journal = {Cognitive Computation},
doi = {10.1007/s12559-020-09802-9}
}
"""
REFERENCE_3 = """
@unknown{unknown,
author = {Loey, Mohamed and Smarandache, Florentin and Khalifa, Nour},
year = {2020},
month = {04},
pages = {},
title = {A Deep Transfer Learning Model with Classical Data Augmentation and CGAN to Detect COVID-19 from Chest CT Radiography Digital Images},
doi = {10.20944/preprints202004.0252.v3}
}
"""

# NOTE: https://github.com/UCSD-AI4H/COVID-CT is the source of the raw data
"""@article{zhao2020COVID-CT-Dataset,
  title={COVID-CT-Dataset: a CT scan dataset about COVID-19},
  author={Zhao, Jinyu and Zhang, Yichen and He, Xuehai and Xie, Pengtao},
  journal={arXiv preprint arXiv:2003.13865},
  year={2020}
}"""

TASKS = ["raw", "Aug", "CGAN", "Aug&CGAN"]
dataset_name = "covid_chest_ct"


@register_dsetcreator(dset_name=dataset_name)
def create_dset() -> Path:
    dset_creator = DSetCreator(dset_name=dataset_name)
    dset_creator.kaggle_download(
        dataset="mloey1/covid19-chest-ct-image-augmentation-gan-dataset", data_kind=DataKind.TRAINING_DATA
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


def create_covid_ct_classification(task: str, dset_path: Path, alias: str) -> None:
    assert task in TASKS
    if task == "Aug&CGAN":
        task = "Aug+CGAN"
    additional_tag = [Keyword.ARTIFICIAL] if task != "raw" else []
    creator = TaskCreator(
        dset_path=dset_path,
        name=alias,
        task_type=TaskType.CLASSIFICATION,
        desc="COVID-CT-Dataset is a CT Scan Dataset about COVID-19. The medical chest CT"
        "images have been enriched using classical data augmentation and CGAN to "
        "generate more CT images",
        ref=REFERENCE + REFERENCE_2 + REFERENCE_3,
        url="https://www.kaggle.com/datasets/mloey1/covid19-chest-ct-image-augmentation-gan-dataset",
        instr="download dataset via " "https://www.kaggle.com/mloey1/covid19-chest-ct-image-augmentation-gan-dataset",
        lic=License.DATABASE_CONTENTS_LICENSE_1_0,
        release="2020 - version 4",
        keywords=[Keyword.MEDICAL, Keyword.CHEST, Keyword.CT_SCAN, Keyword.TISSUE_PATHOLOGY] + additional_tag,
    )
    classes = ["COVID", "NonCOVID"]
    suffix = "" if task == "raw" else task.replace("+", "")
    split_suffix = "" if task == "raw" else ("+" + task)
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path
        / DataKind.TRAINING_DATA
        / ("COVID-19" + suffix)
        / ("COVID-19" + split_suffix)
        / ("train" + split_suffix),
        dup_id_flag=True,
        classes=classes,
    )
    # we will use the validation split as additional training data and split the folds ourselves
    val_iterator, _ = get_iterator_and_mapping_from_image_dataset(
        root=dset_path
        / DataKind.TRAINING_DATA
        / ("COVID-19" + suffix)
        / ("COVID-19" + split_suffix)
        / ("val" + split_suffix),
        dup_id_flag=True,
        classes=classes,
    )
    # because there are even identical class + image names across train and val we add a "val" prefix
    for val_item in val_iterator:
        val_item[Modality.SAMPLE_ID] = "val_" + val_item[Modality.SAMPLE_ID]
    test_iterator, _ = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / ("COVID-19" + suffix) / ("COVID-19" + split_suffix) / "test",
        dup_id_flag=True,
        classes=classes,
    )
    creator.find_data(
        train_iterator=train_iterator + val_iterator, test_iterator=test_iterator, idx_to_class=idx_to_class
    )
    creator.split_folds(n_folds=5, ensure_balancing=True)
    creator.infer_stats()
    creator.push_and_test()


for task in TASKS:
    alias = f"covid-19-chest-ct-image-augmentation_{task}"
    creator_func = create_creator_func(create_func=create_covid_ct_classification, task=task, alias=alias)
    register_taskcreator(task_name=alias, dset_name=dataset_name)(creator_func)
