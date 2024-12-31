# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@article{KERMANY20181122,
title = {Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning},
journal = {Cell},
volume = {172},
number = {5},
pages = {1122-1131.e9},
year = {2018},
issn = {0092-8674},
doi = {https://doi.org/10.1016/j.cell.2018.02.010},
url = {https://www.sciencedirect.com/science/article/pii/S0092867418301545},
author = {Daniel S. Kermany and Michael Goldbaum and Wenjia Cai and Carolina C.S. Valentim and Huiying Liang and 
          Sally L. Baxter and Alex McKeown and Ge Yang and Xiaokang Wu and Fangbing Yan and Justin Dong and 
        Made K. Prasadha and Jacqueline Pei and Magdalene Y.L. Ting and Jie Zhu and Christina Li and Sierra Hewett 
        and Jason Dong and Ian Ziyar and Alexander Shi and Runze Zhang and Lianghong Zheng and Rui Hou and 
        William Shi and Xin Fu and Yaou Duan and Viet A.N. Huu and Cindy Wen and Edward D. Zhang and 
        Charlotte L. Zhang and Oulan Li and Xiaobo Wang and Michael A. Singer and Xiaodong Sun and Jie Xu and 
        Ali Tafreshi and M. Anthony Lewis and Huimin Xia and Kang Zhang},
keywords = {artificial intelligence, transfer learning, deep learning, age-related macular degeneration, choroidal 
            neovascularization, diabetic retinopathy, diabetic macular edema, screening, optical coherence tomography, 
            pneumonia}
}
"""  # noqa W291

dataset_name = "pneumonia"


@register_dsetcreator(dset_name=dataset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dataset_name)
    dset_creator.kaggle_download(dataset="tolgadincer/labeled-chest-xray-images", data_kind=DataKind.TRAINING_DATA)
    dset_path = dset_creator.unpack_and_store()
    return dset_path


task_name = "pneumonia_classification"


@register_taskcreator(task_name=task_name, dset_name="pneumonia")
def create_covid_ct_classification(dset_path: Path):
    covid_ct_classification = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="This dataset contains chest X-Ray images labeled as NORMAL or PNEUMONIA",
        ref=REFERENCE,
        url="https://www.kaggle.com/tolgadincer/labeled-chest-xray-images/",
        instr="download dataset via https://www.kaggle.com/tolgadincer/labeled-chest-xray-images/download",
        lic=License.CC_BY_4_0,
        release="2018 v3",
        keywords=[Keyword.MEDICAL, Keyword.CHEST, Keyword.X_RAY, Keyword.TISSUE_PATHOLOGY],
    )
    classes = ["NORMAL", "PNEUMONIA"]
    train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "chest_xray" / "train", classes=classes
    )
    test_iterator, _ = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "chest_xray" / "test", classes=classes
    )
    covid_ct_classification.find_data(
        train_iterator=train_iterator, test_iterator=test_iterator, idx_to_class=idx_to_class
    )
    covid_ct_classification.split_folds(n_folds=5, ensure_balancing=True)
    covid_ct_classification.infer_stats()
    covid_ct_classification.push_and_test()
