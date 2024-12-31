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
@article{zhao2020COVID-CT-Dataset,
  title={COVID-CT-Dataset: a CT scan dataset about COVID-19},
  author={Zhao, Jinyu and Zhang, Yichen and He, Xuehai and Xie, Pengtao},
  journal={arXiv preprint arXiv:2003.13865}, 
  year={2020}
}
"""  # noqa W291

dset_name = "covid_ct_crawled"
task_name = "crawled_covid_ct_classification"


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    instructions = f"""
                 Download link is not available. Please download the dataset by clicking on the download button 
                 manually via https://github.com/UCSD-AI4H/COVID-CT/blob/master/Images-processed/CT_COVID.zip 
                  and https://github.com/UCSD-AI4H/COVID-CT/blob/master/Images-processed/CT_NonCOVID.zip 
                  to download the two zip folders. 
                 Once the downloads are complete place the downloaded folders 'CT_COVID.zip' and 'CT_NonCOVID.zip' in
                 <MML_DATA_ROOT>/DOWNLOADS/{dset_name}
            """  # noqa W291
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.verify_pre_download(
        file_name="CT_NonCOVID.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_creator.verify_pre_download(
        file_name="CT_COVID.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="The COVID-CT-Dataset has 349 CT images containing clinical findings of COVID-19 from 216 "
        "patients. The images are collected from COVID19-related papers from medRxiv, bioRxiv, "
        "NEJM, JAMA, Lancet, etc. CTs containing COVID-19 abnormalities are selected by reading "
        "the figure captions in the papers. All copyrights of the data belong to the authors and "
        "publishers of these papers.",
        ref=REFERENCE,
        url="https://github.com/UCSD-AI4H/COVID-CT",
        instr="download via https://github.com/UCSD-AI4H/COVID-CT/tree/master/Images-processed",
        lic=License.UNKNOWN,
        release="2020",
        keywords=[Keyword.MEDICAL, Keyword.CT_SCAN, Keyword.CHEST, Keyword.TISSUE_PATHOLOGY],
    )
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=["CT_NonCOVID", "CT_COVID"]
    )
    task.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
