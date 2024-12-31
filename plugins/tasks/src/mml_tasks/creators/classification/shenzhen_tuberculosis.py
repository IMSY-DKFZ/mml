# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@article{Jaeger2014TwoPC,
  title={Two public chest X-ray datasets for computer-aided screening of pulmonary diseases.},
  author={Stefan Jaeger and Sema Candemir and S. Antani and Y{\`i}-Xi{\'a}ng J W{\'a}ng and Pu-Xuan Lu and George R. Thoma},
  journal={Quantitative imaging in medicine and surgery},
  year={2014},
  volume={4 6},
  pages={
          475-7
        }
} 
"""  # noqa W291

dset_name = "shenzhen_xray_tb"
task_name = "shenzen_chest_xray_tuberculosis"


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.download(
        url="http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
        file_name="ChinaSet_AllFiles.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="The Shenzhen dataset was collected in collaboration with Shenzhen No.3 People’s "
        "Hospital, Guangdong Medical College, Shenzhen, China. The chest X-rays are from "
        "outpatient clinics and were captured as part of the daily hospital routine within "
        "a 1-month period, mostly in September 2012, using a Philips DR Digital Diagnost "
        "system. The set contains 662 frontal chest X-rays, of which 326 are normal cases "
        "and 336 are cases with manifestations of TB, including pediatric X-rays (AP). The "
        "X-rays are provided in PNG format. Their size can vary but is approximately 3K × 3K "
        "pixels..",
        ref=REFERENCE,
        url="https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html",
        instr="download via http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
        lic=License.UNKNOWN,
        release="2014",
        keywords=[Keyword.MEDICAL, Keyword.X_RAY, Keyword.CHEST, Keyword.TISSUE_PATHOLOGY],
    )
    idx_to_class = {0: "normal", 1: "abnormal"}
    train_iterator = []
    for img_path in sorted((dset_path / DataKind.TRAINING_DATA / "ChinaSet_AllFiles" / "CXR_png").glob("*.png")):
        train_iterator.append(
            {Modality.SAMPLE_ID: img_path.stem[:-2], Modality.IMAGE: img_path, Modality.CLASS: int(img_path.stem[-1])}
        )
    task.find_data(train_iterator=train_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
