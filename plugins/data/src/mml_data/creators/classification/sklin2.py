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
@article{Faria2019LightFI,
  title={Light Field Image Dataset of Skin Lesions},
  author={S{\'e}rgio M. M. Faria and Jose N. Filipe and Pedro M. M. Pereira and Luis M. N. Tavora and Pedro A. Amado 
  Assunç{\~a}o and Miguel O. Santos and Rui Fonseca-Pinto and Felicidade Santiago and Victoria Dominguez and Martinha 
  Henrique},
  journal={2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  year={2019},
  pages={3905-3908}
}
"""  # noqa W291

dset_name = "sklin2"
task_name = "sklin2_skin_lesions"


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    instructions = f"""
                 Download link is not available. Please download the dataset by clicking on the download button 
                 manually via https://www.it.pt/AutomaticPage?id=3459 -> link at ther very top -> select SKLIN2_v1 ->
                 download the "Dermatoscopic" folder. 
                 Once the download is complete place the downloaded folder 'Dermatoscopic.zip' in
                 <MML_DATA_ROOT>/DOWNLOADS/{dset_name}
            """  # noqa W291
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.verify_pre_download(
        file_name="Dermatoscopic.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="This dataset contains 250 light fields, captured with a focused plenoptic "
        "camera and classified into eight clinical categories, according to the type "
        "of lesion. Each light field is comprised of 81 different views of the same "
        "lesion. The database also includes the dermatoscopic image of each lesion.",
        ref=REFERENCE,
        url="https://www.it.pt/AutomaticPage?id=3459",
        instr="download via https://www.it.pt/AutomaticPage?id=3459",
        lic=License.UNKNOWN,
        release="2019 - v1",
        keywords=[Keyword.MEDICAL, Keyword.DERMATOSCOPY, Keyword.TISSUE_PATHOLOGY],
    )
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "Dermatoscopic", classes=None
    )
    task.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
