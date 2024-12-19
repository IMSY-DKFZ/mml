# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.archive_extractors import unpack_files
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import create_creator_func, register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@article{Smedsrud2021,
  title = {{Kvasir-Capsule, a video capsule endoscopy dataset}},
  author = {
    Smedsrud, Pia H and Thambawita, Vajira and Hicks, Steven A and
    Gjestang, Henrik and Nedrejord, Oda Olsen and N{\ae}ss, Espen and
    Borgli, Hanna and Jha, Debesh and Berstad, Tor Jan Derek and
    Eskeland, Sigrun L and Lux, Mathias and Espeland, H{\aa}vard and
    Petlund, Andreas and Nguyen, Duc Tien Dang and Garcia-Ceja, Enrique and
    Johansen, Dag and Schmidt, Peter T and Toth, Ervin and
    Hammer, Hugo L and de Lange, Thomas and Riegler, Michael A and
    Halvorsen, P{\aa}l
  },
  doi = {10.1038/s41597-021-00920-z},
  journal = {Scientific Data},
  number = {1},
  pages = {142},
  volume = {8},
  year = {2021}
}
"""

dset_name = "kvasir_capsule"
ALL_TASKS = ["anatomy", "content", "pathologies"]


# TODO think about including the labeled videos similar to Cholec80


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    instructions = f"""
                 Download link is not working as expected. Please download the dataset manually by following these 
                 instructions: Go to https://osf.io/dv2ag/files/ and select the "labelled_images" folder in the 
                 Google Drive section. Afterwards a "Download as zip" button pops up at the top, which needs to be 
                 pressed. Once the download is complete place the downloaded folder 'labelled_images.zip' in
                 <MML_DATA_ROOT>/DOWNLOADS/{dset_name}
            """  # noqa W291
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.verify_pre_download(
        instructions=instructions, file_name="labelled_images.zip", data_kind=DataKind.TRAINING_DATA
    )
    # the zip contains itself archives that need to be extracted, this requires some "hacky solution"
    # first remove the originally selected archive from automatic processing
    arch = dset_creator.archives.pop()
    # then extract manually inside DOWNLOADS
    unpack_files(archives=[arch], target=dset_creator.download_path)
    for sub_folder in [
        "ampulla_of_vater",
        "angiectasia",
        "blood_fresh",
        "blood_hematin",
        "erosion",
        "erythema",
        "foreign_body",
        "ileocecal_valve",
        "lymphangiectasia",
        "normal_clean_mucosa",
        "pylorus",
        "polyp",
        "reduced_mucosal_view",
        "ulcer",
    ]:
        second_instructions = f"""
        This is an intermediate step, if errors occur, remove the intermediate folders in 
        {dset_creator.download_path} (except the labelled_images.zip) and restart."""  # noqa W291
        dset_creator.verify_pre_download(
            instructions=second_instructions, data_kind=DataKind.TRAINING_DATA, file_name=f"{sub_folder}.tar.gz"
        )
    # finally do the usual routine
    dset_path = dset_creator.unpack_and_store()
    return dset_path


def create_kvasir_capsule_subtask(task: str, dset_path: Path, alias: str) -> None:
    tags = {
        "anatomy": [Keyword.MEDICAL, Keyword.ENDOSCOPY, Keyword.CAPSULE_ENDOSCOPY, Keyword.ANATOMICAL_STRUCTURES],
        "content": [Keyword.MEDICAL, Keyword.ENDOSCOPY, Keyword.CAPSULE_ENDOSCOPY],
        "pathologies": [Keyword.MEDICAL, Keyword.ENDOSCOPY, Keyword.CAPSULE_ENDOSCOPY, Keyword.TISSUE_PATHOLOGY],
    }[task]
    creator = TaskCreator(
        dset_path=dset_path,
        name=alias,
        task_type=TaskType.CLASSIFICATION,
        desc="This is the official repository for the Kvasir-Capsule dataset, which is the largest "
        "publicly released PillCAM dataset. In total, the dataset contains 47,238 labeled "
        "images and 117 videos, where it captures anatomical landmarks and pathological and "
        "normal findings. The results is more than 4,741,621 images and video frames all "
        "together.",
        ref=REFERENCE,
        url="https://datasets.simula.no/kvasir-capsule/",
        instr="download via https://osf.io/dv2ag/files/",
        lic=License.CC_BY_4_0,
        release="2021",
        keywords=tags,
    )
    # ATTENTION: For simplicity we drop classes below 200 samples!
    classes = {
        "anatomy": [
            "Pylorus",
            # 'Ampulla of vater', # to few samples and one overlap with a pylorus sample
            "Ileocecal valve",
        ],
        "content": [
            "Normal clean mucosa",
            "Reduced mucosal view",
            "Blood - fresh",
            # 'Blood - hematin',  # to few samples
            "Foreign body",
        ],
        "pathologies": [
            "Normal clean mucosa",
            # 'Erythema',  # to few samples
            "Angiectasia",
            "Erosion",
            "Ulcer",
            "Lymphangiectasia",
            # 'Polyp'   # to few samples
        ],
    }[task]
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA, classes=classes
    )
    creator.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    creator.split_folds(n_folds=5, ensure_balancing=True)
    creator.infer_stats()
    creator.push_and_test()


for task in ALL_TASKS:
    alias = f"{dset_name}_{task}"
    creator_func = create_creator_func(create_func=create_kvasir_capsule_subtask, task=task, alias=alias)
    register_taskcreator(task_name=alias, dset_name=dset_name)(creator_func)
