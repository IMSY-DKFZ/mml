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

REFERENCE = """
@ARTICLE{2017arXiv171206957R,
       author = {{Rajpurkar}, Pranav and {Irvin}, Jeremy and {Bagul}, Aarti and {Ding}, Daisy and {Duan}, Tony and 
       {Mehta}, Hershel and {Yang}, Brandon and {Zhu}, Kaylie and {Laird}, Dillon and {Ball}, Robyn L. and {Langlotz}, 
       Curtis and {Shpanskaya}, Katie and {Lungren}, Matthew P. and {Ng}, Andrew Y.},
        title = "{MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs}",
      journal = {arXiv e-prints},
     keywords = {Physics - Medical Physics, Computer Science - Artificial Intelligence},
         year = 2017,
        month = dec,
          eid = {arXiv:1712.06957},
        pages = {arXiv:1712.06957},
archivePrefix = {arXiv},
       eprint = {1712.06957},
 primaryClass = {physics.med-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017arXiv171206957R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""  # noqa W291
# labels in the csv: 0 negative, 1 positive
TASKS = ["XR_WRIST", "XR_SHOULDER", "XR_HUMERUS", "XR_HAND", "XR_FOREARM", "XR_FINGER", "XR_ELBOW"]


@register_dsetcreator(dset_name="mura")
def create_dset():
    instructions = """
         Download link is not available. Please download the dataset by clicking on the download button manually via
         https://stanfordmlgroup.github.io/competitions/mura/
         Once the download is complete place the downloaded folder 'MURA-v1.1.zip' in
         <MML_DATA_ROOT>\\DOWNLOADS\\mura
    """
    dset_creator = DSetCreator(dset_name="mura")
    dset_creator.verify_pre_download(
        file_name="MURA-v1.1.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


def create_mura_subtask(task: str, dset_path: Path, alias: str) -> None:
    assert task in TASKS
    classify_mura = TaskCreator(
        dset_path=dset_path,
        name=alias,
        task_type=TaskType.CLASSIFICATION,
        desc="MURA (musculoskeletal radiographs) is a large dataset of bone X-rays.",
        ref=REFERENCE,
        url="https://stanfordmlgroup.github.io/competitions/mura/",
        instr="download via stanfordmlgroup.github.io/competitions/mura/",
        lic=License.UNKNOWN,
        release="2017",
        keywords=[Keyword.MEDICAL, Keyword.X_RAY, Keyword.BONE, Keyword.TISSUE_PATHOLOGY],
    )
    train_task_folder = dset_path / DataKind.TRAINING_DATA / "MURA-v1.1" / "train" / task
    test_task_folder = dset_path / DataKind.TRAINING_DATA / "MURA-v1.1" / "valid" / task
    train_iterator, test_iterator = [], []
    for iterator, task_folder in [(train_iterator, train_task_folder), (test_iterator, test_task_folder)]:
        for patient in task_folder.iterdir():
            for study in patient.iterdir():
                if "negative" in study.name:
                    class_idx = 0
                else:
                    class_idx = 1
                for img_path in study.iterdir():
                    if img_path.stem.startswith("._"):  # there are three corrupted images in the wrist task
                        continue
                    iterator.append(
                        {
                            Modality.SAMPLE_ID: task + "_" + patient.name + "_" + study.name[:8] + img_path.name,
                            Modality.IMAGE: img_path,
                            Modality.CLASS: class_idx,
                        }
                    )
    classify_mura.find_data(
        train_iterator=train_iterator, test_iterator=test_iterator, idx_to_class={0: "Negative", 1: "Positive"}
    )
    classify_mura.split_folds(n_folds=5, ensure_balancing=True)
    classify_mura.infer_stats()
    classify_mura.push_and_test()


for task in TASKS:
    alias = f"mura_{task.lower()}"
    creator_func = create_creator_func(create_func=create_mura_subtask, task=task, alias=alias)
    register_taskcreator(task_name=alias, dset_name="mura")(creator_func)
