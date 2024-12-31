# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import pandas as pd

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import create_creator_func, register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@misc{irvin2019chexpert,
      title={CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison}, 
      author={Jeremy Irvin and Pranav Rajpurkar and Michael Ko and Yifan Yu and Silviana Ciurea-Ilcus and Chris Chute 
      and Henrik Marklund and Behzad Haghgoo and Robyn Ball and Katie Shpanskaya and Jayne Seekins and David A. Mong and
      Safwan S. Halabi and Jesse K. Sandberg and Ricky Jones and David B. Larson and Curtis P. Langlotz and 
      Bhavik N. Patel and Matthew P. Lungren and Andrew Y. Ng},
      year={2019},
      eprint={1901.07031},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""  # noqa W291
# labels in the csv: -1 uncertain, 0 negative, 1 positive, blank unmentioned
# -> ignore -1 cases, thus focus on certain pathologies only
TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
# the no finding class is obsolete with regard to the specific pathologies
TASKS.remove("No Finding")


@register_dsetcreator(dset_name="chexpert")
def create_dset():
    instructions = """
         Download link is not available. Please download the dataset by clicking on the download button manually via
         https://stanfordmlgroup.github.io/competitions/chexpert/
         Once the download is complete place the downloaded folder 'CheXpert-v1.0-small.zip' in
         <MML_DATA_ROOT>\\DOWNLOADS\\chexpert
    """
    dset_creator = DSetCreator(dset_name="chexpert")
    dset_creator.verify_pre_download(
        file_name="CheXpert-v1.0-small.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


def create_chexpert_subtask(task: str, dset_path: Path, alias: str) -> None:
    assert task in TASKS
    classify_chexpert = TaskCreator(
        dset_path=dset_path,
        name=alias,
        task_type=TaskType.CLASSIFICATION,
        desc="CheXpert is a large public dataset for chest radiograph interpretation",
        ref=REFERENCE,
        url="https://stanfordmlgroup.github.io/competitions/chexpert/",
        instr="download via stanfordmlgroup.github.io/competitions/chexpert/",
        lic=License.UNKNOWN,
        release="2019",
        keywords=[Keyword.MEDICAL, Keyword.X_RAY, Keyword.CHEST, Keyword.TISSUE_PATHOLOGY],
    )
    train_iterator, test_iterator = [], []
    train_matrix = pd.read_csv(dset_path / DataKind.TRAINING_DATA / "CheXpert-v1.0-small" / "train.csv")
    test_matrix = pd.read_csv(dset_path / DataKind.TRAINING_DATA / "CheXpert-v1.0-small" / "valid.csv")
    for iterator, matrix in [(train_iterator, train_matrix), (test_iterator, test_matrix)]:
        for _, row in matrix.iterrows():
            if row[task] in [0, 1]:
                iterator.append(
                    {
                        Modality.SAMPLE_ID: row["Path"],
                        Modality.IMAGE: dset_path / DataKind.TRAINING_DATA / row["Path"],
                        Modality.CLASS: int(row[task]),
                    }
                )
    classify_chexpert.find_data(
        train_iterator=train_iterator, test_iterator=test_iterator, idx_to_class={0: "Negative", 1: "Positive"}
    )
    classify_chexpert.split_folds(n_folds=5, ensure_balancing=True)
    classify_chexpert.infer_stats()
    classify_chexpert.push_and_test()


for task in TASKS:
    alias = f"chexpert_{task.lower().replace(' ', '_')}"
    creator_func = create_creator_func(create_func=create_chexpert_subtask, task=task, alias=alias)
    register_taskcreator(task_name=alias, dset_name="chexpert")(creator_func)
