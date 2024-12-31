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
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@article{Kawahara2019SevenPointCA,
  title={Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets},
  author={Jeremy Kawahara and Sara Daneshvar and Giuseppe Argenziano and G. Hamarneh},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2019},
  volume={23},
  pages={538-546}
}
"""

dset_name = "derm7pt"
task_name = "derm7pt_skin_lesions"
instructions = f"""
                 It is necessary to register prior to downloading the data. Follow the link 
                 https://derm.cs.sfu.ca/Download.html and register yourself. Afterwards use the download link 
                 http://derm.cs.sfu.ca/restricted/release_v0.zip together with your credentials to get the data.
                 Once the download is complete place the downloaded folder 'release_v0.zip' in
                 <MML_DATA_ROOT>/DOWNLOADS/{dset_name}
            """  # noqa W291


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.verify_pre_download(
        file_name="release_v0.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name=task_name, dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name=task_name,
        task_type=TaskType.CLASSIFICATION,
        desc="We provide a database for evaluating computerized image-based prediction "
        "of the 7-point skin lesion malignancy checklist. The dataset includes over "
        "2000 clinical and dermoscopy color images, along with corresponding "
        "structured metadata tailored for training and evaluating computer aided "
        "diagnosis (CAD) systems.",
        ref=REFERENCE,
        url="https://derm.cs.sfu.ca/Download.html",
        instr=instructions,
        lic=License.UNKNOWN,
        release="2019 - v0",
        keywords=[Keyword.MEDICAL, Keyword.DERMATOSCOPY, Keyword.TISSUE_PATHOLOGY],
    )
    # merging infrequent classes according to https://github.com/jeremykawahara/derm7pt/blob/master/derm7pt/dataset.py#L475
    infrequent_mapper = {"basal cell carcinoma": "BCC", "seborrheic keratosis": "SK"}
    infrequent_mapper.update(
        {
            infrequent: "NEV"
            for infrequent in [
                "nevus",
                "blue nevus",
                "clark nevus",
                "combined nevus",
                "congenital nevus",
                "dermal nevus",
                "recurrent nevus",
                "reed or spitz nevus",
            ]
        }
    )
    infrequent_mapper.update(
        {
            infrequent: "MEL"
            for infrequent in [
                "melanoma",
                "melanoma",
                "melanoma (in situ)",
                "melanoma (less than 0.76 mm)",
                "melanoma (0.76 to 1.5 mm)",
                "melanoma (more than 1.5 mm)",
                "melanoma metastasis",
            ]
        }
    )
    infrequent_mapper.update(
        {
            infrequent: "MISC"
            for infrequent in [
                "DF/LT/MLS/MISC",
                "dermatofibroma",
                "lentigo",
                "melanosis",
                "miscellaneous",
                "vascular lesion",
            ]
        }
    )
    meta_path = dset_path / DataKind.TRAINING_DATA / "release_v0" / "meta"
    meta_info = pd.read_csv(meta_path / "meta.csv", header=0, index_col=0)
    classes = sorted(list(set(infrequent_mapper.values())))
    indices = {
        "train": pd.read_csv(meta_path / "train_indexes.csv")["indexes"].tolist(),
        "val": pd.read_csv(meta_path / "valid_indexes.csv")["indexes"].tolist(),
        "test": pd.read_csv(meta_path / "test_indexes.csv")["indexes"].tolist(),
    }
    data_iterator = {}
    for split in ["train", "val", "test"]:
        data_iterator[split] = []
        for idx in indices[split]:
            data_iterator[split].append(
                {
                    Modality.SAMPLE_ID: str(idx),
                    Modality.IMAGE: (
                        dset_path / DataKind.TRAINING_DATA / "release_v0" / "images" / meta_info.iloc[idx]["derm"]
                    ),
                    Modality.CLASS: classes.index(infrequent_mapper[meta_info.iloc[idx]["diagnosis"]]),
                }
            )
    idx_to_class = {classes.index(cl): cl for cl in classes}
    # we merge train and validation split to folderize later on ourselves
    task.find_data(
        train_iterator=data_iterator["train"] + data_iterator["val"],
        test_iterator=data_iterator["test"],
        idx_to_class=idx_to_class,
    )
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
