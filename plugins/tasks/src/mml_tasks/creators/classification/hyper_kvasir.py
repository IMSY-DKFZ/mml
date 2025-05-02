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
@misc{borgli2020, 
title     = {Hyper-Kvasir: A Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy},
url       = {osf.io/mkzcq}, 
DOI       = {10.31219/osf.io/mkzcq}, 
publisher = {OSF Preprints}, 
author    = {Borgli, Hanna and Thambawita, Vajira and Smedsrud, Pia H and Hicks, Steven and Jha, Debesh and Eskeland, 
            Sigrun L and Randel, Kristin R and Pogorelov, Konstantin and Lux, Mathias and Nguyen, Duc T D and Johansen,
            Dag and Griwodz, Carsten and Stensland, H{\aa}kon K and Garcia-Ceja, Enrique and Schmidt, Peter T and Hammer,
            Hugo L and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas}, 
year      = {2019},
month     = {Dec}}
"""  # noqa W291


@register_dsetcreator(dset_name="hyperkvasir")
def create_hyperkvasir_labeled():
    dset_creator = DSetCreator(dset_name="hyperkvasir")
    dset_creator.download(
        url="https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip",
        file_name="hyper-kvasir-labeled-images.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_creator.download(
        url="https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-unlabeled-images.zip",
        file_name="hyper-kvasir-unlabeled-images.zip",
        data_kind=DataKind.UNLABELED_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


# these are all tasks provided by the hyperkvasir dataset, we split it up into 4 classification tasks
ALL_CLASSES = [
    "cecum",
    "ileum",
    "retroflex-rectum",
    "hemorrhoids",
    "polyps",
    "ulcerative-colitis-grade-0-1",
    "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-1-2",
    "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-2-3",
    "ulcerative-colitis-grade-3",
    "bbps-0-1",
    "bbps-2-3",
    "impacted-stool",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "pylorus",
    "retroflex-stomach",
    "z-line",
    "barretts",
    "barretts-short-segment",
    "esophagitis-a",
    "esophagitis-b-d",
]
# the classification tasks are already defined as subfolders in the data
TASKS = ["anatomical-landmarks", "pathological-findings", "quality-of-mucosal-views", "therapeutic-interventions"]


def create_hyperkvasir_subtask(task: str, dset_path: Path, alias: str) -> None:
    assert task in TASKS
    tags = {
        "anatomical-landmarks": [Keyword.ANATOMICAL_STRUCTURES],
        "pathological-findings": [Keyword.TISSUE_PATHOLOGY],
        "quality-of-mucosal-views": [],
        "therapeutic-interventions": [],
    }[task]
    tags.extend([Keyword.MEDICAL, Keyword.GASTROSCOPY_COLONOSCOPY])
    creator = TaskCreator(
        dset_path=dset_path,
        name=alias,
        task_type=TaskType.CLASSIFICATION,
        desc="Hyper-Kvasir Dataset is a large image and video dataset from the gastrointestinal tract",
        ref=REFERENCE,
        url="https://datasets.simula.no/hyper-kvasir/",
        instr="download zips of labeled and unlabeled images via website",
        lic=License.CC_BY_4_0,
        release="2020",
        keywords=tags,
    )
    classes = {
        "anatomical-landmarks": ["cecum", "ileum", "retroflex-rectum", "pylorus", "retroflex-stomach", "z-line"],
        "pathological-findings": [
            "hemorrhoids",
            "polyps",
            "ulcerative-colitis-grade-0-1",
            "ulcerative-colitis-grade-1",
            "ulcerative-colitis-grade-1-2",
            "ulcerative-colitis-grade-2",
            "ulcerative-colitis-grade-2-3",
            "ulcerative-colitis-grade-3",
            "barretts",
            "barretts-short-segment",
            "esophagitis-a",
            "esophagitis-b-d",
        ],
        "quality-of-mucosal-views": ["bbps-0-1", "bbps-2-3", "impacted-stool"],
        "therapeutic-interventions": ["dyed-lifted-polyps", "dyed-resection-margins"],
    }[task]
    data_iterator = []
    root = dset_path / DataKind.TRAINING_DATA / "labeled-images"
    for tract in ["lower-gi-tract", "upper-gi-tract"]:
        if (root / tract / task).exists():
            for class_folder in (root / tract / task).iterdir():
                assert class_folder.is_dir()
                assert class_folder.name in classes
                for img_path in class_folder.iterdir():
                    data_iterator.append(
                        {
                            Modality.SAMPLE_ID: img_path.stem,
                            Modality.IMAGE: img_path,
                            Modality.CLASS: classes.index(class_folder.name),
                        }
                    )
    idx_to_class = {classes.index(cl): cl for cl in classes}
    creator.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    creator.split_folds(n_folds=5, ensure_balancing=True)
    creator.infer_stats()
    creator.push_and_test()


for task in TASKS:
    alias = f"hyperkvasir_{task}"
    creator_func = create_creator_func(create_func=create_hyperkvasir_subtask, task=task, alias=alias)
    register_taskcreator(task_name=alias, dset_name="hyperkvasir")(creator_func)

# TODO create unlabeled task with unlabaled data
# @register_taskcreator(task_name='hyperkvasir_unlabaled', dset_name='hyperkvasir')
# def create_unlabeled(dset_path):
#     pass
