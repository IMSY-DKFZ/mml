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
@article{article,
author = {Giotis, Ioannis and Molders, Nynke and Land, Sander and Biehl, Michael and Jonkman, Marcel and Petkov, Nicolai},
year = {2015},
month = {05},
pages = {},
title = {MED-NODE: A Computer-Assisted Melanoma Diagnosis System using Non-Dermoscopic Images},
volume = {42},
journal = {Expert Systems with Applications},
doi = {10.1016/j.eswa.2015.04.034}
}
"""


@register_dsetcreator(dset_name="mednode")
def create_mednode():
    dset_creator = DSetCreator(dset_name="mednode")
    dset_creator.download(
        url="http://www.cs.rug.nl/~imaging/databases/melanoma_naevi/complete_mednode_dataset.zip",
        file_name="complete_mednode_dataset.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="mednode_melanoma_classification", dset_name="mednode")
def create_melanoma_classification(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name="mednode_melanoma_classification",
        task_type=TaskType.CLASSIFICATION,
        desc="The dataset consists of melanoma and naevus images from the digital image"
        " archive of the Department of Dermatology of the University Medical "
        "Center Groningen (UMCG)",
        ref=REFERENCE,
        url="http://www.cs.rug.nl/~imaging/databases/melanoma_naevi/",
        instr="download via http://www.cs.rug.nl/~imaging/databases/melanoma_naevi/",
        lic=License.UNKNOWN,
        release="2015",
        keywords=[Keyword.MEDICAL, Keyword.DERMATOSCOPY, Keyword.TISSUE_PATHOLOGY],
    )
    classes = ["naevus", "melanoma"]
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "complete_mednode_dataset", classes=classes
    )
    task.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
