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
@inproceedings{Pogorelov:2017:NBP:3083187.3083216,
author    = {Pogorelov, Konstantin and Randel, Kristin Ranheim and de Lange, Thomas and Eskeland, Sigrun Losada and Griwodz,
             Carsten and Johansen, Dag and Spampinato, Concetto and Taschwer, Mario and Lux, Mathias and Schmidt, 
            Peter Thelin and Riegler, Michael and Halvorsen, P{\aa}l},
title     = {Nerthus: A Bowel Preparation Quality Video Dataset},
booktitle = {Proceedings of the 8th ACM on Multimedia Systems Conference},
series    = {MMSys'17},
year      = {2017},
isbn      = {978-1-4503-5002-0},
location  = {Taipei, Taiwan},
pages     = {170--174},
numpages  = {5},
url       = {http://doi.acm.org/10.1145/3083187.3083216},
doi       = {10.1145/3083187.3083216},
acmid     = {3083216},
publisher = {ACM},
address   = {New York, NY, USA},
}
"""  # noqa W291


@register_dsetcreator(dset_name="nerthus")
def create_nerthus():
    dset_creator = DSetCreator(dset_name="nerthus")
    dset_creator.download(
        url="https://datasets.simula.no/downloads/nerthus/nerthus.zip",
        file_name="nerthus-dataset-frames.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="nerthus_bowel_cleansing_quality", dset_name="nerthus")
def create_nerthus_bowel_cleansing_quality(dset_path: Path):
    bowel_cleansing_quality = TaskCreator(
        dset_path=dset_path,
        name="nerthus_bowel_cleansing_quality",
        task_type=TaskType.CLASSIFICATION,
        desc="Nerthus dataset for bowel cleansing quality",
        ref=REFERENCE,
        url="https://datasets.simula.no/nerthus/",
        instr="download via datasets.simula.no/downloads/nerthus/nerthus.zip",
        lic=License.UNKNOWN,
        release="2017",
        keywords=[Keyword.MEDICAL, Keyword.GASTROSCOPY_COLONOSCOPY, Keyword.ENDOSCOPY],
    )
    classes = ["0", "1", "2", "3"]
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "nerthus-dataset-frames", classes=classes
    )
    bowel_cleansing_quality.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    bowel_cleansing_quality.split_folds(n_folds=5, ensure_balancing=True)
    bowel_cleansing_quality.infer_stats()
    bowel_cleansing_quality.push_and_test()


if __name__ == "__main__":
    dset_path = create_nerthus()
    create_nerthus_bowel_cleansing_quality(dset_path)
