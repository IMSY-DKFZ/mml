# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

# creates the 4 lapgyn datasets and tasks
from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@inproceedings{DBLP:conf/mmsys/LeibetsederPPKM18,
  author    = {Andreas Leibetseder and
               Stefan Petscharnig and
               Manfred J{\"{u}}rgen Primus and
               Sabrina Kletz and
               Bernd M{\"{u}}nzer and
               Klaus Schoeffmann and
               J{\"{o}}rg Keckstein},
  title     = {Lapgyn4: a dataset for 4 automatic content analysis problems in the
               domain of laparoscopic gynecology},
  booktitle = {Proceedings of the 9th {ACM} Multimedia Systems Conference, MMSys
               2018, Amsterdam, The Netherlands, June 12-15, 2018},
  pages     = {357--362},
  publisher = {{ACM}},
  year      = {2018},
  url       = {https://doi.org/10.1145/3204949.3208127},
  doi       = {10.1145/3204949.3208127},
  timestamp = {Wed, 21 Nov 2018 12:44:03 +0100},
  biburl    = {https://dblp.org/rec/conf/mmsys/LeibetsederPPKM18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

ADDITIONAL_REFERENCE = """
@article{DBLP:journals/corr/TwinandaSMMMP16,
  author    = {Andru Putra Twinanda and
               Sherif Shehata and
               Didier Mutter and
               Jacques Marescaux and
               Michel de Mathelin and
               Nicolas Padoy},
  title     = {EndoNet: {A} Deep Architecture for Recognition Tasks on Laparoscopic
               Videos},
  journal   = {CoRR},
  volume    = {abs/1602.03012},
  year      = {2016},
  url       = {http://arxiv.org/abs/1602.03012},
  archivePrefix = {arXiv},
  eprint    = {1602.03012},
  timestamp = {Mon, 13 Aug 2018 16:46:10 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/TwinandaSMMMP16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


@register_dsetcreator(dset_name="lapgyn4")
def create_lapgyn():
    dset_creator = DSetCreator(dset_name="lapgyn4")
    dset_creator.download(
        url="http://ftp.itec.aau.at/datasets/LapGyn4/downloads/v1_2/LapGyn4_v1.2.zip",
        file_name="LapGyn4_v1.2.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="lapgyn4_anatomical_structures", dset_name="lapgyn4")
def create_task_as(dset_path: Path):
    anatomical_structures = TaskCreator(
        dset_path=dset_path,
        name="lapgyn4_anatomical_structures",
        task_type=TaskType.CLASSIFICATION,
        desc="anatomical structures subset of LapGyn4 dataset",
        ref=REFERENCE,
        url="http://ftp.itec.aau.at/datasets/LapGyn4/",
        instr="download via ftp.itec.aau.at/datasets/LapGyn4/LapGyn4_v1.2.zip",
        lic=License.CC_BY_NC_4_0,
        release="v1.2",
        keywords=[Keyword.LAPAROSCOPY, Keyword.ANATOMICAL_STRUCTURES, Keyword.MEDICAL, Keyword.GYNECOLOGY],
    )
    classes = ["Colon", "Liver", "Ovary", "Oviduct", "Uterus"]
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "LapGyn4_v1.2" / "Anatomical_Structures", classes=classes
    )
    anatomical_structures.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    anatomical_structures.split_folds(n_folds=5, ensure_balancing=True)
    anatomical_structures.infer_stats()
    anatomical_structures.push_and_test()


@register_taskcreator(task_name="lapgyn4_surgical_actions", dset_name="lapgyn4")
def create_task_sa(dset_path: Path):
    surgical_actions = TaskCreator(
        dset_path=dset_path,
        name="lapgyn4_surgical_actions",
        task_type=TaskType.CLASSIFICATION,
        desc="surgical action subset of LapGyn4 dataset",
        ref=REFERENCE,
        url="http://ftp.itec.aau.at/datasets/LapGyn4/",
        instr="download via ftp.itec.aau.at/datasets/LapGyn4/LapGyn4_v1.2.zip",
        lic=License.CC_BY_NC_4_0,
        release="v1.2",
        keywords=[Keyword.LAPAROSCOPY, Keyword.ENDOSCOPIC_INSTRUMENTS, Keyword.MEDICAL, Keyword.GYNECOLOGY],
    )
    classes = [
        "Coagulation",
        "Cutting_Cold",
        "Cutting_HF",
        "Dissection_Blunt",
        "Injection",
        "Sling_Hysterectomy",
        "Suction_Irrigation",
        "Suturing",
    ]
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "LapGyn4_v1.2" / "Surgical_Actions", classes=classes
    )
    surgical_actions.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    surgical_actions.split_folds(n_folds=5, ensure_balancing=True)
    surgical_actions.infer_stats()
    surgical_actions.push_and_test()


@register_taskcreator(task_name="lapgyn4_instrument_count", dset_name="lapgyn4")
def create_task_ic(dset_path: Path):
    instrument_count = TaskCreator(
        dset_path=dset_path,
        name="lapgyn4_instrument_count",
        task_type=TaskType.CLASSIFICATION,
        desc="instrument count subset of LapGyn4 dataset",
        ref=REFERENCE + ADDITIONAL_REFERENCE,
        url="http://ftp.itec.aau.at/datasets/LapGyn4/",
        instr="download via ftp.itec.aau.at/datasets/LapGyn4/LapGyn4_v1.2.zip",
        lic=License.CC_BY_NC_4_0,
        release="v1.2",
        keywords=[
            Keyword.LAPAROSCOPY,
            Keyword.ENDOSCOPIC_INSTRUMENTS,
            Keyword.INSTRUMENT_COUNT,
            Keyword.MEDICAL,
            Keyword.GYNECOLOGY,
        ],
    )
    classes = ["0", "1", "2", "3"]
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "LapGyn4_v1.2" / "Instrument_Count", classes=classes
    )
    instrument_count.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    instrument_count.split_folds(n_folds=5, ensure_balancing=True)
    instrument_count.infer_stats()
    instrument_count.push_and_test()


@register_taskcreator(task_name="lapgyn4_anatomical_actions", dset_name="lapgyn4")
def create_task_aa(dset_path: Path):
    anatomical_actions = TaskCreator(
        dset_path=dset_path,
        name="lapgyn4_anatomical_actions",
        task_type=TaskType.CLASSIFICATION,
        desc="anatomical actions subset of LapGyn4 dataset",
        ref=REFERENCE,
        url="http://ftp.itec.aau.at/datasets/LapGyn4/",
        instr="download via ftp.itec.aau.at/datasets/LapGyn4/LapGyn4_v1.2.zip",
        lic=License.CC_BY_NC_4_0,
        release="v1.2",
        keywords=[Keyword.LAPAROSCOPY, Keyword.ANATOMICAL_STRUCTURES, Keyword.MEDICAL, Keyword.GYNECOLOGY],
    )
    classes = ["Suturing_Other", "Suturing_Ovary", "Suturing_Uterus", "Suturing_Vagina"]
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "LapGyn4_v1.2" / "Actions_on_Anatomy", classes=classes
    )
    anatomical_actions.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    anatomical_actions.split_folds(n_folds=5, ensure_balancing=True)
    anatomical_actions.infer_stats()
    anatomical_actions.push_and_test()
