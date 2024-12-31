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
@inbook{inbook,
author = {Mendonça, Teresa and Ferreira, Pedro and Marçal, André and Barata, Catarina and Marques, Jorge and Rocha, Joana and Rozeira, Jorge},
year = {2015},
month = {09},
pages = {419-439},
title = {PH2: A Public Database for the Analysis of Dermoscopic Images},
isbn = {978-1-4822-5326-9},
doi = {10.1201/b19107-14}
}
}
"""

dataset_name = "PH2"


def get_idx_to_class():
    idx_to_class = {0: "Background", 1: "Lesion"}
    return idx_to_class


@register_dsetcreator(dset_name=dataset_name)
def create_dset():
    instructions = f"""
                 Download link is not available. Please download the dataset by clicking on the download button manually via
                 https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar
                 Once the download is complete place the downloaded folder 'PH2Dataset.rar' in
                 <MML_DATA_ROOT>/DOWNLOADS/{dataset_name}
            """
    dset_creator = DSetCreator(dset_name=dataset_name)
    dset_creator.verify_pre_download(
        file_name="PH2Dataset.rar", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="ph2-melanocytic-lesions-segmentation", dset_name=dataset_name)
def create_seg_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name="ph2-melanocytic-lesions-segmentation",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="This image database contains dermoscopic images of melanocytic lesions",
        ref=REFERENCE,
        url="https://www.fc.up.pt/addi/ph2%20database.html",
        instr="download data PH2Dataset via website",
        lic=License.UNKNOWN,
        release="2015",
        keywords=[Keyword.DERMATOSCOPY, Keyword.MEDICAL, Keyword.TISSUE_PATHOLOGY],
    )
    data_iterator = []
    training_data_root = dset_path / DataKind.TRAINING_DATA / "PH2Dataset" / "PH2 Dataset images"
    for image in training_data_root.iterdir():
        img_name = image.name + "_Dermoscopic_Image"
        mask_name = image.name + "_lesion"
        data_iterator.append(
            {
                Modality.SAMPLE_ID: image.name,
                Modality.IMAGE: training_data_root / f"{image.name}" / f"{img_name}" / f"{image.name}.bmp",
                Modality.MASK: training_data_root / f"{image.name}" / f"{mask_name}" / f"{mask_name}.bmp",
            }
        )
    task.find_data(train_iterator=data_iterator, idx_to_class=get_idx_to_class())
    task.split_folds(n_folds=5)
    task.infer_stats()
    task.push_and_test()


MELANOMA = [
    "IMD058",
    "IMD061",
    "IMD063",
    "IMD064",
    "IMD065",
    "IMD080",
    "IMD085",
    "IMD088",
    "IMD090",
    "IMD091",
    "IMD168",
    "IMD211",
    "IMD219",
    "IMD240",
    "IMD242",
    "IMD284",
    "IMD285",
    "IMD348",
    "IMD349",
    "IMD403",
    "IMD404",
    "IMD405",
    "IMD407",
    "IMD408",
    "IMD409",
    "IMD410",
    "IMD413",
    "IMD417",
    "IMD418",
    "IMD419",
    "IMD406",
    "IMD411",
    "IMD420",
    "IMD421",
    "IMD423",
    "IMD424",
    "IMD425",
    "IMD426",
    "IMD429",
    "IMD435",
]
COMMON_NEVUS = [
    "IMD003",
    "IMD009",
    "IMD016",
    "IMD022",
    "IMD024",
    "IMD025",
    "IMD035",
    "IMD038",
    "IMD042",
    "IMD044",
    "IMD045",
    "IMD050",
    "IMD092",
    "IMD101",
    "IMD103",
    "IMD112",
    "IMD118",
    "IMD125",
    "IMD132",
    "IMD134",
    "IMD135",
    "IMD144",
    "IMD146",
    "IMD147",
    "IMD150",
    "IMD152",
    "IMD156",
    "IMD159",
    "IMD161",
    "IMD162",
    "IMD175",
    "IMD177",
    "IMD182",
    "IMD198",
    "IMD200",
    "IMD010",
    "IMD017",
    "IMD020",
    "IMD039",
    "IMD041",
    "IMD105",
    "IMD107",
    "IMD108",
    "IMD133",
    "IMD142",
    "IMD143",
    "IMD160",
    "IMD173",
    "IMD176",
    "IMD196",
    "IMD197",
    "IMD199",
    "IMD203",
    "IMD204",
    "IMD206",
    "IMD207",
    "IMD208",
    "IMD364",
    "IMD365",
    "IMD367",
    "IMD371",
    "IMD372",
    "IMD374",
    "IMD375",
    "IMD378",
    "IMD379",
    "IMD380",
    "IMD381",
    "IMD383",
    "IMD384",
    "IMD385",
    "IMD389",
    "IMD390",
    "IMD392",
    "IMD394",
    "IMD395",
    "IMD397",
    "IMD399",
    "IMD400",
    "IMD402",
]
ATYPICAL_NEVUS = [
    "IMD002",
    "IMD004",
    "IMD013",
    "IMD015",
    "IMD019",
    "IMD021",
    "IMD027",
    "IMD030",
    "IMD032",
    "IMD033",
    "IMD037",
    "IMD040",
    "IMD043",
    "IMD047",
    "IMD048",
    "IMD049",
    "IMD057",
    "IMD075",
    "IMD076",
    "IMD078",
    "IMD120",
    "IMD126",
    "IMD137",
    "IMD138",
    "IMD139",
    "IMD140",
    "IMD149",
    "IMD153",
    "IMD157",
    "IMD164",
    "IMD166",
    "IMD169",
    "IMD171",
    "IMD210",
    "IMD347",
    "IMD155",
    "IMD376",
    "IMD006",
    "IMD008",
    "IMD014",
    "IMD018",
    "IMD023",
    "IMD031",
    "IMD036",
    "IMD154",
    "IMD170",
    "IMD226",
    "IMD243",
    "IMD251",
    "IMD254",
    "IMD256",
    "IMD278",
    "IMD279",
    "IMD280",
    "IMD304",
    "IMD305",
    "IMD306",
    "IMD312",
    "IMD328",
    "IMD331",
    "IMD339",
    "IMD356",
    "IMD360",
    "IMD368",
    "IMD369",
    "IMD370",
    "IMD382",
    "IMD386",
    "IMD388",
    "IMD393",
    "IMD396",
    "IMD398",
    "IMD427",
    "IMD430",
    "IMD431",
    "IMD432",
    "IMD433",
    "IMD434",
    "IMD436",
    "IMD437",
]


@register_taskcreator(task_name="ph2-melanocytic-lesions-classification", dset_name=dataset_name)
def create_clas_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name="ph2-melanocytic-lesions-classification",
        task_type=TaskType.CLASSIFICATION,
        desc="This image database contains dermoscopic images of melanocytic lesions",
        ref=REFERENCE,
        url="https://www.fc.up.pt/addi/ph2%20database.html",
        instr="download data PH2Dataset via website",
        lic=License.UNKNOWN,
        release="2015",
        keywords=[Keyword.DERMATOSCOPY, Keyword.MEDICAL, Keyword.TISSUE_PATHOLOGY],
    )
    data_iterator = []
    training_data_root = dset_path / DataKind.TRAINING_DATA / "PH2Dataset" / "PH2 Dataset images"
    idx_to_class = {0: "melanoma", 1: "common nevus", 2: "atypical nevus"}
    for diagnosis, samples in zip([0, 1, 2], [MELANOMA, COMMON_NEVUS, ATYPICAL_NEVUS]):
        for sample in samples:
            img_folder = sample + "_Dermoscopic_Image"
            data_iterator.append(
                {
                    Modality.SAMPLE_ID: sample,
                    Modality.IMAGE: training_data_root / f"{sample}" / f"{img_folder}" / f"{sample}.bmp",
                    Modality.CLASS: diagnosis,
                }
            )
    task.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    task.split_folds(n_folds=5)
    task.infer_stats()
    task.push_and_test()
