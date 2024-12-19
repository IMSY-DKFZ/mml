# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import shutil
from itertools import chain
from pathlib import Path
from typing import List

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@misc{allan20202018,
      title={2018 Robotic Scene Segmentation Challenge},
      author={Max Allan and Satoshi Kondo and Sebastian Bodenstedt and Stefan Leger and Rahim Kadkhodamohammadi and 
      Imanol Luengo and Felix Fuentes and Evangello Flouty and Ahmed Mohammed and Marius Pedersen and Avinash Kori 
      and Varghese Alex and Ganapathy Krishnamurthi and David Rauber and Robert Mendel and Christoph Palm and Sophia 
      Bano and Guinther Saibro and Chi-Sheng Shih and Hsun-An Chiang and Juntang Zhuang and Junlin Yang and Vladimir
      Iglovikov and Anton Dobrenkii and Madhu Reddiboina and Anubhav Reddy and Xingtong Liu and Cong Gao and Mathias 
      Unberath and Myeonghyeon Kim and Chanho Kim and Chaewon Kim and Hyejin Kim and Gyeongmin Lee and Ihsan Ullah and 
      Miguel Luna and Sang Hyun Park and Mahdi Azizian and Danail Stoyanov and Lena Maier-Hein and Stefanie Speidel},
      year={2020},
      eprint={2001.11190},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""  # noqa W291

dset_name = "endovis18_rob_instr"
class_mapping = {
    (0, 0, 0): "background-tissue",
    (0, 255, 0): "instrument-shaft",
    (0, 255, 255): "instrument-clasper",
    (125, 255, 12): "instrument-wrist",
    (255, 55, 0): "kidney-parenchyma",
    (24, 55, 125): "covered-kidney",
    (187, 155, 25): "thread",
    (0, 255, 125): "clamps",
    (255, 255, 125): "suturing-needle",
    (123, 15, 175): "suction-instrument",
    (124, 155, 5): "small-intestine",
    (12, 255, 141): "ultrasound-probe",
}


def get_sequences(dset_path: Path) -> List[Path]:
    """
    Returns all sequence folder paths.

    :param dset_path:
    :return:
    """
    releases = [
        "miccai_challenge_2018_release_1",
        "miccai_challenge_release_2",
        "miccai_challenge_release_3",
        "miccai_challenge_release_4",
    ]
    return [p for rel in releases for p in (dset_path / DataKind.TRAINING_DATA / rel).iterdir() if p.is_dir()]


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    instructions = """"
     Download link is not available. Please download the dataset by clicking on the download buttons manually via
     https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/
     Download the three files under Training data release 1 and the two files under Training data release 2
     Once the download is complete place the 5 downloaded files 'miccai_challenge_2018_release_1.zip',
     'repairs.zip', 'miccai_challenge_release_2.zip', 'miccai_challenge_release_3.zip' and
     'miccai_challenge_release_4.zip' in <MML_DATA_ROOT>\\DOWNLOADS\\endovis18_rob_instr
    """
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.verify_pre_download(
        file_name="miccai_challenge_2018_release_1.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_creator.verify_pre_download(
        file_name="repairs.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_creator.verify_pre_download(
        file_name="miccai_challenge_release_2.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_creator.verify_pre_download(
        file_name="miccai_challenge_release_3.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_creator.verify_pre_download(
        file_name="miccai_challenge_release_4.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    # replace repaired image masks for sequences 1 and 4 of first release
    repair_root = dset_path / DataKind.TRAINING_DATA / "repairs"
    for file in repair_root.iterdir():
        seq = file.name.split("_")[1]
        name = file.name[6:]
        target = (
            dset_path / DataKind.TRAINING_DATA / "miccai_challenge_2018_release_1" / f"seq_{seq}" / "labels" / f"{name}"
        )
        shutil.move(file, target)
    # gather all masks and transform them accordingly
    masks = list(chain(*[(seq / "labels").rglob("*.png") for seq in get_sequences(dset_path)]))
    dset_creator.transform_masks(
        masks=masks,
        load="rgb",
        transform={pix_val: idx for idx, pix_val in enumerate(class_mapping.keys())},
        train=True,
    )
    return dset_path


@register_taskcreator(task_name="endovissub18_robotic_instrument_seg", dset_name=dset_name)
def create_task_glenda(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name="endovissub18_robotic_instrument_seg",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="The dataset includes pixel wise segmentation of robotic instruments as well as anatomical "
        "objects and non-robotic surgical instruments such as suturing needles and gauze",
        ref=REFERENCE,
        url="https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/",
        instr="download the data using the 5 links via website",
        lic=License.UNKNOWN,
        release="2018",
        keywords=[Keyword.MEDICAL, Keyword.NEPHRECTOMY, Keyword.ENDOSCOPIC_INSTRUMENTS, Keyword.ANATOMICAL_STRUCTURES],
    )
    data_iterator = []
    for seq in get_sequences(dset_path):
        for img in (seq / "left_frames").iterdir():
            if not (img.is_file() and img.suffix == ".png"):
                continue
            data_iterator.append(
                {
                    Modality.SAMPLE_ID: f"seq_{seq.name.split('_')[-1]}_{img.stem}",
                    Modality.IMAGE: img,
                    Modality.MASK: (
                        dset_path
                        / DataKind.TRAINING_LABELS
                        / "transformed_masks"
                        / seq.parent.name
                        / seq.name
                        / "labels"
                        / f"{img.name}"
                    ),
                }
            )
    task.find_data(
        train_iterator=data_iterator, idx_to_class={ix: cls for ix, cls in enumerate(class_mapping.values())}
    )
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
