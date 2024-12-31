# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import pandas as pd
import torch.nn.functional

from mml.core.data_loading.task_attributes import Keyword, License, Modality, RGBInfo, Sizes, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import create_creator_func, register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@article{Nwoye2022RendezvousAM,
  title={Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos},
  author={Chinedu Innocent Nwoye and Tong Yu and Cristians Gonzalez and Barbara Seeliger and Pietro Mascagni and Didier Mutter and Jacques Marescaux and Nicolas Padoy},
  journal={Medical image analysis},
  year={2022},
  volume={78},
  pages={102433}
}
"""  # noqa W291

dset_name = "cholect45"


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    instructions = f"""
                 Download link is not available. Please download the dataset by filling in the form at
                 https://github.com/CAMMA-public/cholect45 to download the "XXXX" folder. 
                 Once the download is complete place the downloaded folder 'XXX' in
                 <MML_DATA_ROOT>/DOWNLOADS/{dset_name}
            """  # noqa W291
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.verify_pre_download(
        file_name="CholecT45.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


def create_cholec_triplet_task(dset_path: Path, alias: str, target: str):
    task = TaskCreator(
        dset_path=dset_path,
        name=alias,
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        desc="https://cholectriplet2022.grand-challenge.org/",
        ref=REFERENCE,
        url="https://github.com/CAMMA-public/cholect45",
        instr="download via https://github.com/CAMMA-public/cholect45",
        lic=License.CC_BY_NC_SA_4_0,
        release="2022",
        keywords=[Keyword.MEDICAL, Keyword.ENDOSCOPY, Keyword.ENDOSCOPIC_INSTRUMENTS, Keyword.LAPAROSCOPY],
    )
    data_iterator = []
    with open(dset_path / DataKind.TRAINING_DATA / "CholecT45" / "dict" / f"{target}.txt", "r") as file:
        classes = [line[line.index(":") + 1 :].strip() for line in file.readlines()]
    idx_to_class = {classes.index(cl): cl for cl in classes}
    if alias.split("_")[-1] == "soft":
        # original: /dkfz/cluster/gpu/data/OE0176/m246i/cholec45/resized/CholecT45/dataframes/O_swin_bas_False_100_5_0_sd00_tar100.csv
        generated_soft_labels = pd.read_csv(dset_path / DataKind.TRAINING_DATA / "CholecT45" / "soft_labels.csv")
        cols = ["nid"] + [f"{ix}" for ix in range(100)]
        generated_soft_labels = generated_soft_labels[cols].set_index("nid")
        # drop the empty class, there is no prediction aka soft label for that
        del idx_to_class[len(classes) - 1]
        classes = classes[:-1]
        soft = True
    else:
        label_folder = dset_path / DataKind.TRAINING_DATA / "CholecT45" / f"{target}"
        soft = False
    fold_definition = [[] for _ in range(len(vid_folds))]
    for vid_folder in (dset_path / DataKind.TRAINING_DATA / "CholecT45" / "data").iterdir():
        if not soft:
            label_df = pd.read_csv(label_folder / f"{vid_folder.name}.txt", index_col=0, header=None)
        video_ids = []
        for frame in vid_folder.iterdir():
            video_ids.append(vid_folder.name + frame.stem)
            if not soft:
                frame_labels = label_df.loc[int(frame.stem)]
                frame_labels = frame_labels[frame_labels == 1].index.to_list()
                data_iterator.append(
                    {
                        Modality.SAMPLE_ID: vid_folder.name + frame.stem,
                        Modality.IMAGE: frame,
                        Modality.CLASSES: tuple(elem - 1 for elem in frame_labels),
                    }
                )
            else:
                soft_labels = generated_soft_labels.loc[f"{vid_folder.name}/{frame.stem}.png"].values
                soft_labels = tuple(torch.nn.functional.softmax(torch.tensor(soft_labels), dim=0).numpy().tolist())
                data_iterator.append(
                    {
                        Modality.SAMPLE_ID: vid_folder.name + frame.stem,
                        Modality.IMAGE: frame,
                        Modality.SOFT_CLASSES: soft_labels,
                    }
                )
        fold = [vid_folder.name in fold for fold in vid_folds].index(True)
        fold_definition[fold].extend(video_ids)
    task.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    task.use_existing_folds(fold_definition=fold_definition)
    task.set_stats(
        means=RGBInfo(*[0.33762359619140625, 0.21954330801963806, 0.21322497725486755]),
        sizes=Sizes(*[480, 1080, 854, 1920]),
        stds=RGBInfo(*[0.25544214248657227, 0.21357162296772003, 0.20953822135925293]),
    )
    task.push_and_test()


vid_folds = [
    ["VID79", "VID02", "VID51", "VID06", "VID25", "VID14", "VID66", "VID23", "VID50"],
    ["VID80", "VID32", "VID05", "VID15", "VID40", "VID47", "VID26", "VID48", "VID70"],
    ["VID31", "VID57", "VID36", "VID18", "VID52", "VID68", "VID10", "VID08", "VID73"],
    ["VID42", "VID29", "VID60", "VID27", "VID65", "VID75", "VID22", "VID49", "VID12"],
    ["VID78", "VID43", "VID62", "VID35", "VID74", "VID01", "VID56", "VID04", "VID13"],
]

TASKS = ["triplet", "instrument", "verb", "target", "triplet_soft"]

for task in TASKS:
    alias = f"{dset_name}_{task}"
    creator_func = create_creator_func(create_func=create_cholec_triplet_task, target=task.split("_")[0], alias=alias)
    register_taskcreator(task_name=alias, dset_name=dset_name)(creator_func)
