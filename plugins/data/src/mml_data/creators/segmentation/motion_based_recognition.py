# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

# creates the "endocsopic instrument segmentation with crowdsourced data" dataset and tasks
from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@inproceedings{ BrostowSFC:ECCV08,
  author    = {Gabriel J. Brostow and Jamie Shotton and Julien Fauqueur and Roberto Cipolla},
  title     = {Segmentation and Recognition Using Structure from Motion Point Clouds},
  booktitle = {ECCV (1)},
  year      = {2008},
  pages     = {44-57}
}

@article{ BrostowFC:PRL2008,
    author = "Gabriel J. Brostow and Julien Fauqueur and Roberto Cipolla",
    title = "Semantic Object Classes in Video: A High-Definition Ground Truth Database",
    journal = "Pattern Recognition Letters",
    volume = "xx",
    number = "x",   
    pages = "xx-xx",
    year = "2008"
}
"""  # noqa W291

dset_name = "motion-based-rec"
class_mapping = {
    (0, 0, 0): "Void",  # make sure background is first
    (64, 128, 64): "Animal",
    (192, 0, 128): "Archway",
    (0, 128, 192): "Bicyclist",
    (0, 128, 64): "Bridge",
    (128, 0, 0): "Building",
    (64, 0, 128): "Car",
    (64, 0, 192): "CartLuggagePram",
    (192, 128, 64): "Child",
    (192, 192, 128): "Column_Pole",
    (64, 64, 128): "Fence",
    (128, 0, 192): "LaneMkgsDriv",
    (192, 0, 64): "LaneMkgsNonDriv",
    (128, 128, 64): "Misc_Text",
    (192, 0, 192): "MotorcycleScooter",
    (128, 64, 64): "OtherMoving",
    (64, 192, 128): "ParkingBlock",
    (64, 64, 0): "Pedestrian",
    (128, 64, 128): "Road",
    (128, 128, 192): "RoadShoulder",
    (0, 0, 192): "Sidewalk",
    (192, 128, 128): "SignSymbol",
    (128, 128, 128): "Sky",
    (64, 128, 192): "SUVPickupTruck",
    (0, 0, 64): "TrafficCone",
    (0, 64, 64): "TrafficLight",
    (192, 64, 128): "Train",
    (128, 128, 0): "Tree",
    (192, 128, 192): "Truck_Bus",
    (64, 0, 64): "Tunnel",
    (192, 192, 0): "VegetationMisc",
    (64, 192, 0): "Wall",
}


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.download(
        url="http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip",
        file_name="701_StillsRaw_full.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_creator.download(
        url="http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip",
        file_name="LabeledApproved_full.zip",
        data_kind=DataKind.TRAINING_LABELS,
    )
    dset_path = dset_creator.unpack_and_store()

    masks = list((dset_path / DataKind.TRAINING_LABELS).rglob("*_L.png"))
    masks.remove(dset_path / DataKind.TRAINING_LABELS / "Seq05VD_f02610_L.png")  # this file contains invalid pixels
    out_base = dset_creator.transform_masks(
        masks=masks,
        load="rgb",
        transform={pix_val: idx for idx, pix_val in enumerate(class_mapping.keys())},
        train=True,
    )
    assert out_base == dset_path / DataKind.TRAINING_LABELS / "transformed_masks", f"{out_base=}"
    return dset_path


@register_taskcreator(task_name="motion-based-segmentation", dset_name=dset_name)
def create_task(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name="motion-based-segmentation",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="segmenting 32 classes in driving scenes",
        ref=REFERENCE,
        url="http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/",
        instr="download zips of extracted images and painted class labels via website",
        lic=License.UNKNOWN,
        release="2008",
        keywords=[Keyword.SCENES, Keyword.NATURAL_OBJECTS, Keyword.DRIVING],
    )
    data_iterator = []
    for img_id in [
        p.stem
        for p in (dset_path / DataKind.TRAINING_DATA / "701_StillsRaw_full").iterdir()
        if p.is_file() and p.suffix == ".png"
    ]:
        if img_id == "Seq05VD_f02610":
            # this is the invalid mask from above
            continue
        data_iterator.append(
            {
                Modality.SAMPLE_ID: img_id,
                Modality.IMAGE: dset_path / DataKind.TRAINING_DATA / "701_StillsRaw_full" / f"{img_id}.png",
                Modality.MASK: dset_path / DataKind.TRAINING_LABELS / "transformed_masks" / f"{img_id}_L.png",
            }
        )
    task.find_data(
        train_iterator=data_iterator, idx_to_class={ix: cls for ix, cls in enumerate(class_mapping.values())}
    )
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
