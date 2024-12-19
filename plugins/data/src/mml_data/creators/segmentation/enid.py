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

REFERENCE = """"
@article{DBLP:conf/mta/LeibetsederS21,
    author    = {Andreas Leibetseder and
                    Klaus Schoeffmann and
                    J{\"{o}}rg Keckstein and
                    Simon Keckstein},
    title     = {Endometriosis Detection and Localization in Laparoscopic Gynecology},
}
"""

dset_name = "enid"
class_mapping = {
    (0, 0, 0): "background",  # Background/No Pathology
    (204, 81, 81): "Endo-Peritoneum",  # endometriosis
}
# these values are borderline pixels, not directly mappable to one of the classes, they will be ignored
wrong = [
    [1, 0, 0],
    [2, 1, 1],
    [3, 1, 1],
    [4, 2, 2],
    [5, 2, 2],
    [6, 2, 2],
    [6, 3, 3],
    [7, 3, 3],
    [8, 3, 3],
    [9, 3, 3],
    [10, 4, 4],
    [11, 4, 4],
    [12, 5, 5],
    [13, 5, 5],
    [14, 5, 5],
    [14, 6, 6],
    [15, 6, 6],
    [16, 6, 6],
    [17, 7, 7],
    [18, 7, 7],
    [19, 8, 8],
    [20, 8, 8],
    [21, 8, 8],
    [22, 9, 9],
    [23, 9, 9],
    [24, 9, 9],
    [25, 10, 10],
    [26, 10, 10],
    [27, 11, 11],
    [28, 11, 11],
    [29, 11, 11],
    [29, 12, 12],
    [30, 12, 12],
    [31, 12, 12],
    [32, 13, 13],
    [33, 13, 13],
    [34, 14, 14],
    [35, 14, 14],
    [36, 14, 14],
    [37, 15, 15],
    [38, 15, 15],
    [39, 16, 16],
    [40, 16, 16],
    [41, 16, 16],
    [42, 17, 17],
    [43, 17, 17],
    [44, 17, 17],
    [45, 18, 18],
    [46, 18, 18],
    [47, 19, 19],
    [48, 19, 19],
    [49, 19, 19],
    [49, 20, 20],
    [50, 20, 20],
    [51, 20, 20],
    [52, 21, 21],
    [53, 21, 21],
    [54, 22, 22],
    [55, 22, 22],
    [56, 22, 22],
    [57, 22, 22],
    [57, 23, 23],
    [58, 23, 23],
    [59, 23, 23],
    [60, 24, 24],
    [61, 24, 24],
    [62, 25, 25],
    [63, 25, 25],
    [64, 25, 25],
    [65, 26, 26],
    [66, 26, 26],
    [67, 27, 27],
    [68, 27, 27],
    [69, 27, 27],
    [69, 28, 28],
    [70, 28, 28],
    [71, 28, 28],
    [72, 28, 28],
    [73, 29, 29],
    [74, 29, 29],
    [75, 30, 30],
    [76, 30, 30],
    [77, 30, 30],
    [77, 31, 31],
    [78, 31, 31],
    [79, 31, 31],
    [80, 32, 32],
    [81, 32, 32],
    [82, 33, 33],
    [83, 33, 33],
    [84, 33, 33],
    [84, 34, 34],
    [85, 34, 34],
    [86, 34, 34],
    [87, 34, 34],
    [88, 35, 35],
    [89, 35, 35],
    [90, 36, 36],
    [91, 36, 36],
    [92, 36, 36],
    [92, 37, 37],
    [93, 37, 37],
    [94, 37, 37],
    [95, 38, 38],
    [96, 38, 38],
    [97, 39, 39],
    [98, 39, 39],
    [99, 39, 39],
    [100, 40, 40],
    [101, 40, 40],
    [103, 41, 41],
    [104, 41, 41],
    [105, 42, 42],
    [106, 42, 42],
    [107, 42, 42],
    [108, 43, 43],
    [109, 43, 43],
    [110, 44, 44],
    [111, 44, 44],
    [112, 44, 44],
    [112, 45, 45],
    [113, 45, 45],
    [114, 45, 45],
    [115, 46, 46],
    [116, 46, 46],
    [117, 47, 47],
    [118, 47, 47],
    [119, 47, 47],
    [120, 47, 47],
    [120, 48, 48],
    [121, 48, 48],
    [122, 48, 48],
    [123, 49, 49],
    [124, 49, 49],
    [125, 50, 50],
    [126, 50, 50],
    [127, 50, 50],
    [128, 51, 51],
    [129, 51, 51],
    [130, 52, 52],
    [131, 52, 52],
    [132, 53, 53],
    [133, 53, 53],
    [134, 53, 53],
    [135, 53, 53],
    [135, 54, 54],
    [136, 54, 54],
    [137, 54, 54],
    [138, 55, 55],
    [139, 55, 55],
    [140, 56, 56],
    [141, 56, 56],
    [142, 56, 56],
    [143, 57, 57],
    [144, 57, 57],
    [145, 58, 58],
    [146, 58, 58],
    [147, 58, 58],
    [147, 59, 59],
    [148, 59, 59],
    [149, 59, 59],
    [150, 59, 59],
    [151, 60, 60],
    [152, 60, 60],
    [153, 61, 61],
    [154, 61, 61],
    [155, 61, 61],
    [155, 62, 62],
    [156, 62, 62],
    [157, 62, 62],
    [158, 63, 63],
    [159, 63, 63],
    [160, 64, 64],
    [161, 64, 64],
    [162, 64, 64],
    [163, 65, 65],
    [164, 65, 65],
    [165, 65, 65],
    [165, 66, 66],
    [166, 66, 66],
    [167, 66, 66],
    [168, 67, 67],
    [169, 67, 67],
    [170, 67, 67],
    [170, 68, 68],
    [171, 68, 68],
    [172, 68, 68],
    [172, 69, 69],
    [173, 69, 69],
    [174, 69, 69],
    [175, 69, 69],
    [175, 70, 70],
    [176, 70, 70],
    [177, 70, 70],
    [178, 71, 71],
    [179, 71, 71],
    [180, 71, 71],
    [180, 72, 72],
    [181, 72, 72],
    [182, 72, 72],
    [183, 72, 72],
    [183, 73, 73],
    [184, 73, 73],
    [185, 73, 73],
    [185, 74, 74],
    [186, 74, 74],
    [187, 74, 74],
    [188, 74, 74],
    [188, 75, 75],
    [189, 75, 75],
    [190, 75, 75],
    [190, 76, 76],
    [191, 76, 76],
    [192, 76, 76],
    [193, 76, 76],
    [193, 77, 77],
    [194, 77, 77],
    [195, 77, 77],
    [195, 78, 78],
    [196, 78, 78],
    [197, 78, 78],
    [198, 78, 78],
    [198, 79, 79],
    [199, 79, 79],
    [200, 79, 79],
    [200, 80, 80],
    [201, 80, 80],
    [202, 80, 80],
    [203, 80, 80],
    [203, 81, 81],
]


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.download(
        url="http://ftp.itec.aau.at/datasets/ENID/downloads/datasets/ENID_v1.0_dataset.zip",
        file_name="ENID_v1.0_dataset.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    masks = list((dset_path / DataKind.TRAINING_DATA / "ENID_v1.0_dataset" / "annots").rglob("*.png"))
    keys = list(class_mapping.keys())
    out_base = dset_creator.transform_masks(
        masks=masks,
        load="rgb",
        transform={pix_val: idx for idx, pix_val in enumerate(keys)},
        train=True,
        ignore=[tuple(ix) for ix in wrong],
    )

    assert out_base == dset_path / DataKind.TRAINING_LABELS / "transformed_masks", f"{out_base=}"
    return dset_path


@register_taskcreator(task_name="endometrial_implants", dset_name=dset_name)
def create_task_enid(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name="endometrial_implants",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="The dataset includes region-based annotations of a specific visual endometriosis "
        "manifestation: dark endometrial implants. ",
        ref=REFERENCE,
        url="http://ftp.itec.aau.at/datasets/ENID",
        instr="download data ENID_v1.0_dataset via website",
        lic=License.CC_BY_NC_4_0,
        release="2021",
        keywords=[Keyword.LAPAROSCOPY, Keyword.MEDICAL, Keyword.GYNECOLOGY],
    )
    data_iterator = []
    for img_id in [
        p.stem
        for p in (dset_path / DataKind.TRAINING_DATA / "ENID_v1.0_dataset" / "frames").iterdir()
        if p.is_file() and p.suffix == ".jpg"
    ]:
        data_iterator.append(
            {
                Modality.SAMPLE_ID: img_id,
                Modality.IMAGE: dset_path / DataKind.TRAINING_DATA / "ENID_v1.0_dataset" / "frames" / f"{img_id}.jpg",
                Modality.MASK: (
                    dset_path
                    / DataKind.TRAINING_LABELS
                    / "transformed_masks"
                    / "ENID_v1.0_dataset"
                    / "annots"
                    / f"{img_id}.png"
                ),
            }
        )
    task.find_data(
        train_iterator=data_iterator, idx_to_class={ix: cls for ix, cls in enumerate(class_mapping.values())}
    )
    task.split_folds(n_folds=5)
    task.infer_stats()
    task.push_and_test()
