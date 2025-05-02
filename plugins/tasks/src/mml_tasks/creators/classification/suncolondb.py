# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path
from typing import Dict

from mml.core.data_loading.task_attributes import Keyword, License, Modality, RGBInfo, Sizes, TaskType
from mml.core.data_preparation.archive_extractors import ask_password
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

logger = logging.getLogger(__name__)

REFERENCE = """
@article{MISAWA2021960,
title = {Development of a computer-aided detection system for colonoscopy and a publicly accessible large colonoscopy video database (with video)},
journal = {Gastrointestinal Endoscopy},
volume = {93},
number = {4},
pages = {960-967.e3},
year = {2021},
issn = {0016-5107},
doi = {https://doi.org/10.1016/j.gie.2020.07.060},
url = {https://www.sciencedirect.com/science/article/pii/S0016510720346551},
author = {Masashi Misawa and Shin-ei Kudo and Yuichi Mori and Kinichi Hotta and Kazuo Ohtsuka
and Takahisa Matsuda and Shoichi Saito and Toyoki Kudo and Toshiyuki Baba and Fumio Ishida and Hayato Itoh and Masahiro Oda and Kensaku Mori},
}
"""

dset_name = "SUNdatabase"
ALL_TASKS = ["polyp", "no_polyp"]

instructions = f"""
Download link is not available. Please request access to the dataset on http://amed8k.sundatabase.org/,
afterwards download the six .zip files and place them in the <MML_DATA_ROOT>/DOWNLOADS/{dset_name}.

"""


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    """Creates dataset

    :param dset_name: Name of the dataset
    :type dset_name: string
    :return: Path to dataset directory
    :rtype: PosixPath
    """
    dset_creator = DSetCreator(dset_name=dset_name)
    file_list = [
        "sundatabase_positive_part1.zip",
        "sundatabase_positive_part2.zip",
        "sundatabase_negative_part1.zip",
        "sundatabase_negative_part2.zip",
        "sundatabase_negative_part3.zip",
        "sundatabase_negative_part4.zip",
    ]
    pwd = ask_password()  # same password for all archives
    for file_name in file_list:
        archive = dset_creator.verify_pre_download(
            file_name=file_name, data_kind=DataKind.TRAINING_DATA, instructions=instructions
        )
        archive.password = pwd
        archive.keep_top_level = True
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="suncolondb-classification", dset_name=dset_name)
def create_task(dset_path: Path) -> None:
    """Creates task for suncolondb classification

    :param dset_path: Path to dataset
    :type dset_path: Path
    """
    task_creator = TaskCreator(
        dset_path=dset_path,
        name="suncolondb-classification",
        task_type=TaskType.CLASSIFICATION,
        desc="The SUN database includes 49,136 polyp frames taken from "
        "different 100 polyps, which were fully annotated with bounding "
        "boxes. Non-polyp scenes of 109,554 frames are also included "
        "in this database. In polyp-existing frames, each polyp is "
        "annotated with a bounding box. The file formats of images, "
        "and bounding boxes are jpeg and a text file, respectively. "
        "In the text file, each row represents a bounding box of a "
        "polyp, that is, "
        '"Filename min_Xcoordinate,min_Ycoordinate,max_Xcorrdinate,max_Ycoordinate,class_id", '
        "where class_id of 0 and 1 represent polyp and non-polyp frames, "
        "respectively.",
        ref=REFERENCE,
        url="http://amed8k.sundatabase.org/",
        instr="request access via http://amed8k.sundatabase.org/",
        lic=License.CUSTOM,
        release="2020",
        keywords=[Keyword.MEDICAL, Keyword.ENDOSCOPY, Keyword.TISSUE_PATHOLOGY],
    )

    # the subdirectories where the negative cases are stored
    data_parts_negative = [
        "sundatabase_" + var for var in ["negative_part1", "negative_part2", "negative_part3", "negative_part4"]
    ]
    # the subdirectoreis where the positive cases are stored
    data_parts_positive = ["sundatabase_" + var for var in ["positive_part1", "positive_part2"]]

    case_paths_neg = get_case_paths(data_parts_negative, dset_path, ispos=False)
    case_paths_pos = get_case_paths(data_parts_positive, dset_path, ispos=True)
    case_paths = case_paths_neg + case_paths_pos

    # load annotations
    all_annotations = load_annotations(dset_path)

    data_iterators = get_data_iterators(case_paths, all_annotations)

    idx_to_class = {0: "not_present", 1: "present"}
    task_creator.find_data(
        train_iterator=data_iterators["train"], test_iterator=data_iterators["test"], idx_to_class=idx_to_class
    )
    task_creator.split_folds(n_folds=5, ensure_balancing=True)
    # stats are identical across tasks, save calculation time here
    task_creator.set_stats(
        means=RGBInfo(*[0.5652844309806824, 0.32137545943260193, 0.23057971894741058]),
        stds=RGBInfo(*[0.2662612497806549, 0.21721985936164856, 0.16612227261066437]),
        sizes=Sizes(*[1008, 1080, 1158, 1240]),
    )
    task_creator.push_and_test()
    pass


def get_case_paths(data_parts, dset_path, ispos):
    """Returns list with paths of cases in subdirectories as well as whether they are positive or negative

    :param data_parts: list of directories where cases are saved
    :type data_parts: list of strings
    :param dset_path: path to dataset directory
    :type dset_path: PosixPath
    :param ispos: Whether the data_parts contain the positive samples
    :type ispos: bool
    :return: List of paths to all the cases in dataset, along with whether they are positive or negative
    :rtype: list of [PosixPath, bool]
    """
    case_paths = []
    for data_part in data_parts:
        data_part_path = dset_path / "training_data" / data_part
        for subdir in Path(data_part_path).iterdir():
            if subdir.is_dir() and subdir.name.startswith("case"):  # exclude annotations
                case_paths.append([subdir, ispos])
    return case_paths


def load_annotations(dset_path: Path) -> Dict[str, Dict[str, int]]:
    """Load annotations and stores them in dict

    :param dset_path: path to directory
    :type dset_path: PosixPath
    :return: Dict contating one dict per video containing class values per frame
    :rtype: Dict of dicts of ints
    """
    all_annotations = {}
    annotation_dir = dset_path / "training_data" / "sundatabase_positive_part1" / "annotation_txt"
    anno_file_list = [file for file in Path(annotation_dir).iterdir() if file.is_file()]
    for anno_file_name in anno_file_list:
        case = anno_file_name.stem
        all_annotations[case] = {}
        anno_file_path = Path(annotation_dir) / anno_file_name
        with open(anno_file_path, "r") as anno_file:
            # line format: 'filename min_Xcoordinate,min_Ycoordinate, max_Xcoordinate, max_Ycoordinate, class_id'
            for line in anno_file:
                line_elements = line.strip().split(",")
                file_name = line_elements[0].split(" ")[0]
                if not file_name.endswith(".jpg"):
                    raise ValueError(f"File name {file_name} does not end with '.jpg'")
                class_val = int(line_elements[-1])
                if class_val not in [0, 1]:
                    raise ValueError(f"Class value {class_val} has to be either 0 or 1.")
                all_annotations[case][file_name] = class_val
    return all_annotations


def get_data_iterators(case_paths, all_annotations):
    """Splits the dataset into train and test and generates data iterators

    :param case_paths: list of paths and whether they are from positive or negative samples
    :type case_paths: list of [PosixPath, bool]
    :param all_annotations: Dict contating one dict per video containing class values per frame
    :type all_annotations: Dict of dicts of ints
    :return: Dict containing train and test sets, containing a list of identifier, path and class
    :rtype: dict of list of dict
    """
    # splits data along videos
    vids_split = {"train": case_paths[:68], "test": case_paths[68:]}
    data_iterators = {k: [] for k in vids_split}
    for phase in vids_split:  # loops over train/test
        for vid in vids_split[phase]:  # loops over videos
            if vid[1] == 0:  # videos without polyps
                for frame in vid[0].iterdir():  # loops over frames
                    data_iterators[phase].append(
                        {
                            Modality.SAMPLE_ID: f"neg_vid{vid[0].stem}_frame{frame.stem.split('_')[-3]}_{frame.stem[-6:]}",
                            Modality.IMAGE: frame,
                            Modality.CLASS: 0,
                        }
                    )
            else:  # videos with polyps
                for frame in vid[0].iterdir():  # loops over frames
                    no_polyp = all_annotations[str(vid[0]).split("/")[-1]][frame.stem + ".jpg"]
                    if not no_polyp:  # check whether frame contains polyp, if not, ignore
                        data_iterators[phase].append(
                            {
                                Modality.SAMPLE_ID: f"""pos_vid{vid[0].stem}_frame{frame.stem.split("_")[-4]}
                                                      _{frame.stem.split("_")[-3]}_{frame.stem[-4:]}""",
                                Modality.IMAGE: frame,
                                Modality.CLASS: 1,
                            }
                        )
    return data_iterators
