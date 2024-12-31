# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

import cv2
import pandas as pd

from mml.core.data_loading.task_attributes import Keyword, License, Modality, RGBInfo, Sizes, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import create_creator_func, register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

logger = logging.getLogger(__name__)

REFERENCE = """
@article{Twinanda2017EndoNetAD,
  title={EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos},
  author={Andru Putra Twinanda and S. Shehata and Didier Mutter and Jacques Marescaux and Michel de Mathelin and Nicolas Padoy},
  journal={IEEE Transactions on Medical Imaging},
  year={2017},
  volume={36},
  pages={86-97}
}
"""

dset_name = "cholec80"
ALL_TASKS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
# be aware grasper and hook are quite common ~ 55% of frames but all others are roughly between 1% and 6%

instructions = f"""
Download link is not available. Please download the dataset by registering via the request form at
http://camma.u-strasbg.fr/datasets, afterwards download the download the "cholec80.zip" folder. 
Once the download is complete place the downloaded folder 'cholec80.zip' in
<MML_DATA_ROOT>/DOWNLOADS/{dset_name}
"""  # noqa W291


@register_dsetcreator(dset_name=dset_name)
def create_cholec80():
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.verify_pre_download(
        file_name="cholec80.zip", data_kind=DataKind.TRAINING_DATA, instructions=instructions
    )
    dset_path = dset_creator.unpack_and_store()
    # now videos have to be processed, images are extracted according to 1fps annotation
    video_root = dset_path / DataKind.TRAINING_DATA / "videos"
    image_root = dset_path / DataKind.TRAINING_DATA / "images"
    image_root.mkdir(exist_ok=True)
    video_paths = sorted([p for p in video_root.iterdir() if p.suffix == ".mp4"])
    for ix, vid in enumerate(video_paths):
        logger.info(f"Starting frame extraction of {vid.name}... ({ix + 1}/{len(video_paths)})")
        target = image_root / vid.stem
        target.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(vid))
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            if counter % 25 == 0:
                cv2.imwrite(str(target / f"{counter}.jpg"), frame)
            counter += 1
        cap.release()
    logger.info("Done extracting! Now checking with annotations...")
    for ix, vid_imgs in enumerate(image_root.iterdir()):
        frames = [int(img.stem) for img in vid_imgs.iterdir()]
        annotation_path = dset_path / DataKind.TRAINING_DATA / "tool_annotations" / (vid_imgs.name + "-tool.txt")
        annotations = pd.read_csv(annotation_path, header=0, index_col=0, sep="\t")
        assert annotations.columns.to_list() == [
            "Grasper",
            "Bipolar",
            "Hook",
            "Scissors",
            "Clipper",
            "Irrigator",
            "SpecimenBag",
        ], f"Incorrect columns at video {vid_imgs.name}"
        frames_not_annotated = [f for f in frames if f not in annotations.index]
        annotations_without_frames = [f for f in annotations.index if f not in frames]
        # remove frames without annotation raise error for annotations without frames
        if len(annotations_without_frames) > 0:
            msg = f"Within video {vid_imgs.name} the frames {annotations_without_frames} are missing!"
            logger.error(msg)
            raise RuntimeError(msg)
        logger.debug(f"Within {vid_imgs.name} removing {len(frames_not_annotated)} frames without annotations...")
        for f in frames_not_annotated:
            frame_path = vid_imgs / f"{f}.jpg"
            frame_path.unlink()
        logger.debug(f"... done {ix} / {len(video_paths)}")
    logger.debug("Cleaned all extracted frames")
    # remove video files to save disk space
    for vid in video_paths:
        vid.unlink()
    logger.info("Removed videos to save disk space.")
    return dset_path


def create_cholec80_subtask(task: str, dset_path: Path, alias: str) -> None:
    assert task in ALL_TASKS
    task_creator = TaskCreator(
        dset_path=dset_path,
        name=alias,
        task_type=TaskType.CLASSIFICATION,
        desc="The Cholec80 dataset contains 80 videos of cholecystectomy surgeries performed "
        "by 13 surgeons. The videos are captured at 25 fps. The dataset is labeled with "
        "the phase (at 25 fps) and tool presence annotations (at 1 fps). The phases have "
        "been defined by a senior surgeon in our partner hospital. Since the tools are "
        "sometimes hardly visible in the images and thus difficult to be recognized "
        "visually, we define a tool as present in an image if at least half of the tool "
        "tip is visible.",
        ref=REFERENCE,
        url="http://camma.u-strasbg.fr/datasets/cholec80",
        instr="download via http://www.cs.rug.nl/~imaging/databases/melanoma_naevi/",
        lic=License.CC_BY_NC_SA_4_0,
        release="2017",
        keywords=[Keyword.MEDICAL, Keyword.ENDOSCOPY, Keyword.ENDOSCOPIC_INSTRUMENTS, Keyword.LAPAROSCOPY],
    )
    all_vids = list((dset_path / DataKind.TRAINING_DATA / "images").iterdir())
    vids_split = {"train": all_vids[:40], "test": all_vids[40:]}
    idx_to_class = {0: "not_present", 1: "present"}
    data_iterators = {k: [] for k in vids_split}
    for phase in vids_split:
        # load annotations
        for vid in vids_split[phase]:
            # read out task annotations for given video as pandas series
            annots = pd.read_csv(
                dset_path / DataKind.TRAINING_DATA / "tool_annotations" / (vid.name + "-tool.txt"),
                index_col=0,
                header=0,
                sep="\t",
            )[task]
            for frame in vid.iterdir():
                data_iterators[phase].append(
                    {
                        Modality.SAMPLE_ID: f"vid{vid.name}_frame{frame.stem}",
                        Modality.IMAGE: frame,
                        Modality.CLASS: int(annots.loc[int(frame.stem)]),
                    }
                )
    task_creator.find_data(
        train_iterator=data_iterators["train"], test_iterator=data_iterators["test"], idx_to_class=idx_to_class
    )
    task_creator.split_folds(n_folds=5, ensure_balancing=True)
    # stats are identical across tasks, save calculation time here
    task_creator.set_stats(
        means=RGBInfo(*[0.34170591831207275, 0.22947388887405396, 0.22436298429965973]),
        stds=RGBInfo(*[0.2470075637102127, 0.2106027603149414, 0.20748572051525116]),
        sizes=Sizes(*[480, 1080, 854, 1920]),
    )
    task_creator.push_and_test()


for task in ALL_TASKS:
    alias = f"{dset_name}_{task.lower()}_presence"
    creator_func = create_creator_func(create_func=create_cholec80_subtask, task=task, alias=alias)
    register_taskcreator(task_name=alias, dset_name=dset_name)(creator_func)
