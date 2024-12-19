# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import os
from enum import IntEnum
from pathlib import Path
from threading import Event, Thread
from typing import Callable, Dict, List, Optional, Tuple, Union

import requests
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

import mml.core
from mml.core.data_loading.task_attributes import DataSplit, Modality, RGBInfo, Sizes
from mml.core.data_loading.task_dataset import TaskDataset
from mml.core.data_loading.task_description import SampleDescription
from mml.core.scripts.exceptions import InvalidTransitionError
from mml.core.scripts.utils import Singleton

logger = logging.getLogger(__name__)


def download_file(path_to_store: Path, download_url: str, file_name: str) -> None:
    """
    Downloads file and places it accordingly. Skips finished downloads.

    :param Path path_to_store:
    :param str download_url:
    :param str file_name:
    :return: None
    """
    if not path_to_store.exists() or not path_to_store.is_dir():
        raise ValueError(f"Invalid path {path_to_store}, must be existing dir.")
    if (path_to_store / file_name).exists() and "wip_" != file_name[:4]:
        logger.warning(f"File {file_name} already present in {path_to_store}. Skipping download!")
        return
    logger.info(f"Downloading {file_name} to {path_to_store}.")
    # use a temporary file while downloading, so above test won't accept interrupted downloads
    tmp_file_name = "wip_" + file_name
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        # progress bar inspired by https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701#37573701
        progress = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(str(path_to_store / tmp_file_name), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                progress.update(len(chunk))
                f.write(chunk)
        progress.close()
        if total_size_in_bytes != 0 and progress.n != total_size_in_bytes:
            logger.error(f"Something went wrong during download of {file_name}.")
    # rename finished download to actual file name
    (path_to_store / tmp_file_name).rename(path_to_store / file_name)
    logger.info(f"Successfully downloaded {file_name} to {path_to_store}.")


def calc_means_stds_sizes(
    task_path: Path,
    means: bool = True,
    stds: bool = True,
    sizes: bool = True,
    const_size: bool = False,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, Union[Sizes, RGBInfo]]:
    """
    Calculates means, stds and/or sizes of a task. Requires at most 2 runs through the dataset. Might take some time.

    :param task_path: path to task .json file.
    :param means: if means should be calculated
    :param stds: if stds should be calculated
    :param sizes: if sizes should be calculated
    :param const_size: if images have constant size (allows for faster loading, if 'sizes' is True and finds constant
        sizes, this is detected internally)
    :param device: device to be used for computations
    :return: dict with possible keys 'sizes', 'means', 'stds' and Sizes / RGBInfo values
    """
    logger.info("Calculating mean, std and size. This may take a couple of minutes.")
    ds = TaskDataset(root=task_path, split=DataSplit.FULL_TRAIN, transform=ToTensorV2())
    if len(ds) == 0:
        # if task is primarily unlabeled we gather those stats
        ds = TaskDataset(root=task_path, split=DataSplit.UNLABELLED, transform=ToTensorV2())
    info = {}
    if sizes:
        img_files = [ds.root.parent / sample[Modality.IMAGE] for sample in ds.samples]
        min_height, max_height, min_width, max_width = 100000, 0, 100000, 0
        for file_path in tqdm(img_files, desc="Gathering sizes"):
            # use Pillow since we only need lazy loading to gather size information
            img = Image.open(file_path)
            min_height = min(min_height, img.height)
            max_height = max(max_height, img.height)
            min_width = min(min_width, img.width)
            max_width = max(max_width, img.width)
            if const_size:
                break
        info["sizes"] = Sizes(min_height=min_height, max_height=max_height, min_width=min_width, max_width=max_width)
        if min_height == max_height and min_width == max_width:
            const_size = True
        logger.debug(f"Sizes are {info['sizes']}.")
    if means or stds:
        loader = DataLoader(dataset=ds, batch_size=100 if const_size else 1)
        counter = 0
        channel_sum = torch.zeros(3, dtype=torch.double, device=device)
        channel_sq = torch.zeros(3, dtype=torch.double, device=device)
        for batch in tqdm(loader, desc="Gathering mean and std"):
            images = (batch[Modality.IMAGE.value] / 255).to(torch.double).to(device)
            counter += images.size()[0]
            channel_sum += images.mean(dim=[2, 3]).sum(dim=0)
            channel_sq += images.square().mean(dim=[2, 3]).sum(dim=0)
        _means = channel_sum / counter
        _stds = torch.sqrt(channel_sq / (counter - 1) - (counter * _means.square() / (counter - 1)))
        # _stds = torch.sqrt((channel_sq - (channel_sum.square() / counter)) / (counter - 1))
        if means:
            info["means"] = RGBInfo(*_means.to(torch.float).cpu().numpy().tolist())
        if stds:
            info["stds"] = RGBInfo(*_stds.to(torch.float).cpu().numpy().tolist())
        logger.debug(f"Means are {info['means']}. Stds are {info['stds']}.")
    return info


def get_iterator_and_mapping_from_image_dataset(
    root: Path, dup_id_flag: Optional[bool] = False, classes: Optional[List[str]] = None
) -> Tuple[List[SampleDescription], Dict[int, str]]:
    """
    Utility func for the reoccurring case, that classification datasets are ordered as in
    :class:`~torchvision.datasets.ImageFolder`. The iterator will store the file stem of the image as id.

    :param Path root: root path
    :param Optional[bool] dup_id_flag: (optional) flag for same filenames in different classes
    :param Optional[List[str]] classes: (optional) list defining classes, if not given any dir will be used as class
    :return: data iterator and idx_to_class as used for TaskCreator.find_data
    """
    if classes is None:
        classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    else:
        folders = [p.name for p in root.iterdir() if p.is_dir()]
        assert all([cl in folders for cl in classes]), "some class folder is not existent"
    idx_to_class = {classes.index(cl): cl for cl in classes}
    data_iterator = []
    for class_folder in root.iterdir():
        assert class_folder.is_dir()
        if class_folder.name not in classes:
            continue
        for img_path in class_folder.iterdir():
            if dup_id_flag:
                data_iterator.append(
                    {
                        Modality.SAMPLE_ID: class_folder.name + img_path.stem,
                        Modality.IMAGE: img_path,
                        Modality.CLASS: classes.index(class_folder.name),
                    }
                )
            else:
                data_iterator.append(
                    {
                        Modality.SAMPLE_ID: img_path.stem,
                        Modality.IMAGE: img_path,
                        Modality.CLASS: classes.index(class_folder.name),
                    }
                )
    return data_iterator, idx_to_class


def get_iterator_from_segmentation_dataset(
    images_root: Path, masks_root: Path, path_matcher: Callable[[Path], Path] = lambda x: x
) -> List[SampleDescription]:
    """
    Utility func for the reoccuring case, that segmentation datasets are ordered as follows: there are two separate
    folders containing the images and labels respectively. There is also some similar structure / pattern in the naming
    of these. The iterator will store the file stem of the image as id.

    :param images_root: root path of image data
    :param masks_root: root path of mask data
    :param path_matcher: (optional) function to get the (relative) mask path from the (relative) image path, relative
        corresponds to the provided root paths, default value is the identity function
    :return: data iterator as used for TaskCreator.find_data
    """
    assert images_root.exists() and masks_root.exists(), f"{images_root=}, {masks_root=}"
    unmatched_counter = 0
    data_iterator = []
    for root, _, files in os.walk(str(images_root)):
        for file in files:
            image_path = Path(root) / file
            mask_path = masks_root / path_matcher(image_path.relative_to(images_root))
            if mask_path.exists():
                data_iterator.append(
                    {
                        Modality.SAMPLE_ID: str(image_path.relative_to(images_root)),
                        Modality.IMAGE: image_path,
                        Modality.MASK: mask_path,
                    }
                )
            else:
                logger.debug(f"unable to match {image_path=} to mask, was given {mask_path=}")
                unmatched_counter += 1
    logger.info(f"Was able to match {len(data_iterator)} items, encountered {unmatched_counter} unmatchable images.")
    return data_iterator


def get_iterator_from_unlabeled_dataset(root: Path) -> List[SampleDescription]:
    """
    Utility func for the reoccurring case, that unlaballed data is simply organised in a single folder. The iterator
    will store the file stem of the image as id.

    :param Path root: root path
    :return: data iterator as used for TaskCreator.find_data
    """
    data_iterator = []
    for img_path in root.iterdir():
        data_iterator.append(
            {
                Modality.SAMPLE_ID: img_path.stem,
                Modality.IMAGE: img_path,
            }
        )
    return data_iterator


class TaskCreatorActions(mml.core.scripts.utils.StrEnum):
    """
    Abstract action that can be done on a task creator.
    """

    FIND_DATA = "find_data"
    LOAD = "load"
    MODIFY = "modify"
    SET_STATS = "set_stats"
    SET_FOLDING = "set_folds"
    FINISH = "finish"
    NONE = "none"


class TaskCreatorState(IntEnum):
    """
    Abstract states a task creator can be in. Default traversal path is:
    INIT --find_data--> DATA_FOUND --set_folding--> FOLDS_SPLIT --infer/set_stats--> STATS_SET --finish--> FINISHED
    """

    INIT = 0  # default start state
    DATA_FOUND = 1  # ensure data samples are read in (i.e. self.data is set)
    FOLDS_SPLIT = 2  # ensure samples are stored and folds are split (e.g. self.current_meta.train_folds)
    STATS_SET = 3  # ensure folds are split and means, stds and sizes are set in current_meta
    FINISHED = 4  # ultimate state, creator has reached "end-of-life"

    def traverse(self, action: TaskCreatorActions) -> "TaskCreatorState":
        """
        Implements the legal traversals of states and actions within a task creator.

        :param TaskCreatorActions action: The action that is tried to be applied on the current state.
        :return: The follow-up state of the task creator.
        """
        if self == TaskCreatorState.FINISHED:
            raise InvalidTransitionError(
                "TaskCreator already finished. It is considered better practice to create a " "new one."
            )
        elif action == TaskCreatorActions.NONE:
            return self
        elif action == TaskCreatorActions.FIND_DATA:
            return TaskCreatorState.DATA_FOUND
        elif self == TaskCreatorState.INIT and action == TaskCreatorActions.LOAD:
            return TaskCreatorState.STATS_SET
        elif self >= TaskCreatorState.DATA_FOUND and action == TaskCreatorActions.SET_FOLDING:
            return TaskCreatorState.FOLDS_SPLIT
        elif self >= TaskCreatorState.FOLDS_SPLIT and action == TaskCreatorActions.SET_STATS:
            return TaskCreatorState.STATS_SET
        elif self >= TaskCreatorState.STATS_SET and action == TaskCreatorActions.FINISH:
            return TaskCreatorState.FINISHED
        else:
            raise InvalidTransitionError(f"Invalid traversal from {self} with action {action}.")


class WIPBar(Singleton):
    """
    A singleton class that shows a loading loop managed by a thread (e.g. while data is copied).
    Description can be updated during loading and in the end will print either a success or failure message,
    depending on whether an exception was raised. Exception handling can be done either inside or outside
    the context. Does not interfere with itself if used in a nested fashion, but the user
    is responsible for updating the description after each inner loop closes.

    Usage:

    .. code-block:: python

        with WIPBar() as bar:
            bar.desc = 'Copying'
            shutil.copytree(...)
            bar.desc = 'Extracting'
            zipfile ...
        # continue without WIPBar

    """

    def __init__(self):
        self.desc = None
        self.success_message = None
        self.failed_message = None
        self.symbols = ["⣾", "⣷", "⣯", "⣟", "⡿", "⢿", "⣻", "⣽"]
        # alternative: ['▘', '▌', '▙', '█', '▟', '▐', '▝', ' ']
        self.failed = False
        self.__bar = None
        self.__counter = 0  # counts open nestings
        self.__StopEvent = Event()  # used to signal to end the thread
        self.__thread: Optional[Thread] = None  # this thread will update the bar text
        self.__context_mgr = None  # this mgr keeps track of the logging redirect
        self.reset_messages()

    def __enter__(self):
        # whenever we enter we expect to reset previous failures and kill events
        self.failed = False
        self.__StopEvent.clear()
        self.__counter += 1
        # create bar and handle logging
        if self.__bar is None:
            self.__context_mgr = tqdm_logging_redirect(bar_format="{elapsed} {desc}", desc="", leave=True)
            self.__bar: tqdm = self.__context_mgr.__enter__()
        # start thread if not running yet
        if self.__thread is None:
            self.__thread = Thread(target=self.__update, daemon=True)
            self.__thread.start()
        return self

    def reset_messages(self):
        self.desc = ""
        self.success_message = "✅ Success"
        self.failed_message = "❌ Failed"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__counter -= 1
        # check if an exception was raised
        if (exc_tb, exc_val, exc_tb) != (None, None, None):
            self.failed = True
        if self.__counter == 0:
            # signal the thread to terminate
            self.__StopEvent.set()
            # and wait to terminate
            if self.__thread:
                self.__thread.join()
            # reset thread and bar
            self.__thread = None
            self.__bar.close()
            self.__bar = None
            self.reset_messages()
            self.__context_mgr.__exit__(exc_type, exc_val, exc_tb)

    def __update(self):
        """
        Implements the loop inside the thread to show a repeating pattern of symbols.
        """
        i = 0
        while True:
            i = (i + 1) % len(self.symbols)
            self.__bar.set_description_str(self.symbols[i] + " " + self.desc)
            # check for interruption and exit if signaled
            if self.__StopEvent.wait(0.2):
                break
        if self.failed:
            self.__bar.set_description_str(self.desc + " " + self.failed_message)
        else:
            self.__bar.set_description_str(self.desc + " " + self.success_message)
