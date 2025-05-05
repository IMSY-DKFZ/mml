# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import os
import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from p_tqdm import p_umap
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import mml.core.data_preparation.utils as prep_utils
import mml.core.scripts.utils as core_utils
from mml.core.data_loading.file_manager import DSET_PREFIX, MMLFileManager
from mml.core.data_loading.task_attributes import Modality, TaskType
from mml.core.data_preparation.archive_extractors import unpack_files
from mml.core.data_preparation.data_archive import DataArchive, DataKind
from mml.core.scripts.exceptions import MMLMisconfigurationException

logger = logging.getLogger(__name__)

try:
    from kaggle import api as kaggle_api

    _kaggle_available = True
except OSError:
    _kaggle_available = False
    kaggle_api = None


class DSetCreator:
    """
    The dataset creator handles all relevant steps to prepare the dataset on your device. This includes:
    - downloading the data and checking hashes
    - extracting the data from archives
    - alternatively extract data from existing pytorch datasets
    - storing the data at the correct spots (unlabeled, train and test data)
    - optionally transforming masks of segmentation data

    Main usage:
        (a) Based on use case:
                (i) call .download() / .kaggle_download() / .verify_pre_download() function to download/register files
                (ii) call .unpack_and_store() to unpack files and move them to the correct spot
            OR:
                (i) create / import a pytorch dataset
                (ii) call .extract_from_pytorch_datasets to extract that data
        (b) (optionally) turn the masks of segmentation tasks to the correct format with .transform_masks()
    """

    def __init__(self, dset_name: str, download_path: Optional[Path] = None, dset_path: Optional[Path] = None):
        """
        Creator class for datasets.

        :param dset_name: name of the dataset (should be short, since is used in directory names)
        :param download_path: (optional) a path to already downloaded files
        :param dset_path: (optional) a path to an already created dset folder
        """
        # the instance calls correctly detects an existing file manager, but falls back on creating one in case
        # there has not been created one yet
        core_utils.load_env()
        self.fm = MMLFileManager.instance(
            data_path=Path(os.getenv("MML_DATA_PATH")), proj_path=Path(os.getcwd()), log_path=Path(tempfile.mkdtemp())
        )
        self.dset_name = dset_name
        if len(self.dset_name) > 20:
            raise ValueError("please use a shorter dset name to prevent long directory names")
        if " " in self.dset_name:
            raise ValueError("please avoid blanks in dset name")
        logger.debug(f"Creating dataset {self.dset_name}.")
        if download_path is not None:
            if not download_path.exists():
                raise FileNotFoundError(f"Download path {download_path} not existing.")
            if download_path.is_file():
                raise NotADirectoryError("Download path must be a directory.")
            self.download_path = download_path
        else:
            self.download_path = self.fm.get_download_path(dset_name=self.dset_name)
        self.archives: List[DataArchive] = []
        if dset_path is not None:
            if not dset_path.exists() or DSET_PREFIX not in dset_path.stem:
                raise ValueError(f"Incorrect dset_path given: {dset_path}")
        self.dset_path = dset_path

    def download(
        self, url: str, file_name: str, data_kind: DataKind = DataKind.MIXED, md5: Optional[str] = None
    ) -> DataArchive:
        """
        Downloads files from the web to your local drive.

        :param str url: URL to download from
        :param str file_name: name of the file
        :param DataKind data_kind: (optional) type of the data
        :param Optional[str] md5: (optional) md5 sum of the downloaded obj
        :return: a reference to the created archive, may be used to modify keep_top_dir before extraction
        """
        if data_kind not in DataKind:
            raise ValueError(f"data_kind {data_kind} invalid")
        prep_utils.download_file(path_to_store=self.download_path, download_url=url, file_name=file_name)
        archive = DataArchive(path=self.download_path / file_name, md5sum=md5, kind=data_kind)
        archive.check_hash()
        if archive.path in [arch.path for arch in self.archives]:
            raise RuntimeError(f"File {file_name} already assigned!")
        self.archives.append(archive)
        return archive

    def kaggle_download(
        self, competition: Optional[str] = None, dataset: Optional[str] = None, data_kind: DataKind = DataKind.MIXED
    ) -> List[DataArchive]:
        """
        Downloads all the data of a kaggle competition or dataset. Either specify competition XOR dataset parameter!
        Currently only a single kaggle download is supported per dataset!

        :param competition: (optional) kaggle competition identifier, mutually exclusive with dataset parameter
        :param dataset: (optional) kaggle dataset identifier, mutually exclusive with competition parameter
        :param DataKind data_kind: (optional) type of the data
        :return: a List of references to the created archives, may be used to modify keep_top_dir before extraction
        """
        if not _kaggle_available:
            raise MMLMisconfigurationException(
                "Kaggle authentication failed. Make sure to provide "
                "credential either through mml.env file or otherwise "
                "(see https://github.com/Kaggle/kaggle-api)."
            )
        if data_kind not in DataKind:
            raise ValueError(f"data_kind {data_kind} invalid")
        target = self.download_path / "kaggle"
        if target.exists():
            raise RuntimeError(
                "Current kaggle download does not support rerun or multiple downloads from kaggle to one dataset."
            )
        target.mkdir()
        if sum((bool(competition), bool(dataset))) != 1:
            raise ValueError("give either competition or dataset")
        if competition:
            logger.info(f"Downloading kaggle competition {competition} to {target}.")
            kaggle_api.competition_download_files(competition=competition, path=target, force=False, quiet=False)
            logger.info(f"Successfully downloaded {competition} kaggle competition.")
        if dataset:
            logger.info(f"Downloading kaggle dataset {dataset} to {target}.")
            kaggle_api.dataset_download_files(dataset=dataset, path=target, force=False, quiet=False, unzip=False)
            logger.info(f"Successfully downloaded {dataset} kaggle dataset.")
        # no information on file names provided, assume all new files are relevant
        archives = []
        for file in target.iterdir():
            if file not in [arch.path for arch in self.archives]:
                archives.append(DataArchive(path=file, kind=data_kind))
        if len(archives) == 0:
            raise FileNotFoundError("Kaggle download failed...")
        self.archives.extend(archives)
        return archives

    def verify_pre_download(
        self, file_name: str, instructions: str, data_kind: DataKind = DataKind.MIXED, md5: Optional[str] = None
    ) -> DataArchive:
        """
        Verifies a file that has been previously downloaded is present and adds it to internal archive list. This
        is useful if e.g. downloading requires credentials / registering or the data is non-public.

        :param str file_name: name of the downloaded file or folder
        :param str instructions: how to get the data
        :param DataKind data_kind: (optional) kind of the data
        :param str md5: (optional) md5 sum of the downloaded obj, only effective for non-folder files
        :return: a reference to the created archive, may be used to modify keep_top_dir before extraction
        """
        if data_kind not in DataKind:
            raise ValueError(f"data_kind {data_kind} invalid")
        path = self.download_path / file_name
        if not path.exists():
            raise ValueError(
                f"file {file_name} not found at {self.download_path}, please follow these instructions: {instructions}."
            )
        if md5 and path.is_dir():
            raise IsADirectoryError(
                f"Hash checking only supported for archived folders or files. {file_name} is not a file"
            )
        archive = DataArchive(path=path, md5sum=md5, kind=data_kind)
        archive.check_hash()
        if archive.path in [arch.path for arch in self.archives]:
            raise RuntimeError(f"File {file_name} already assigned!")
        self.archives.append(archive)
        return archive

    def unpack_and_store(self, clear_download_folder: bool = False) -> Path:
        """
        Unpacks all files and stores them at the correct spot.

        :param clear_download_folder: if True deletes the download folder
        :return: the dataset root path (to be used by TaskCreator)
        """
        if self.dset_path is not None:
            raise RuntimeError("dset_path should not be given beforehand if unpacking new data!")
        self.dset_path = self.fm.get_dataset_path(dset_name=self.dset_name, preprocessing=None)
        for sub in DataKind.list():
            sub_path = self.dset_path / sub
            archives = [arch for arch in self.archives if arch.kind == sub]
            if len(archives) == 0:
                continue
            sub_path.mkdir(exist_ok=True)
            unpack_files(archives=archives, target=sub_path)
        if clear_download_folder:
            shutil.rmtree(self.download_path)
            logger.debug("Removed download folder.")
        logger.debug("Done with dataset creation!")
        return self.dset_path

    def extract_from_pytorch_datasets(
        self,
        datasets: Dict[str, Dataset],
        task_type: TaskType,
        allow_transforms: bool = False,
        class_names: Optional[List[str]] = None,
    ) -> Path:
        """
        Can be used to store an existing dataset (e.g. imported from some other repository or from torchvision). Expects
        (image, class) tuples for classification and (image, mask) tuples for semantic segmentation to return over
        __getitem__. Unlabeled dataset is expected to return no tuple, but only image.

        :param datasets: dict of datasets to be stored, keys must be from [training, testing, unlabeled]
        :param task_type: task type of the dataset
        :param allow_transforms: if the transform attribute is present and unequal to None whether to raise an error
        :param class_names: optional list of class names to store data with class name directories
        :return: the dataset root path (to be used by TaskCreator)
        """
        assert self.dset_path is None, "dset_path should not be given beforehand if storing new data"
        self.dset_path = self.fm.get_dataset_path(dset_name=self.dset_name, preprocessing=None)
        # check datasets
        for key, dataset in datasets.items():
            assert len(dataset) > 0, f"Was provided with empty dataset {key}."
            assert key in ["training", "testing", "unlabeled"], f"invalid key {key}"
            if not allow_transforms:
                if hasattr(dataset, "transform"):
                    assert dataset.transform is None, (
                        "dataset has a transform applied, set allow_transforms True if desired"
                    )
                if hasattr(dataset, "target_transform"):
                    assert dataset.target_transform is None, (
                        "dataset has a target_transform applied, set allow_transforms True if desired"
                    )
        # iterate over samples, load these and store correctly at readable format and intuitive place
        id_iterator = 0  # used as IDs
        for key, dataset in datasets.items():
            unlabeled = key == "unlabeled"
            sub = DataKind.UNLABELED_DATA if unlabeled else DataKind(key + "_data")
            sub_path = self.dset_path / sub.value
            sub_path.mkdir(exist_ok=False)
            if task_type == TaskType.SEMANTIC_SEGMENTATION and not unlabeled:
                sub_masks = DataKind(key + "_labels")
                sub_masks_path = self.dset_path / sub_masks.value
                sub_masks_path.mkdir(exist_ok=False)
            for sample in tqdm(dataset, desc=f"Extracting {key} data"):
                id_iterator += 1
                # unpack sample
                if unlabeled:
                    image = sample
                else:
                    # try to unpack
                    if isinstance(sample, tuple):
                        image, target = sample
                    elif isinstance(sample, dict):
                        if Modality.IMAGE.value not in sample or any(k not in Modality for k in sample):
                            raise ValueError(
                                f"Dataset loads as dictionary, but keys are not compliant with MML "
                                f"Modality options. Available options are: {Modality.list()}"
                            )
                        image = sample[Modality.IMAGE.value]
                        if task_type == TaskType.CLASSIFICATION:
                            target = sample[Modality.CLASS.value]
                        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
                            target = sample[Modality.MASK.value]
                        else:
                            raise NotImplementedError(f"Task type {task_type} is not suitable for dict extraction yet.")
                # first handle image object
                if isinstance(image, Image.Image):
                    # PIL image
                    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                elif isinstance(image, np.ndarray):
                    # numpy array as returned by cv2
                    pass
                else:
                    raise TypeError(f"image returned is of type {type(image)}")
                if len(image.shape) != 3:
                    # TODO probably convert greyscale images here?
                    raise ValueError(f"image is not well formatted, has shape {image.shape}")
                img_folder = sub_path
                if task_type == TaskType.CLASSIFICATION and not unlabeled:
                    img_folder /= str(target).zfill(3) if class_names is None else class_names[target]
                    img_folder.mkdir(exist_ok=True)
                image_path = img_folder / (str(id_iterator).zfill(6) + ".png")
                cv2.imwrite(str(image_path), image)
                # handle segmentation mask
                if task_type == TaskType.SEMANTIC_SEGMENTATION and not unlabeled:
                    # target object should be a mask
                    if isinstance(target, Image.Image):
                        # PIL image
                        target = cv2.cvtColor(np.asarray(target.convert(mode="RGB")), cv2.COLOR_RGB2BGR)
                    elif isinstance(target, np.ndarray):
                        # numpy array as returned by cv2
                        pass
                    else:
                        raise TypeError(f"target returned is of type {type(image)}")
                    if id_iterator == 1:
                        logger.info(f"targets of this dataset have shape {target.shape}")
                    cv2.imwrite(str(sub_masks_path / (str(id_iterator).zfill(6) + ".png")), target)
        return self.dset_path

    def transform_masks(
        self,
        masks: List[Path],
        transform: Dict[tuple, int],
        load: str = "rgb",
        train: bool = True,
        ignore: Optional[List[tuple]] = None,
    ) -> Path:
        """
        Takes the job of transforming masks to the frameworks format (cv2 readable greyscale images). Will write the
        transformed masks to dataset root -> training_labels / testing_labels -> transformed_masks -> same relative path
        as before (to mask within one of the data subfolders).

        :param masks: list of paths to the mask files
        :param transform: dict defining the transformation (mapping of mask values to classes)
        :param load: mode defining the loading of a file
        :param train: bool indicating if treated data is train or test
        :param ignore: list of mask values that should be mapped to the ignored value of 255 (similar to transform)
        :return: base folder for the transformed masks
        """
        # TODO in the future this might e.g. also include multi instance -> semantic segmentation
        # storing base path
        data_kind = DataKind.TRAINING_LABELS if train else DataKind.TESTING_LABELS
        out_base = self.dset_path / data_kind.value / "transformed_masks"
        out_base.mkdir(exist_ok=True, parents=True)
        # transform mapping
        if ignore is None:
            ignore = []

        example = list(transform.keys())[0]

        assert all([len(mask_value) == len(example) for mask_value in list(transform.keys()) + ignore]), (
            "tuples used as transform (or ignore) keys require identical shape"
        )
        assert all([0 <= val < 255 for val in transform.values()]), (
            f"was provided with incorrect classes ({transform.values()}) to fit into greyscale."
        )
        assert all([key not in ignore for key in transform.keys()]), "provided mask values are not unique"

        mapping = transform.copy()
        mapping.update({idxs: 255 for idxs in ignore})

        def mapper(*args):
            return mapping.__getitem__(args)

        vmapper = np.vectorize(mapper)
        logger.info(f"transforming {len(masks)} masks...")
        success = p_umap(
            partial(mask_transform, dset_path=self.dset_path, out_base=out_base, load=load, vmapper=vmapper), masks
        )
        logger.info(f"{sum(success)} of {len(success)} masks transformed successfully.")
        return out_base


def mask_transform(mask_path: Path, dset_path: Path, out_base: Path, load: str, vmapper: Callable) -> bool:
    """
    Parallelizable part of mask transform. Params keep names from within DsetCreator.transform_mask.

    :return: boolean indicating success of saving the transformed mask
    """
    assert dset_path in mask_path.parents, (
        f"path {mask_path} not within dataset, make sure to unpack before transforming"
    )
    # loading
    mask = None  # IDE related
    if load == "rgb":
        mask = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)
    elif load == "grayscale":
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    else:
        ValueError(f"Loading mode {load} not supported.")
    # transforming
    try:
        mask = vmapper(*np.split(mask, indices_or_sections=mask.shape[2], axis=2))
    except KeyError as error:
        logger.error(f"A key ({error}) is not present in the transform mapping! Mask can be found at {mask_path}.")
        raise error
    # saving
    path = out_base
    for part in mask_path.relative_to(dset_path).parts[1:]:
        path /= part
    path.parent.mkdir(exist_ok=True, parents=True)
    path = path.with_suffix(".png")
    success = cv2.imwrite(filename=str(path), img=mask)
    return success
