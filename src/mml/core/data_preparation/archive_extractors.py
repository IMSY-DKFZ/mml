# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import platform
import shutil
import subprocess
import tarfile
import warnings
from getpass import getpass
from pathlib import Path
from typing import Callable, List, Optional
from zipfile import ZipFile

from rarfile import RarCannotExec, RarFile

from mml.core.data_preparation.data_archive import DataArchive
from mml.core.data_preparation.utils import WIPBar
from mml.core.scripts.decorators import timeout

logger = logging.getLogger(__name__)


def default_file_copy(archive: DataArchive, folder: Path) -> None:
    """
    Default extraction is to fall back to a simple copy for files (e.g. sometimes plain .csv files are shared).

    :param DataArchive archive: DataArchive to extract.
    :param Path folder: target folder.
    :return:
    """
    shutil.copy2(str(archive.path), str(folder))


def default_folder_copy(archive: DataArchive, folder: Path) -> None:
    """
    Default extraction is to fall back to a simple copy for folders (e.g. for very small datasets).

    :param DataArchive archive: DataArchive to extract.
    :param Path folder: target folder.
    :return:
    """
    if not archive.keep_top_level:
        raise RuntimeError("Please set DataArchive.keep_top_level to true for plain folder extraction.")
    try:
        shutil.copytree(src=str(archive.path), dst=str(folder))
    except FileExistsError:
        raise RuntimeError(
            f"Apparently folder {archive.path.name} is used multiple times during unpacking/copying "
            f"at target path {folder}. Make sure to use each target only once!"
        )


def zip_extractor(archive: DataArchive, folder: Path) -> None:
    """
    Extraction for ZIP files.

    :param DataArchive archive: DataArchive to extract.
    :param Path folder: target folder.
    :return:
    """
    with ZipFile(archive.path) as myzip:
        # simple case first: no encryption
        if not is_encrypted(zipfile=myzip):
            myzip.extractall(path=folder)
        else:
            logger.info(f"trying to decrypt archive {archive.path.name} ...")
            if not archive.password:
                archive.password = ask_password(file=archive.path)
            # check OS
            os_type = platform.system()
            if os_type not in ["Linux", "Darwin"]:  # Darwin for macOS
                warnings.warn(
                    f"Unsupported OS type: {os_type} for fast zip extraction. If extraction is too slow, "
                    f"cancel process and manually extract {myzip.filename} into {folder}."
                )
                myzip.extractall(path=folder, pwd=archive.password.encode())
            else:
                # UNIX file system allows to call "unzip" as a subprocess for way faster extraction
                command = ["unzip", "-P", archive.password, archive.path, "-d", folder]
                try:
                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError:
                    logger.error("Wrong Password! Try again.")
                    raise RuntimeError("Wrong Password! Restart and try again.")
                except FileNotFoundError:
                    warnings.warn(
                        "'unzip' command not installed. You may want to install it (e.g. 'sudo apt install "
                        "unzip' for ubuntu/debian/mint) and then restart. In the meantime will fall back to "
                        "slow extraction."
                    )
                    myzip.extractall(path=folder, pwd=archive.password.encode())


def rar_extractor(archive: DataArchive, folder: Path) -> None:
    """
    Extraction for RAR files.

    :param DataArchive archive: DataArchive to extract.
    :param Path folder: target folder.
    :return:
    """
    try:
        with RarFile(archive.path) as myrar:
            myrar.extractall(path=folder)
    except RarCannotExec:
        raise RuntimeError(
            "It seems like there is a backend issue regarding the rarfile package. For debian based "
            "systems you might want to <<sudo apt install unrar>>. For windows users see: "
            "https://rarfile.readthedocs.io/faq.html#how-can-i-get-it-work-on-windows"
        )


def tar_extractor(archive: DataArchive, folder: Path) -> None:
    """
    Extraction for TAR files.

    :param DataArchive archive: DataArchive to extract.
    :param Path folder: target folder.
    :return:
    """
    with tarfile.open(archive.path, "r") as mytar:
        # check for CVE-2001-1267 - see https://github.com/advisories/GHSA-gw9q-c7gh-j9vm
        for member in mytar.getmembers():
            path = (folder / member.name).resolve()
            if folder not in path.parents:
                raise RuntimeError(f"{member=} has a suspicious path structure!")
        mytar.extractall(path=folder)


# maps file suffix to extractor function, plugins may add extractors
ARCHIVE_EXTRACTOR_FUNCTIONS = {
    ".zip": zip_extractor,
    ".tgz": tar_extractor,
    ".tar": tar_extractor,
    ".gz": tar_extractor,
    ".rar": rar_extractor,
}


def unpack_files(archives: List[DataArchive], target: Path) -> None:
    """
    Extracts and copies archives to the target folder. Supports zip, rar and tar archives. Also copies non-archive files
    as well as folders.

    :param List[DataArchive] archives: List of data archives.
    :param Path target: Target folder path.
    :return: None
    """
    if not target.exists():
        raise FileNotFoundError(f"Extraction root folder {target} has to exist!")
    logger.info(f"starting extracting {len(archives)} file(s) to {target} ...")
    # extract files (or copy if not stored as archives)
    with WIPBar() as bar:
        bar.desc = "Extracting archives"
        for ix, arch in enumerate(archives):
            if not arch.path.exists():
                raise FileNotFoundError(f"Did not find file {arch.path}!")
            extract_dir = target / arch.path.stem if arch.keep_top_level else target
            if arch.path.suffix in ARCHIVE_EXTRACTOR_FUNCTIONS:
                func: Callable[[DataArchive, Path], None] = ARCHIVE_EXTRACTOR_FUNCTIONS[arch.path.suffix]
                logger.debug(f"extracting archive {arch} with extractor {func.__name__}")
                func(arch, extract_dir)
            else:
                logger.info(
                    f"No specific extractor found for archive {arch}, will copy. Please ensure no extraction is "
                    f"necessary. Otherwise add an extractor to ARCHIVE_EXTRACTOR_FUNCTIONS in {__name__}."
                )
                if arch.path.is_file():
                    default_file_copy(archive=arch, folder=extract_dir)
                else:
                    default_folder_copy(archive=arch, folder=extract_dir)
            if len(archives) > 1:
                # report on extraction progress in case of multiple archives
                logger.info(f"Done {arch.path.name} ({ix + 1}/{len(archives)}).")
    logger.info(f"successfully extracted all files to {target}")


@timeout(seconds=300)
def ask_password(file: Optional[Path] = None) -> str:
    """Lets user input a password to decrypt a data file

    :param file: The file to unpack
    :type file: Optional[Path]
    :return: password
    :rtype: str
    """
    if file:
        msg = f"Please enter the password to decrypt the file '{file.name}: "
    else:
        msg = "Please enter the password to decrypt file(s): "
    try:
        pwd = getpass(prompt=msg)
    except TimeoutError:
        logger.error(
            "No input provided for necessary password, during dataset creation. Please restart and provide password!"
        )
        raise
    return pwd


def is_encrypted(zipfile: ZipFile) -> bool:
    """Checks whether a ZipFile is password protected

    :param zipfile: file to check
    :type zipfile: ZipFile
    :return: True if password protected, False otherwise
    :rtype: bool
    """
    pwd_required = False
    for zinfo in zipfile.infolist():
        # Read encryption status: https://hg.python.org/cpython/file/2.7/Lib/zipfile.py#l986
        is_encrypted = zinfo.flag_bits & 0x1
        if is_encrypted:
            pwd_required = True
            logger.info("Zip File is password protected")
            break
    return pwd_required
