# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator
from mml.core.data_preparation.utils import get_iterator_from_segmentation_dataset

REFERENCE = """
@misc{borgli2020, 
title     = {Hyper-Kvasir: A Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy},
url       = {osf.io/mkzcq}, 
DOI       = {10.31219/osf.io/mkzcq}, 
publisher = {OSF Preprints}, 
author    = {Borgli, Hanna and Thambawita, Vajira and Smedsrud, Pia H and Hicks, Steven and Jha, Debesh and Eskeland, 
            Sigrun L and Randel, Kristin R and Pogorelov, Konstantin and Lux, Mathias and Nguyen, Duc T D and Johansen,
            Dag and Griwodz, Carsten and Stensland, H{\aa}kon K and Garcia-Ceja, Enrique and Schmidt, Peter T and Hammer,
            Hugo L and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas}, 
year      = {2019},
month     = {Dec}}
"""  # noqa W291


@register_dsetcreator(dset_name="hyperkvasir_seg")
def create_hyperkvasir_labeled():
    dset_creator = DSetCreator(dset_name="hyperkvasir_seg")
    dset_creator.download(
        url="https://datasets.simula.no/hyper-kvasir/hyper-kvasir-segmented-images.zip",
        file_name="hyper-kvasir-segmented-images.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    masks = list((dset_path / DataKind.TRAINING_DATA / "segmented-images" / "masks").iterdir())
    blacks = [246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
    whites = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    transform = {(pix_val,): 0 for pix_val in blacks}
    transform.update({(pix_val,): 1 for pix_val in whites})
    dset_creator.transform_masks(masks=masks, load="grayscale", train=True, transform=transform)
    return dset_path


@register_taskcreator(task_name="hyperkvasir_polyp_segmentation", dset_name="hyperkvasir_seg")
def create_task(dset_path: Path) -> None:
    creator = TaskCreator(
        dset_path=dset_path,
        name="hyperkvasir_polyp_segmentation",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="Hyper-Kvasir Dataset is a large image and video dataset from the gastrointestinal tract",
        ref=REFERENCE,
        url="https://datasets.simula.no/hyper-kvasir/",
        instr="download zips of segmented images via website",
        lic=License.CC_BY_4_0,
        release="2020",
        keywords=[Keyword.MEDICAL, Keyword.GASTROSCOPY_COLONOSCOPY, Keyword.TISSUE_PATHOLOGY],
    )
    train_iterator = get_iterator_from_segmentation_dataset(
        images_root=dset_path / DataKind.TRAINING_DATA / "segmented-images" / "images",
        masks_root=dset_path / DataKind.TRAINING_LABELS / "transformed_masks" / "segmented-images" / "masks",
        path_matcher=lambda x: x.with_suffix(".png"),
    )
    creator.find_data(train_iterator=train_iterator, idx_to_class={0: "background", 1: "polyp"})
    creator.split_folds(n_folds=5)
    creator.infer_stats()
    creator.push_and_test()
