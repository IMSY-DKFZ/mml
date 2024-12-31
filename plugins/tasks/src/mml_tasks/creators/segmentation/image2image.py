# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

# creates the image2image dataset and tasks
# TODO -> image2image requires 7zip extraction which we have not yet find a MIT compatible way of auto-extraction,
#  hence this is not supported currently
from pathlib import Path

from mml.core.data_loading.task_attributes import Keyword, License, Modality, TaskType
from mml.core.data_preparation.data_archive import DataKind
from mml.core.data_preparation.dset_creator import DSetCreator
from mml.core.data_preparation.registry import register_dsetcreator, register_taskcreator
from mml.core.data_preparation.task_creator import TaskCreator

REFERENCE = """
@misc{pfeiffer2019generating,
      title={Generating large labeled data sets for laparoscopic image processing tasks using unpaired image-to-image translation}, 
      author={Micha Pfeiffer and Isabel Funke and Maria R. Robu and Sebastian Bodenstedt and Leon Strenger and Sandy Engelhardt and Tobias Roß and Matthew J. Clarkson and Kurinchi Gurusamy and Brian R. Davidson and Lena Maier-Hein and Carina Riediger and Thilo Welsch and Jürgen Weitz and Stefanie Speidel},
      year={2019},
      eprint={1907.02882},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""  # noqa W291


def get_idx_to_class():
    # from https://gitlab.com/nct_tso_public/laparoscopic-image-2-image-translation/-/blob/master/data.py
    # but these are incorrect, after loading the labels appear to be 26, 77, 102, 128, 153, 179
    # idx_to_class = {0: 'Background',
    #                 89: 'Liver',
    #                 170: 'Fat',
    #                 149: 'Diaphragm',
    #                 188: 'Ligament',
    #                 218: 'Tool Shaft',
    #                 203: 'Tool Tip',
    #                 124: 'Gallbladder'}
    idx_to_class = {
        26: "Background?",
        51: "Liver?",
        77: "Fat?",
        102: "Abdominal Wall?",
        128: "Tool Shaft?",
        153: "Tool Tip?",
        179: "Gallbladder?",
    }
    return idx_to_class


@register_dsetcreator(dset_name="image2image")
def create_image2image():
    raise RuntimeError(
        "image2image dataset requires 7zip extraction which we have not yet find a MIT compatible way "
        "of auto-extraction, hence this is not supported currently"
    )
    dset_creator = DSetCreator(dset_name="image2image")
    dset_creator.download(
        url="http://opencas.dkfz.de/image2image/data/inputs.7z", file_name="inputs.7z", data_kind=DataKind.TRAINING_DATA
    )
    dset_creator.download(
        url="http://opencas.dkfz.de/image2image/data/stylernd.7z",
        file_name="stylernd.7z",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_creator.download(
        url="http://opencas.dkfz.de/image2image/data/styleFromCholec80.7z",
        file_name="styleFromCholec80.7z",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_creator.download(
        url="http://opencas.dkfz.de/image2image/data/labels.7z",
        data_kind=DataKind.TRAINING_LABELS,
        file_name="labels.7z",
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="image2image-raw", dset_name="image2image")
def create_image2image_raw(dset_path: Path):
    raw = TaskCreator(
        dset_path=dset_path,
        name="image2image-raw",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="anatomical structures segmented from simulated laparoscopic images",
        ref=REFERENCE,
        url="http://opencas.dkfz.de/image2image/",
        instr="download <input images> and <labels:segmentation> via website",
        lic=License.UNKNOWN,
        release="2019",
        keywords=[
            Keyword.LAPAROSCOPY,
            Keyword.ANATOMICAL_STRUCTURES,
            Keyword.ARTIFICIAL,
            Keyword.MEDICAL,
            Keyword.ENDOSCOPIC_INSTRUMENTS,
        ],
    )
    data_iterator = []
    img_root = dset_path / DataKind.TRAINING_DATA / "simulated"
    mask_root = dset_path / DataKind.TRAINING_LABELS / "simulated"
    for patient_id in [p.name for p in img_root.iterdir() if p.is_dir()]:
        for img_id in [p.name for p in (img_root / patient_id / "inputs").iterdir() if p.is_file()]:
            data_iterator.append(
                {
                    Modality.SAMPLE_ID: f"{patient_id}_{img_id}",
                    Modality.IMAGE: img_root / patient_id / "inputs" / img_id,
                    Modality.MASK: mask_root / patient_id / "labels" / (img_id.replace("img", "lbl")),
                }
            )
    raw.find_data(train_iterator=data_iterator, idx_to_class=get_idx_to_class())
    raw.split_folds(n_folds=5, ensure_balancing=True)
    raw.infer_stats()
    raw.push_and_test()


@register_taskcreator(task_name="image2image-rand", dset_name="image2image")
def create_image2image_rand(dset_path: Path):
    raw = TaskCreator(
        dset_path=dset_path,
        name="image2image-rand",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="anatomical structures segmented from simulated laparoscopic images, used random styles",
        ref=REFERENCE,
        url="http://opencas.dkfz.de/image2image/",
        instr="download <translations, random style> and <labels:segmentation> via website",
        lic=License.UNKNOWN,
        release="2019",
        keywords=[
            Keyword.LAPAROSCOPY,
            Keyword.ANATOMICAL_STRUCTURES,
            Keyword.ARTIFICIAL,
            Keyword.MEDICAL,
            Keyword.ENDOSCOPIC_INSTRUMENTS,
        ],
    )
    data_iterator = []
    img_root = dset_path / DataKind.TRAINING_DATA / "stylernd"
    mask_root = dset_path / DataKind.TRAINING_LABELS / "simulated"
    for patient_id in [p.name for p in img_root.iterdir() if p.is_dir()]:
        for img_id in [p.name for p in (img_root / patient_id / "style_00").iterdir() if p.is_file()]:
            data_iterator.append(
                {
                    Modality.SAMPLE_ID: f"{patient_id}_{img_id}",
                    Modality.IMAGE: img_root / patient_id / "style_00" / img_id,
                    Modality.MASK: mask_root / patient_id / "labels" / (img_id.replace("img", "lbl")),
                }
            )
    raw.find_data(train_iterator=data_iterator, idx_to_class=get_idx_to_class())
    raw.split_folds(n_folds=5, ensure_balancing=True)
    raw.infer_stats()
    raw.push_and_test()


@register_taskcreator(task_name="image2image-cholec80", dset_name="image2image")
def create_image2image_cholec80(dset_path: Path):
    raw = TaskCreator(
        dset_path=dset_path,
        name="image2image-cholec80",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="anatomical structures segmented from simulated laparoscopic images, used cholec80 styles",
        ref=REFERENCE,
        url="http://opencas.dkfz.de/image2image/",
        instr="download <translations, cholec80 style> and <labels:segmentation> via website",
        lic=License.UNKNOWN,
        release="2019",
        keywords=[
            Keyword.LAPAROSCOPY,
            Keyword.ANATOMICAL_STRUCTURES,
            Keyword.ARTIFICIAL,
            Keyword.MEDICAL,
            Keyword.ENDOSCOPIC_INSTRUMENTS,
        ],
    )
    data_iterator = []
    img_root = dset_path / DataKind.TRAINING_DATA / "styleFromCholec80"
    mask_root = dset_path / DataKind.TRAINING_LABELS / "simulated"
    for patient_id in [p.name for p in img_root.iterdir() if p.is_dir()]:
        for img_id in [p.name for p in (img_root / patient_id / "style_00").iterdir() if p.is_file()]:
            data_iterator.append(
                {
                    Modality.SAMPLE_ID: f"{patient_id}_{img_id}",
                    Modality.IMAGE: img_root / patient_id / "style_00" / img_id,
                    Modality.MASK: mask_root / patient_id / "labels" / (img_id.replace("img", "lbl")),
                }
            )
    raw.find_data(train_iterator=data_iterator, idx_to_class=get_idx_to_class())
    raw.split_folds(n_folds=5, ensure_balancing=True)
    raw.infer_stats()
    raw.push_and_test()
