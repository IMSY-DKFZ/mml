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
@InProceedings{10.1007/978-3-319-10470-6_55,
author="Maier-Hein, Lena
and Mersmann, Sven
and Kondermann, Daniel
and Bodenstedt, Sebastian
and Sanchez, Alexandro
and Stock, Christian
and Kenngott, Hannes Gotz
and Eisenmann, Mathias
and Speidel, Stefanie",
editor="Golland, Polina
and Hata, Nobuhiko
and Barillot, Christian
and Hornegger, Joachim
and Howe, Robert",
title="Can Masses of Non-Experts Train Highly Accurate Image Classifiers?",
booktitle="Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2014",
year="2014",
publisher="Springer International Publishing",
address="Cham",
pages="438--445",
isbn="978-3-319-10470-6"
}
"""

dset_name = "crowdsourced-EIS"
# make sure background goes first
class_mapping = {"background": (0, 0, 0), "instrument": (128, 0, 0)}


@register_dsetcreator(dset_name=dset_name)
def create_dset():
    dset_creator = DSetCreator(dset_name=dset_name)
    dset_creator.download(
        url="https://opencas.webarchiv.kit.edu/data/AnnotatedImages.zip",
        file_name="AnnotatedImages.zip",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    masks = list((dset_path / DataKind.TRAINING_DATA).rglob("*_inst_GT*.bmp"))
    out_base = dset_creator.transform_masks(
        masks=masks, load="rgb", transform={class_mapping["background"]: 0, class_mapping["instrument"]: 1}, train=True
    )
    assert out_base == dset_path / DataKind.TRAINING_LABELS / "transformed_masks", f"{out_base=}"
    return dset_path


@register_taskcreator(task_name="crowdsourced-endoscopic-instrument-segmentation-crowd-only", dset_name=dset_name)
def create_crowd(dset_path: Path):
    task = TaskCreator(
        dset_path=dset_path,
        name="crowdsourced-endoscopic-instrument-segmentation-crowd-only",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        desc="segmenting endoscopic instruments with labels provided by the crowd",
        ref=REFERENCE,
        url="https://opencas.webarchiv.kit.edu/?q=InstrumentCrowd",
        instr="download data archive via website",
        lic=License.UNKNOWN,
        release="2014",
        keywords=[Keyword.LAPAROSCOPY, Keyword.MEDICAL, Keyword.ENDOSCOPIC_INSTRUMENTS],
    )
    data_iterator = []
    for patient_id in [p.name for p in (dset_path / DataKind.TRAINING_DATA).iterdir() if p.is_dir()]:
        for img_id in [
            p.stem
            for p in (dset_path / DataKind.TRAINING_DATA / patient_id / "crowd").iterdir()
            if p.is_file() and "_inst_GTcrowd" not in p.name and p.suffix == ".bmp"
        ]:
            data_iterator.append(
                {
                    Modality.SAMPLE_ID: img_id,
                    Modality.IMAGE: dset_path / DataKind.TRAINING_DATA / patient_id / "crowd" / f"{img_id}.bmp",
                    Modality.MASK: (
                        dset_path
                        / DataKind.TRAINING_LABELS
                        / "transformed_masks"
                        / patient_id
                        / "crowd"
                        / f"{img_id}_inst_GTcrowd.png"
                    ),
                }
            )
    task.find_data(train_iterator=data_iterator, idx_to_class={ix: cls for ix, cls in enumerate(class_mapping.keys())})
    task.split_folds(n_folds=5, ensure_balancing=True)
    task.infer_stats()
    task.push_and_test()
