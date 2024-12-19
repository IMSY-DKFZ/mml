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
from mml.core.data_preparation.utils import get_iterator_and_mapping_from_image_dataset

REFERENCE = """
@inproceedings{KhoslaYaoJayadevaprakashFeiFei_FGVC2011,
author    = {Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei},
title     = {Novel Dataset for Fine-Grained Image Categorization},
booktitle = {First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition},
year      = {2011},
month     ={June},
address   = {Colorado Springs, CO},
}
"""

ADDITIONAL_REFERENCE = """"
@INPROCEEDINGS{5206848,
author    = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Kai Li and Li Fei-Fei},
booktitle = {2009 IEEE Conference on Computer Vision and Pattern Recognition}, 
title     = {ImageNet: A large-scale hierarchical image database}, 
year      = {2009},
pages     = {248-255},
doi       = {10.1109/CVPR.2009.5206848}}
"""  # noqa W291


@register_dsetcreator(dset_name="stanford_dogs")
def create_stanford_dogs():
    dset_creator = DSetCreator(dset_name="stanford_dogs")
    dset_creator.download(
        url="http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
        file_name="images.tar",
        data_kind=DataKind.TRAINING_DATA,
    )
    dset_path = dset_creator.unpack_and_store()
    return dset_path


@register_taskcreator(task_name="stanford_dogs_image_categorization", dset_name="stanford_dogs")
def create_stanford_dogs_image_categorization(dset_path: Path):
    dogs_categorization = TaskCreator(
        dset_path=dset_path,
        name="stanford_dogs_image_categorization",
        task_type=TaskType.CLASSIFICATION,
        desc="Stanford Dogs Dataset for Image Categorization",
        ref=REFERENCE + "\n" + ADDITIONAL_REFERENCE,
        url="http://vision.stanford.edu/aditya86/ImageNetDogs/",
        instr="download via vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
        lic=License.UNKNOWN,
        release="2011",
        keywords=[Keyword.ANIMALS, Keyword.NATURAL_OBJECTS],
    )
    classes = [
        "n02085620-Chihuahua",
        "n02085782-Japanese_spaniel",
        "n02085936-Maltese_dog",
        "n02086079-Pekinese",
        "n02086240-Shih-Tzu",
        "n02086646-Blenheim_spaniel",
        "n02086910-papillon",
        "n02087046-toy_terrier",
        "n02087394-Rhodesian_ridgeback",
        "n02088094-Afghan_hound",
        "n02088238-basset",
        "n02088364-beagle",
        "n02088466-bloodhound",
        "n02088632-bluetick",
        "n02089078-black-and-tan_coonhound",
        "n02089867-Walker_hound",
        "n02089973-English_foxhound",
        "n02090379-redbone",
        "n02090622-borzoi",
        "n02090721-Irish_wolfhound",
        "n02091032-Italian_greyhound",
        "n02091134-whippet",
        "n02091244-Ibizan_hound",
        "n02091467-Norwegian_elkhound",
        "n02091635-otterhound",
        "n02091831-Saluki",
        "n02092002-Scottish_deerhound",
        "n02092339-Weimaraner",
        "n02093256-Staffordshire_bullterrier",
        "n02093428-American_Staffordshire_terrier",
        "n02093647-Bedlington_terrier",
        "n02093754-Border_terrier",
        "n02093859-Kerry_blue_terrier",
        "n02093991-Irish_terrier",
        "n02094114-Norfolk_terrier",
        "n02094258-Norwich_terrier",
        "n02094433-Yorkshire_terrier",
        "n02095314-wire-haired_fox_terrier",
        "n02095570-Lakeland_terrier",
        "n02095889-Sealyham_terrier",
        "n02096051-Airedale",
        "n02096177-cairn",
        "n02096294-Australian_terrier",
        "n02096437-Dandie_Dinmont",
        "n02096585-Boston_bull",
        "n02097047-miniature_schnauzer",
        "n02097130-giant_schnauzer",
        "n02097209-standard_schnauzer",
        "n02097298-Scotch_terrier",
        "n02097474-Tibetan_terrier",
        "n02097658-silky_terrier",
        "n02098105-soft-coated_wheaten_terrier",
        "n02098286-West_Highland_white_terrier",
        "n02098413-Lhasa",
        "n02099267-flat-coated_retriever",
        "n02099429-curly-coated_retriever",
        "n02099601-golden_retriever",
        "n02099712-Labrador_retriever",
        "n02099849-Chesapeake_Bay_retriever",
        "n02100236-German_short-haired_pointer",
        "n02100583-vizsla",
        "n02100735-English_setter",
        "n02100877-Irish_setter",
        "n02101006-Gordon_setter",
        "n02101388-Brittany_spaniel",
        "n02101556-clumber",
        "n02102040-English_springer",
        "n02102177-Welsh_springer_spaniel",
        "n02102318-cocker_spaniel",
        "n02102480-Sussex_spaniel",
        "n02102973-Irish_water_spaniel",
        "n02104029-kuvasz",
        "n02104365-schipperke",
        "n02105056-groenendael",
        "n02105162-malinois",
        "n02105251-briard",
        "n02105412-kelpie",
        "n02105505-komondor",
        "n02105641-Old_English_sheepdog",
        "n02105855-Shetland_sheepdog",
        "n02106030-collie",
        "n02106166-Border_collie",
        "n02106382-Bouvier_des_Flandres",
        "n02106550-Rottweiler",
        "n02106662-German_shepherd",
        "n02107142-Doberman",
        "n02107312-miniature_pinscher",
        "n02107574-Greater_Swiss_Mountain_dog",
        "n02107683-Bernese_mountain_dog",
        "n02107683-Bernese_mountain_dog",
        "n02108000-EntleBucher",
        "n02108089-boxer",
        "n02108422-bull_mastiff",
        "n02108551-Tibetan_mastiff",
        "n02108915-French_bulldog",
        "n02109047-Great_Dane",
        "n02109525-Saint_Bernard",
        "n02109961-Eskimo_dog",
        "n02110063-malamute",
        "n02110185-Siberian_husky",
        "n02110627-affenpinscher",
        "n02110806-basenji",
        "n02110958-pug",
        "n02111129-Leonberg",
        "n02111277-Newfoundland",
        "n02111500-Great_Pyrenees",
        "n02111889-Samoyed",
        "n02112018-Pomeranian",
        "n02112137-chow",
        "n02112350-keeshond",
        "n02112706-Brabancon_griffon",
        "n02113023-Pembroke",
        "n02113186-Cardigan",
        "n02113624-toy_poodle",
        "n02113712-miniature_poodle",
        "n02113799-standard_poodle",
        "n02113978-Mexican_hairless",
        "n02115641-dingo",
        "n02115913-dhole",
        "n02116738-African_hunting_dog",
    ]
    data_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
        root=dset_path / DataKind.TRAINING_DATA / "Images", classes=classes
    )
    dogs_categorization.find_data(train_iterator=data_iterator, idx_to_class=idx_to_class)
    dogs_categorization.split_folds(n_folds=5, ensure_balancing=True)
    dogs_categorization.infer_stats()
    dogs_categorization.push_and_test()


if __name__ == "__main__":
    dset_path = create_stanford_dogs()
    create_stanford_dogs_image_categorization(dset_path)
