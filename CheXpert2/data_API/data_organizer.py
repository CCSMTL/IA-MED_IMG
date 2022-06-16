import os
import shutil
import pandas as pd
from pathlib import Path
import numpy as np

data_folder = "data/images"
final_folder = "data/"
n_images = 30805 + 1


# training 70 , val 20, test 10 %
n_train = int(0.7 * n_images)
n_val = int(0.2 * n_images)
n_test = n_images - n_val - n_train

mapping = {
    "Cardiomegaly": 0,
    "Emphysema": 1,
    "Effusion": 2,
    "Consolidation": 3,
    "Hernia": 4,
    "Infiltration": 5,
    "Mass": 6,
    "Nodule": 7,
    "Atelectasis": 8,
    "Pneumothorax": 9,
    "Pleural_Thickening": 10,
    "Pneumonia": 11,
    "Fibrosis": 12,
    "Edema": 13,
    "No Finding": 14,
}


def reformat_labels(labels):
    """

    :param labels: array of values extracted from csv
    :return: list of ids for each disease classes
    """
    class_name = labels[0]
    names = class_name.split("|")
    class_id = []
    for name in names:
        class_id.append(mapping[name])

    return class_id


if not os.path.exists("data/Data_Entry_2017_v2020.csv"):
    print(
        "Copier le fichier Data_Entry_2017_v2020 depuis : https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468"
    )
    exit()

labels = pd.read_csv("data/Data_Entry_2017_v2020.csv")
labels.set_index("Image Index", drop=True, inplace=True)

count = 0

for dataset in ["training/", "validation/", "test/"]:
    for folder in ["images", "labels"]:
        path = Path(final_folder + dataset + folder)
        if not path.exists():
            path.mkdir(parents=True)

root = "data/images"

if not Path(root).exists():
    print("Creating the data/images directory")
    Path(root).mkdir()
    # hard coded

    # importing the "tarfile" module https://www.geeksforgeeks.org/how-to-uncompress-a-tar-gz-file-using-python/
    import tarfile

    # open file
    file = tarfile.open("data/images_01.tar.gz")

    # extracting file
    file.extractall("data")

    file.close()

for i in range(0, n_images):
    j = 0
    file = f"{i:08}_000.png"
    while os.path.exists(f"{root}/{file}"):

        label = labels.loc[file].values
        label_ids = reformat_labels(label)

        if count < n_train:
            # add to training set
            shutil.copy(f"{root}/{file}", f"{final_folder}/training/images/{file}")
            f = open(f"{final_folder}training/labels/{file[:-3]}txt", "w")
        elif count < n_train + n_val:
            shutil.copy(f"{root}/{file}", f"{final_folder}/validation/images/{file}")
            f = open(f"{final_folder}validation/labels/{file[:-3]}txt", "w")
        else:
            shutil.copy(f"{root}/{file}", f"{final_folder}/test/images/{file}")
            f = open(f"{final_folder}test/labels/{file[:-3]}txt", "w")

        for k in label_ids:
            string = f"{k}  "
            for l in label[1::]:
                string += f"{l}   "
            f.write(string)
            f.write("\n")  # multiple diseases = multiple lines
        j += 1
        file = f"{i:08}_{j:03}.png"

    count += 1
