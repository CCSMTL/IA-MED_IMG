

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
    "Support Devices": 14,
    "Fracture": 15,
    "consolidation": 16,
    "lesion": 17,
    "Pleural Other": 18,
    "Enlarged Cardiom.": 19,
    "pleural": [10, 9, 18],
    "lung opacity": [13, 16, 11, 17, 8],
    "No Finding": 20,
}
import numpy as np
import pandas as pd

train = pd.read_csv("data/CheXpert-v1.0-small/train.csv")

valid = pd.read_csv("data/CheXpert-v1.0-small/valid.csv")

# step 1 : reformat chexpert csv

# train=train["Path","age","No Finding"	,"Enlarged Cardiomediastinum"	,"Cardiomegaly",	"Lung Opacity"	,"Lung Lesion"	,"Edema",	"Consolidation"	,"Pneumonia"	,"Atelectasis",	"Pneumothorax"	,"Pleural Effusion"	,"Pleural Other"	,"Fracture"	,"Support Devices"]
train.drop("Sex", inplace=True)
train.drop("Frontal/Lateral", inplace=True)
train["Path"] = "data/" + train["Path"]

chexnet = pd.read_csv("data/Data_Entry_2017_v2020.csv")
for i, row in enumerate(chexnet.iterrows()):
    empty = np.zeros((21))
    disease = row["Finding Labels"].split("|")
    empty[mapping[disease]] = 1

    path = f"chexnet/{row['Image Index']}"
    new_row = row["age"]
    new_row["Path"] = path
    new_row[list(mapping.keys())] = empty
    train.append(new_row)

train.save("new_train.csv")
