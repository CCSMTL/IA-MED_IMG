import pandas as pd
import os
import tqdm

results = []


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
inv_map = {v: k for k, v in mapping.items()}

for dataset in ["training", "validation", "test"]:

    for file in tqdm.tqdm(os.listdir(f"../data/{dataset}/labels")):
        with open(f"../data/{dataset}/labels/{file}") as f:
            for line in f.readlines():
                line = [dataset] + line.split(" ")
                line[1] = inv_map[int(line[1])]
                results.append(line[0:5])

results = pd.DataFrame(
    results, columns=["dataset", "category", "age", "gender", "view"]
)

# graph time

stop = 1
