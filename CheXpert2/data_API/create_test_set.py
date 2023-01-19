# Created by Jonathan Beaulieu-Emond
# 2022-11-21

import pymongo
from CheXpert2 import names
from CheXpert2.dataloaders.MongoDB import MongoDB
import pandas as pd
from functools import reduce
import random

# ----General config-----
address = "10.128.107.212"
port = 27017
client = pymongo.MongoClient(address, port)
collection = client["CIUSSS"]["images"]
names = ["Lung Opacity"] + names[3:]
# ---- Main Script
def main():
    global collection, names
    test_set = []
    for name in names:

        print(name)
        # query = {name : {"$in" : [-1,1]}}
        # results = list(collection.find(query))
        # data = pd.DataFrame(results)
        # for column in names:
        #     if column not in data.columns:
        #         data[column] = 0
        # n = len(data)
        # assert n>20, f"Only {n} results found for {name}"
        data = MongoDB(address, port, ["CIUSSS"], use_frontal=False).dataset(
            "Valid", classnames=[name]
        )
        data["_id"] = data["_id"].astype(str)
        columns = data.columns
        ids = data.groupby("Exam ID")["_id"].apply(list).tolist()
        views = data.groupby("Exam ID")["Frontal/Lateral"].apply(list).tolist()

        n = len(ids)
        indexes = random.choices(list(range(0, n)), k=200)
        exams = [ids[index] for index in indexes]
        exams_views = [views[index] for index in indexes]
        n = 0
        for ids, views in zip(exams, exams_views):
            if len(ids) < 2:
                continue
            ids = ids[0:2]
            views = views[0:2]
            if n == 20:
                break
            if not ("F" in views and "L" in views):
                continue

            n += 1

            for id, view in zip(ids, views):

                element = data[data["_id"] == id].values.tolist()
                test_set.append(element[0])

        assert n == 20, f"Only {n} element for {name}"

    df = pd.DataFrame(test_set, columns=columns)
    df.to_csv("test_set.csv")
    return df


if __name__ == "__main__":
    df = main()
    df.drop_duplicates(inplace=True)

    if "test" in client["CIUSSS"].list_collection_names():
        client["CIUSSS"].drop_collection("test")

    new_collection = client["CIUSSS"]["test"]
    new_collection.insert_many(df.to_dict("records"))
