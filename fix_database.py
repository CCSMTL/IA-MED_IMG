#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-23$

@author: Jonathan Beaulieu-Emond
"""
import pandas as pd
import pymongo


def main():
    client = pymongo.MongoClient("mongodb://10.128.107.212:27017/")
    db = client["Public_Images"]
    coll = db["ChexPert"]
    # results = coll.find({"Path": {"$regex": "data\\\\"}})
    results = coll.find({})

    train = pd.read_csv("data/public_data/ChexPert/train.csv").fillna(0)

    keys = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    for item in results:
        csv_item = train[train["Path"] == item["Path"][27::]]
        for key in keys:
            if int(csv_item[key]) == -1:
                print(item["Path"][27::], key)

                if key == "No Finding":
                    key = "Normal"
                coll.update_one(
                    {
                        "_id": item["_id"],

                    },
                    {
                        "$set": {key: -1}
                    }

                )


if __name__ == "__main__":
    main()
