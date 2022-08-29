#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-23$

@author: Jonathan Beaulieu-Emond
"""
import pandas as pd
import pymongo
import numpy as np

def main():
    client = pymongo.MongoClient("mongodb://10.128.107.212:27017/")
    db = client["Public_Images"]
    results=[]

    for collection in ["ChexNet","ChexPert","ChexXRay"] :
        coll = db[collection]
        results = list(coll.find({}))
        columns = results[0].keys()
        data = pd.DataFrame(results,columns=columns)
        ids=data.groupby("Patient ID")["_id"].apply(list).tolist()

        n=len(ids)

        idx=np.random.permutation(n)
        train = np.zeros((n))
        train[idx[0:int(0.85*n)]]=1


        for id,train_label in zip(ids,train):
            valid_label = 1 - train_label
            print(train_label,valid_label)
            coll.update_one(
                {
                    "_id": id,

                },
                {
                    "$set": {
                        "Train" : int(train_label),
                        "Valid" : int(valid_label)
                    }
                }

            )


if __name__ == "__main__":
    main()
