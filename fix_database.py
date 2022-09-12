#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-23$

@author: Jonathan Beaulieu-Emond
"""
import pandas as pd
import pymongo
import numpy as np
import functools
import yaml
def fix_valid():
    client = pymongo.MongoClient("mongodb://10.128.107.212:27017/")
    db = client["Public_Images"]

    classes = {
        "Consolidation" : 0,
        "Atelectasis"   : 0,
        "Mass"          : 0,
        "Nodule"        : 0,
        "Lesion"        : 0,
        "Emphysema"     : 0,
        "Pneumothorax"  : 0,
        "Pneumonia"     : 0,
        "Pleural Other" : 0,
        "Fracture"      : 0,
        "Hernia"        : 0,
        "Infiltration"  : 0,
        "Pleural Thickening" : 0,
        "Fibrosis"      : 0,
        "Edema"         : 0,
        "Enlarged Cardiomediastinum" : 0,
        "Opacity" : 0,
        "Cardiomegaly"
        #"Endotracheal Tube Normal"
        #"Nasogastric Tube Normal"
        #"Central Veinous Catheter Normal"
        "Normal" : 1000 , #cant be counted
        "Central Veinous Catheter Abnormal"
        "Nasogastric Tube Abnormal"
        "Endotracheal Tube Abnormal"
    }
    names = list(classes.keys())
    train_dataset = []
    for collection in ["ChexNet","ChexPert","ChexXRay"] :
        coll = db[collection]
        results = list(coll.find({}))
        columns = results[0].keys()
        train_dataset.append(pd.DataFrame(results, columns=columns))

    data = functools.reduce(lambda left, right: pd.merge(left, right, on=list(columns), how='outer'), train_dataset)

    ids_lists=data.groupby("Patient ID")["_id"].apply(list).tolist()

    n=len(ids_lists)

    idx=np.random.permutation(n)
    train = np.zeros((n))
    train[idx[0:int(0.85*n)]]=1
    ids_lists = ids_lists[idx]
    for patient in ids_lists :

        diseases = data[data["_id"]==patient[-1]][names]
        diseases =diseases.astype(int)
        idx=np.where(diseases.to_numpy() == 1)
        diseases=diseases.loc(idx)
        #choose disease randomly if more than one is present




        classes[disease] +=1

        if classes[disease]<200 :
            #add to validation set
            train_label = 0
            val_label   = 1
        else :
            train_label = 1
            val_label   = 0

        for id in patient :
                    # coll.update_one(
                    #     {
                    #         "_id": id,
                    #
                    #     },
                    #     {
                    #         "$set": {
                    #             "Train" : int(train_label),
                    #             "Valid" : int(valid_label)
                    #         }
                    #     }
                    #
                    # )


def fix_patient_id() :
    client = pymongo.MongoClient("mongodb://10.128.107.212:27017/")
    db = client["Public_Images"]
    for collection in ["ChexNet", "ChexPert", "ChexXRay"]:
        coll = db[collection]
        results = list(coll.find({}))

        for item in results :


            coll.update_one(
                {
                    "_id": id,

                },
                {
                    "$set": {
                        "Patient ID" : collection+"_"+item["Patient_ID"],

                    }
                }

            )


if __name__ == "__main__":
    #main()
