import numpy as np
from functools import reduce
import os
import pandas as pd
import pymongo
import yaml
import urllib
from CheXpert2 import names
import logging
class MongoDB:
    def __init__(self, address, port, collectionnames,use_frontal=False):

        self.client = pymongo.MongoClient(address, port)
        self.db_public = self.client["Public_Images"]

        self.data = []
        self.collectionnames = collectionnames

        if "CIUSSS" in collectionnames :
            self.db_CIUSSS = self.client["CIUSSS"]
            self.data.append(self.db_CIUSSS["images"])
            collectionnames.remove("CIUSSS")

        for collectionname in collectionnames:
            assert collectionname in self.db_public.list_collection_names()

        columns=names

        self.use_frontal = use_frontal
        self.names = columns + ["Path","collection","Exam ID","Frontal/Lateral"]

        for name in collectionnames:
            self.data.append(self.db_public[name])


    def dataset(self, datasetname, classnames):
        assert datasetname == "Train" or datasetname == "Valid"
        train_dataset = [pd.DataFrame([],columns=self.names)]
        query = {datasetname: 1}

        if self.use_frontal:
            query["Frontal/Lateral"] = "F"
        if len(classnames) > 0:
            query["$or"] = [{classname: 1} for classname in classnames]

        for collection in self.data:
            results = list(collection.find(query))


            logging.info(f"Collected query for dataset {collection}")

            if len(results) > 0:

                data=pd.DataFrame(results)

                #data = data[self.names + ["Path"]]
                data["collection"] = collection.name
                #data[self.names] = data[self.names].astype(np.int32)
                for column in self.names :
                    if column not in data.columns :
                        data[column] = 0
                data["Patient ID"] = data["Patient ID"].astype(str)
                data["Exam ID"] = data["Exam ID"].astype(str)
                data.set_index("Patient ID")
                train_dataset.append(data)

        if len(train_dataset) > 1:
            df = reduce(lambda left, right: pd.merge(left, right,on=self.names, how='outer'), train_dataset)
        else:
            raise Exception("No data found")


        #set up parent class
        df.fillna(0, inplace=True)

        df["Opacity"] = df[["Consolidation","Atelectasis","Mass","Nodule","Lung Lesion"]].replace(-1,1).max(axis=1)
        df["Air"]     = df[["Emphysema","Pneumothorax","Pneumo other"]].replace(-1,1).max(axis=1)
        df["Liquid"]  = df[["Edema","Pleural Effusion"]].replace(-1, 1).max(axis=1)
        df.fillna(0, inplace=True)
        df[self.names[:-4]] = df[self.names[:-4]].astype(int)
        #df.to_csv("test.csv",sep=" ")
        return df


if __name__ == "__main__":
    import yaml

    os.environ["DEBUG"] = "True"
    from CheXpert2 import names

    # db = MongoDB("10.128.107.212", 27017, ["ChexPert", "ChexNet", "ChexXRay"])

    db = MongoDB("10.128.107.212", 27017, ["ChexPert"])
    print("database initialized")
    train = db.dataset("Train", [])
    print("training dataset loaded")
    valid = db.dataset("Valid", [])
    print("validation dataset loaded")
    valid.iloc[0:100].to_csv("valid.csv")
    # valid = valid[names]
    print(len(train),len(valid))
