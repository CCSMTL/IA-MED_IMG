import numpy as np
from functools import reduce
import os
import pandas as pd
import pymongo
import yaml
import urllib

class MongoDB:
    def __init__(self, address, port, collectionnames):
        #assert urllib.request.urlopen(f"{address}:{port}").getcode() == 200 #make sure connection is up

        self.client = pymongo.MongoClient(address, port)
        self.db_public = self.client["Public_Images"]
        self.db_CIUSSS = self.client["CIUSSS"]

        self.data = [self.db_CIUSSS["images"]]

        with open("data/data.yaml", "r") as stream:
            columns = yaml.safe_load(stream)["names"]

        # columns.remove("Age")# TODO : Fix this
        #columns.remove("Lung Opacity")
        #columns.remove("Pleural Other")
        # columns.remove("Enlarged Cardiomediastinum")
        #columns.remove("Pleural Thickening")
        self.names = columns

        for name in collectionnames:
            self.data.append(self.db_public[name])

        if os.environ["DEBUG"] == "True":
            self.data =  [self.db_public["ChexPert"]]


    def dataset(self, datasetname, classnames):
        assert datasetname == "Train" or datasetname == "Valid"
        train_dataset = []

        if os.environ["DEBUG"] == "True" :
            query =  {'Path':{'$regex':datasetname.lower()}}
        else :
            query = {datasetname: 1}

        if len(classnames) > 0:
            query["$or"] = [{classname: {"$in" : [1,-1]}} for classname in classnames]

        for collection in self.data:
            results = list(collection.find(query))
            print(f"Collected query for dataset {collection}")

            if len(results) > 0:
                columns = results[0].keys()
                data=pd.DataFrame(results, columns=columns)
                data = data[self.names + ["Path"]]
                data[self.names] = data[self.names].astype(np.int32)
                train_dataset.append(data)

        if len(train_dataset) > 1:
            columns = self.names + ["Path"]
            # columns = list(columns)
            # columns.remove("AP/PA")

            df = reduce(lambda left, right: pd.merge(left, right, on=columns, how='outer'), train_dataset)
        elif len(train_dataset) == 1:
            df = train_dataset[0]
        else:
            raise Exception("No data found")
        df.fillna(0, inplace=True)
        return df


if __name__ == "__main__":
    import yaml

    os.environ["DEBUG"] = "False"
    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]

    # db = MongoDB("10.128.107.212", 27017, ["ChexPert", "ChexNet", "ChexXRay"])

    db = MongoDB("10.128.107.212", 27017, ["ChexPert", "ChexNet"])
    print("database initialized")
    train = db.dataset("Train", [])
    print("training dataset loaded")
    valid = db.dataset("Valid", [])
    print("validation dataset loaded")
    valid.iloc[0:100].to_csv("valid.csv")
    # valid = valid[names]
    print(valid.head(100))
