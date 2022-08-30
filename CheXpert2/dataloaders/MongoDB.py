from functools import reduce
import os
import pandas as pd
import pymongo


class MongoDB:
    def __init__(self, address, port, collectionnames):
        self.client = pymongo.MongoClient(address, port)
        self.db_public = self.client["Public_Images"]
        self.db_CIUSSS = self.client["CIUSSS"]

        # self.data = [self.client["CIUSSS"]["images"]]
        self.data = []
        #TODO : in good time add ciusss images

        if os.environ["DEBUG"] == "True" :
            collectionnames = ["ChexPert"]
        for name in collectionnames:
            self.data.append(self.db_public[name])
        # self.data.append(self.db_CIUSSS[$put_name_here$])

    def dataset(self, datasetname, classnames):
        assert datasetname == "Train" or datasetname == "Valid"
        train_dataset = []
        query = {datasetname: {'$in': ["1", 1]}} #TODO : in future version of dataset remove string

        if len(classnames) > 0:
            query["$or"] = [{classname: {"$in": ["1", "-1",1,-1]}} for classname in classnames]

        for collection in self.data:
            results = list(collection.find(query))

            if len(results) > 0:
                columns = results[0].keys()
                train_dataset.append(pd.DataFrame(results, columns=columns))

        df = reduce(lambda left, right: pd.merge(left, right, on=list(columns), how='outer'), train_dataset)
        df.fillna(0, inplace=True)
        return df


if __name__ == "__main__":
    import yaml

    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]

    db = MongoDB("10.128.107.212", 27017, ["ChexPert", "ChexNet", "ChexXRay"])
    train = db.dataset("Train", ["Lung Opacity", "Enlarged Cardiomediastinum"])
    valid = db.dataset("Valid", [])
    valid.iloc[0:100].to_csv("valid.csv")
    valid = valid[names]
    print(valid.head(100))
