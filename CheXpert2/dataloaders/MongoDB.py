from functools import reduce

import pandas as pd
import pymongo


class MongoDB:
    def __init__(self, address, port, collectionnames):
        self.client = pymongo.MongoClient(address, port)
        self.db_public = self.client["Public_Images"]
        self.db_CIUSSS = self.client["CIUSSS"]
        self.data = []
        for name in collectionnames:
            self.data.append(self.db_public[name])
        # self.data.append(self.db_CIUSSS[$put_name_here$])

    def dataset(self, datasetname, classnames):
        assert datasetname == "Train" or datasetname == "Valid"
        train_dataset = []
        query = {datasetname: {'$in': ["1", "-1"]}}
        for classname in classnames:
            query[classname] = {"$in": {["1", "-1"]}}

        for collection in self.data:
            results = list(collection.find(query))

            if len(results) > 0:
                train_dataset.append(pd.DataFrame(results, columns=results[0].keys()))

        df = reduce(lambda left, right: pd.merge(left, right, on=list(results[0].keys()), how='outer'), train_dataset)
        df.fillna(0, inplace=True)
        return df

    def pretrain(self):
        pass



if __name__  == "__main__" :
    db = MongoDB("localhost",27017,["ChexPert","ChexNet","ChexXRay"])

    valid=db.dataset("Valid")
    print(valid.head(100))
    valid.iloc[0:100].to_csv("valid.csv")