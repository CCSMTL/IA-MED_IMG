import pymongo



class MongoDB :
    def __init__(self,address,port,collectionnames):
        self.client = pymongo.MongoClient(address,port)
        self.db_public = self.client["Public_Images"]
        self.db_CIUSSS = self.client["CIUSSS"]
        self.data = []
        for name in collectionnames :
            self.data.append(self.db_public[name])
        #self.data.append(self.db_CIUSSS[$put_name_here$])

    def train(self):
        train_dataset=pd.DataFrame([])
        query = {"Train": {'$in': [ "1", "-1" ]}}
        for collection in self.data :
            results = collect.find(query)
            for item in results :
                pd.concat([train_dataset,pd.Series(item["x"])],axis=0)

        return train_dataset
    def valid(self):
        val_dataset = pd.DataFrame([])
        query = {"Valid": {'$in': ["1", "-1"]}}
        for collection in self.data:
            results = collect.find(query)
            for item in results:
                pd.concat([val_dataset, pd.Series(item["x"])], axis=0)

        return val_dataset

    def pretrain(self):
        pass