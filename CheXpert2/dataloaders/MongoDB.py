
from functools import reduce
import os
import pandas as pd
import pymongo
import logging
from CheXpert2 import names
class MongoDB:
    def __init__(self, address, port, collectionnames,use_frontal=False,img_dir="",debug=False) :

        self.use_frontal = use_frontal
        self.names = names + ["Path", "collection", "Exam ID", "Frontal/Lateral"]
        self.img_dir = img_dir
        self.debug = debug

        if debug :
            assert collectionnames == ["ChexPert"]
            return

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

        for name in collectionnames:
            self.data.append(self.db_public[name])



    def load_online(self,datasetname):
        if datasetname=="Test" :
            self.data= [self .db_CIUSSS["test"]]
        train_dataset = [pd.DataFrame([],columns=self.names)]
        query = {datasetname: 1}

        if self.use_frontal:
            query["Frontal/Lateral"] = "F"

        for collection in self.data:
            results = list(collection.find(query))


            logging.info(f"Collected query for dataset {collection}")

            if len(results) > 0:

                data=pd.DataFrame(results)

                if "Exam ID" not in data.columns: #TODO: remove this when the database is updated
                    data["Exam ID"] = data["Patient ID"]


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

        return df
    def load_offline(self,datasetname):
        data = pd.read_csv(f"{self.img_dir}/data/ChexPert.csv")
        return data[data[datasetname]==1]
    def dataset(self, datasetname, classnames):
        assert datasetname in ["Train","Valid","Test"],f"{datasetname} is not a valid choice. Please select Train,Valid, or Test"

        if self.debug :
            df = self.load_offline(datasetname)
        else :
            df = self.load_online(datasetname)
        #set up parent class
        df.fillna(0, inplace=True)

        df["Opacity"] = df[["Consolidation","Atelectasis","Mass","Nodule","Lung Lesion"]].replace(-1,1).max(axis=1)
        df["Air"]     = df[["Emphysema","Pneumothorax","Pleural Other"]].replace(-1,1).max(axis=1)
        df["Liquid"]  = df[["Edema","Pleural Effusion"]].replace(-1, 1).max(axis=1)
        df.fillna(0, inplace=True)
        df[self.names[:-4]] = df[self.names[:-4]].astype(int)
        #df.to_csv("test.csv",sep=" ")
        return df


if __name__ == "__main__":
    import yaml

    os.environ["DEBUG"] = "False"
    from CheXpert2 import names

    # db = MongoDB("10.128.107.212", 27017, ["ChexPert", "ChexNet", "ChexXRay"])

    db = MongoDB("10.128.107.212", 27017, ["vinBigData"])
    print("database initialized")
    #train = db.dataset("Train", [])
    #print("training dataset loaded")
    valid = db.dataset("Valid", [])
    print("validation dataset loaded")
    valid.iloc[0:100].to_csv("valid.csv")
    # valid = valid[names]
    #print(len(train),len(valid))
