from functools import reduce
import os
import pandas as pd
import pymongo
import logging
from CheXpert2 import names, hierarchy
from pymongo.errors import ConnectionFailure


class MongoDB:
    def __init__(
        self, address, port, collectionnames, use_frontal=False, img_dir="", debug=False
    ):

        # -------- variable definition ---------------------------
        self.use_frontal = use_frontal
        self.names = names + ["Path", "Exam ID", "Frontal/Lateral"]
        self.img_dir = img_dir
        self.debug = debug
        self.collectionnames = collectionnames
        self.hierarchy = hierarchy
        self.online = True
        try:
            # The ismaster command is cheap and does not require auth.
            client = pymongo.MongoClient(address, port)
            client.admin.command("ismaster")
        except ConnectionFailure:
            logging.critical("Server not available ; switching offline")
            debug = True

        if debug:  # if debug is true, we are not using the database
            self.online = False
            self.data = collectionnames
            for collection in collectionnames:
                assert os.path.exists(
                    f"{self.img_dir}/data/{collection}.csv"
                ), f" {collection}is not available offline"
            return

        # -------- database connection ---------------------------
        self.client = pymongo.MongoClient(address, port)
        self.db_public = self.client["Public_Images"]
        self.data = []
        self.collectionnames = collectionnames

        if "CIUSSS" in collectionnames:
            self.db_CIUSSS = self.client["CIUSSS"]
            self.data.append(self.db_CIUSSS["images"])
            collectionnames.remove("CIUSSS")

        for collectionname in collectionnames:
            assert collectionname in self.db_public.list_collection_names()

        for name in collectionnames:
            self.data.append(self.db_public[name])

    def load(self, datasetname):
        # -------- load data from database ---------------------------

        if (
            datasetname == "Test"
        ):  # for testing we specified the subset of the CIUSSS collection
            # TODO : FIX OFFLINE TEST
            assert self.debug == False, "Offline test is not available"
            self.data = [self.db_CIUSSS["test"]]

        train_dataset = [pd.DataFrame([], columns=self.names)]
        query = {datasetname: 1}

        if self.use_frontal:
            # TODO : Fix mongodb frontal/lateral before using
            query["Frontal/Lateral"] = "F"
            raise NotImplementedError("Frontal/Lateral is not implemented yet")

        for collection in self.data:

            if self.online:
                results = list(collection.find(query))
                data = pd.DataFrame(results)
            else:  # using offline csv
                data = pd.read_csv(f"{self.img_dir}/data/{collection}.csv")
                data = data[data[datasetname] == 1]

            logging.info(f"Collected {datasetname} from dataset {collection}")

            if len(data) == 0:
                raise Exception(f"No data found for {datasetname} in {collection}")

            if (
                "Exam ID" not in data.columns
            ):  # TODO: remove this when the database is updated
                data["Exam ID"] = data["Patient ID"]

            for column in self.names:
                if column not in data.columns:
                    logging.critical(
                        f"Column {column} not found in the database for collection {collection}"
                    )
                    data[column] = 0

            data["Patient ID"] = data["Patient ID"].astype(str)
            data["Exam ID"] = data["Exam ID"].astype(str)
            data.set_index("Patient ID")
            train_dataset.append(data)

        df = pd.concat(train_dataset, ignore_index=True)

        return df

    def dataset(self, datasetname):
        assert datasetname in [
            "Train",
            "Valid",
            "Test",
        ], f"{datasetname} is not a valid choice. Please select Train,Valid, or Test"

        df = self.load(datasetname)
        # set up parent class
        df.fillna(0, inplace=True)

        for parent, children in self.hierarchy.items():
            if parent not in df.columns:
                df[parent] = df[children].replace(-1, 1).max(axis=1)
        df.fillna(0, inplace=True)
        df[self.names[:-4]] = df[self.names[:-4]].astype(int)
        # df.to_csv("test.csv",sep=" ")
        return df


if __name__ == "__main__":
    import yaml

    os.environ["DEBUG"] = "False"
    from CheXpert2 import names

    # db = MongoDB("10.128.107.212", 27017, ["ChexPert", "ChexNet", "ChexXRay"])

    valid = MongoDB("10.128.107.212", 27017, ["CIUSSS"]).dataset("Valid")

    train = MongoDB("10.128.107.212", 27017, ["CIUSSS"]).dataset("Train")

    for parent, children in hierarchy.items():
        print(parent, valid[parent].sum())
