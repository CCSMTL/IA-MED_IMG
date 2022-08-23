#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-23$

@author: Jonathan Beaulieu-Emond
"""
import pymongo


def main():
    client = pymongo.MongoClient("mongodb://10.128.107.212:27017/")
    db = client["Public_Images"]
    coll = db["ChexNet"]
    results = coll.find({"Path": {"$regex": "data\\\\"}})

    for item in results:
        path = item["Path"].split("\\")
        path2 = ""
        for i in range(1, len(path)):
            path2 = path2 + "/" + path[i]

        print(path2)

        coll.update_one(
            {
                "_id": item["_id"],

            },
            {
                "$set": {"Path": path2}
            }

        )


if __name__ == "__main__":
    main()
