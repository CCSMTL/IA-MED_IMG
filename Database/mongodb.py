import pymongo

myclient = pymongo.MongoClient("mongodb://10.128.107.212:27017/")

#Access to databases
print(myclient.list_database_names())

#Access to all documents
mydb = myclient["Public_Images"]
print(mydb.list_collection_names())

ChexPertCollection = mydb["ChexPert"]

#Find the first occurence in the ChexPert Collection
x = ChexPertCollection.find_one()
print(x)

    
#Return Cardiomegaly
CardiomegalyTrainQuery = { "Pleural Effusion": "1", "Train": "1" }
listCardiomegalyTrain = [];
for x in ChexPertCollection.find(CardiomegalyTrainQuery):
    listCardiomegalyTrain.append(x)
    
numCardiomegalyTrain = len(listCardiomegalyTrain)
print(numCardiomegalyTrain)

