import csv
import os
import random
import math
from pathlib import Path
from tkinter.ttk import Separator
from tokenize import String
from PIL import Image

classes_name = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Lung Opacity', 'Lung Lesion', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                'Consolidation', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',
                'Enlarged Cardiomediastinum', 'Opacity', 'Pleural', 'Lesion',
                'No Finding', 'Normal']

column_names = ['Path', 'Patient ID', 'Sex', 'Age', 'Frontal/Latéral', 'AP/PA', 'Image Width', 'Image Height', 'Train', 'Valid']

ValidationRange = 0.15
IndexClasse = 10


fileTrain = open("E:\\CheXpert-v1.0\\train.csv")
fileValid = open("E:\\CheXpert-v1.0\\Valid.csv")
imageDirTrainSrc= 'E:\\CheXpert-v1.0\\train\\'
imageDirValidSrc= 'E:\\CheXpert-v1.0\\valid\\'
imageDirTrainDest= '/data/public_data/chexpert/CheXpert-v1.0/train/'
imageDirValidDest= '/data/public_data/chexpert/CheXpert-v1.0/valid/'
imagesDir = 'E:\\CheXpert-v1.0\\'

def GetFilePathAndName(fileName):
    path = Path().absolute(fileName)
    print(path)

def GetFileList(dirPath):
    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(dirPath):
        # check if current path is a file
        if os.path.isfile(os.path.join(dirPath, path)):
            res.append(path)

    return res

def GetSubDirList(rootDir):
    subdir = []
    for file in os.listdir(rootDir):
        d = os.path.join(rootDir, file)
        if os.path.isdir(d):
            subdir.append(d)
    return subdir

def GetNumRecordsPerClass(rows):

    classSizeList = []

    for className in classes_name:

        classNb = 0

        for row in rows:
            if row[1].find(className) != -1:
                classNb += 1
        
        classSizeList.append(classNb)

    print(classSizeList)

    return classSizeList

def DefineTrainVal(rows, classSizeList):

    
    for idx, classSize in enumerate(classSizeList):

        if classSize > 0:

            myList = []
            randClassList = []
            #Collect all row index with the class cell set to 1
            for idr, row in enumerate(rows):
                if row[IndexClasse+idx] == 1:
                    myList.append(idr)

            if len(myList) > 0:
                #Buld a random list for about 15% of the class size
                randClassList = random.sample(myList,math.floor(classSize*ValidationRange))
                

                #Mark the record like a validation record
                for nb in randClassList:
                        rows[nb][9] = 1
                        rows[nb][8] = 0

def ConvertPng2Jpeg(pngSrc, jpgDest):
    im = Image.open(pngSrc)
    im.convert('RGB').save(jpgDest,"JPEG") #this converts png image as jpeg
    im.close()

#Return the tuple size of a jpeg image
def GetJpgsize(jpegfile):
    im = Image.open(jpegfile)
    size = im.size
    im.close()
    return size

def main():
    listRowData = []
    rows = [];
    rowvs = [];

    # Remove the previous report csv file if it exists
    if os.path.exists(imagesDir+'ChexPert.csv'):
        os.remove(imagesDir+'ChexPert.csv')

    RowData = column_names + classes_name;
    listRowData.append(RowData)

    print("Hello World!")
    csvreaderTrain = csv.reader(fileTrain)
    csvreaderVal = csv.reader(fileValid)

    for row in csvreaderTrain:
        rows.append(row)
    for rowv in csvreaderVal:
        rowvs.append(rowv)

    # List of the image files
    SubDirTrainList = GetSubDirList(imageDirTrainSrc)
    fileValidList = GetSubDirList(imageDirValidSrc)

    print(len(SubDirTrainList), len(fileValidList))

    for i in range(1, len(rows), 1):

        splitStr = rows[i][0].split( "/",  -1)

        patientSplit = str(splitStr[2])
        patientId = patientSplit.replace('patient', '', 1)
        #Check the sex
        if rows[i][1] == 'Female': sex = 'F' 
        else: sex = 'M'
        #check Frontal or Lateral
        if rows[i][3] == 'Frontal':side = 'F'
        else: side = 'L'
        #The image size
        width = GetJpgsize('E:/' + rows[i][0])[0]
        height = GetJpgsize('E:/' + rows[i][0])[1]

        RowData = [imageDirTrainDest+rows[i][0], patientId, sex, rows[i][2], side, rows[i][4], width, height, '1', '0', '0', '0', '0','0', '0','0', '0','0', '0','0', '0','0', '0','0', '0','0', '0','0', '0',
        '0', '0','0', '0','0', '0']

        #Find the status in the initial chexpert csv file 
        for col in range(5, 18, 1):
            if rows[i][col] == '1.0':
                for idx, className in enumerate(classes_name):
                    if rows[0][col].find(className) != -1:
                        RowData[idx+IndexClasse] = 1
            if rows[i][col] == '-1.0':
                for idx, className in enumerate(classes_name):
                    if rows[0][col].find(className) != -1:
                        RowData[idx+IndexClasse] = -1
            

        listRowData.append(RowData)

    #for i,row in enumerate(rowvs, start=1):
    for i in range(1, len(rowvs), 1):

        splitStr = rowvs[i][0].split( "/",  -1)

        patientSplit = str(splitStr[2])
        patientId = patientSplit.replace('patient', '', 1)
    
        #Check the sex
        if rowvs[i][1] == 'Female': sex = 'F' 
        else: sex = 'M'
        #check Frontal or Lateral
        if rowvs[i][3] == 'Frontal':side = 'F'
        else: side = 'L'
        
        #The image size
        width = GetJpgsize('E:/' + rowvs[i][0])[0]
        height = GetJpgsize('E:/' + rowvs[i][0])[1]

        #Validation set to 1
        RowData = [imageDirValidDest+rowvs[i][0], patientId, sex, rowvs[i][2], side, rowvs[i][4], width, height, '0', '1', '0', '0', '0','0', '0','0', '0','0', '0','0', '0','0', '0','0', '0',
        '0', '0','0', '0','0', '0','0', '0','0', '0']

        for col in range(5, 18, 1):
           if rowvs[i][col] == '1.0':
               for idx, className in enumerate(classes_name):
                   if rowvs[0][col].find(className) != -1:
                       RowData[idx+IndexClasse] = 1
           if rowvs[i][col] == '-1.0':
               for idx, className in enumerate(classes_name):
                   if rowvs[0][col].find(className) != -1:
                       RowData[idx+IndexClasse] = -1
        
        listRowData.append(RowData)


    #Write the result in a csv file
    with open(imagesDir+'ChexPert.csv', 'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerows(listRowData)


    print ("End")

if __name__ == "__main__":
    main()