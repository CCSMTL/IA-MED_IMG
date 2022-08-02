import csv
import os
import random
import math
import glob
from pathlib import Path
from PIL import Image

classes_name = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Lung Opacity', 'Lung Lesion', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                'Consolidation', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',
                'Enlarged Cardiomediastinum', 'Opacity', 'Pleural', 'Lesion',
                'No Finding', 'Normal']

column_names = ['Path', 'Patient ID', 'Sex', 'Age', 'Frontal/Latéral', 'AP/PA', 'Image Width', 'Image Height', 'Train', 'Valid']

ValidationRange = 0.15
IndexClasse = 10

file = open("E:\\ChexNet\\Data_Entry_2017_v2020.csv")

imageDirSrc= 'E:\\ChexNet\Images\\'

imageDirDest= '\data\public_data\ChexNet\\Images\\'

type(file)

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

def main():
    listRowData = []
    rows = [];

    # Remove the previous report csv file if it exists
    if os.path.exists(imageDirSrc+'ChexNet.csv'):
        os.remove(imageDirSrc+'ChexNet.csv')

    #Remove all jpg files
    jpgFiles = glob.glob(imageDirSrc + '*.jpg')
    for jpg in jpgFiles:
        os.remove(jpg)

    RowData = column_names + classes_name;
    listRowData.append(RowData)

    print("Hello World!")
    csvreader = csv.reader(file)

    for row in csvreader:
        rows.append(row)

    # List of the image files
    fileList = GetFileList(imageDirSrc)

    # Get the file paths
    print (len(fileList), (len(rows)-1))

    if(len(fileList) == (len(rows)-1)):
        for i in range(len(fileList)):
            pathFileName = imageDirDest+fileList[i]
            pathFileName_WithoutExt = os.path.splitext(pathFileName)[0]
            pathFileName = pathFileName_WithoutExt + '.jpg'

            RowData = [pathFileName, rows[i+1][3], rows[i+1][5], rows[i+1][4], '0', rows[i+1][6], rows[i+1][7], rows[i+1][8], '1', '0']

            for className in classes_name:
                if rows[i+1][1].find(className) != -1:
                    RowData.append(1)
                else:
                    RowData.append(0)

            listRowData.append(RowData)

    classSizeList = GetNumRecordsPerClass(rows)
    DefineTrainVal(listRowData, classSizeList)

    #Write the result in a csv file
    #Remove the empty lines
    with open(imageDirSrc+'ChexNet.csv', 'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(listRowData)

    #Convert png into jpeg
    for fileName in fileList:
        fileName_without_ext = os.path.splitext(fileName)[0]
        ConvertPng2Jpeg(imageDirSrc+fileName, imageDirSrc+fileName_without_ext+'.jpg')

    print ("End")

if __name__ == "__main__":
    main()