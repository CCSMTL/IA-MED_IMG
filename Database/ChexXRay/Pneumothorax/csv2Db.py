import csv
import os
import shutil
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

imageDir = 'E:\\chest_xray\\Pneumothorax\\'
imageDirTrainSrc = 'E:\\chest_xray\\Pneumothorax\\png_images'
imageDirTrainDest = '\\data\public_data\\chestXRay\\Pneumothorax\\'


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

#Return the tuple size of a jpeg image
def GetJpgsize(jpegfile):
    im = Image.open(jpegfile)
    size = im.size
    im.close()
    return size

def main():
    listRowData = []
    rows = [];

    # Remove the previous report csv file if it exists
    if os.path.exists(imageDirTrainSrc+'ChexXRayPneumothorax.csv'):
        os.remove(imageDirTrainSrc+'ChexXRayPneumothorax.csv')

    #Remove all jpg files
    jpgFiles = glob.glob(imageDirTrainSrc + '*.jpg')
    for jpg in jpgFiles:
        os.remove(jpg)

    #Start with a header defining the content of the columns
    RowData = column_names + classes_name;
    listRowData.append(RowData)

    print("Hello World!")

    #ALL files with Training status
    # List of the training image files
    fileList = GetFileList(imageDirTrainSrc)

    indexFile = 0
    fileWithIndexList = []
    for fileName in fileList:

        #Patient index
        filestr = fileName.split('_')
        # Path
        path = os.path.join(imageDirTrainSrc, 'Patient'+indexFile)
        os.mkdir(path)

        #Validation image
        if fileName.find('test') == True:
            #Build validation record
            RowData = [path+'//'+file, indexFile, 'U', 'U', 'U', 'U', width, height, '0', '1',
                    '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']




        
        #List of images of the same patient
        if fileName.startswith(indexFile+'_') == True:
            fileWithIndexList.append(imageDirTrainSrc+fileName)
        else:
            # Path
            path = os.path.join(imageDirTrainSrc, 'Patient'+indexFile)
            os.mkdir(path)

            #Move the files into the directory
            for file in fileWithIndexList:
                shutil.move (file, path)
                #The image size
                width = GetJpgsize(file)[0]
                height = GetJpgsize(file)[1]

                #Build training record
                RowData = [path+'//'+file, indexFile, 'U', 'U', 'U', 'U', width, height, '1', '0',
                '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

                #Pneumothorax class
                RowData[27] = '1'

                listRowData.append(RowData)

            #Increment the file index
            indexFile+=1
            fileWithIndexList = []

    #Write the result in a csv file
    #Remove the empty lines from the csv records
    with open(imageDir+'ChexXRayPneumothorax.csv', 'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(listRowData)


    print ("End")

if __name__ == "__main__":
    main()