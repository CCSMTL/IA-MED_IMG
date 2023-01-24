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

imageDir = 'E:\\chest_xray\\'
imageDirTrainSrc= 'E:\\chest_xray\\Pneumonia\\train\\'
imageDirValidationSrc= 'E:\\chest_xray\\Pneumonia\\val\\'
imageDirDest= '\data\public_data\chestXRay\\Pneumonia\\'

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

def GetPatientIdFromList(patientId, patientIdList):

    for Id in patientIdList:
        if Id == patientId:
            return True

    #Add the patient id into the patient id list
    patientIdList.append(patientId)
    return False

def SetFile(listRowData, DirSrc):

    patientIdList = []

    #Image with diseases
    pathPneumonia = DirSrc + 'PNEUMONIA\\'

    # List of the training image files
    fileList = GetFileList(pathPneumonia)

    patientIdList = []
    for fileName in fileList:

        indexStr = fileName.split('_')[0].replace('person', '')
        patientId = indexStr
        
        #The image size
        width = GetJpgsize(pathPneumonia+'\\' + fileName)[0]
        height = GetJpgsize(pathPneumonia+'\\' + fileName)[1]

        #List of images of the same patient
        path = os.path.join(pathPneumonia, 'Patient'+ patientId)

        if GetPatientIdFromList(patientId, patientIdList) == True:
            shutil.copyfile (pathPneumonia+'\\' + fileName, path + '\\' + fileName)
        else:
            #Remove the directory if it is exist
            shutil.rmtree(path, ignore_errors=True, onerror=None)
            os.mkdir(path)
            shutil.copyfile (pathPneumonia+'\\' + fileName, path + '\\' + fileName)

        #Build training record
        if DirSrc.find('train') != -1:
            RowData = [imageDirDest+ '//' + 'train' + '//' + 'Patient'+ str(patientId) +'//'+ fileName, patientId, 'U', 'U', 'U', 'U', width, height, '1', '0',
                '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        else:
            RowData = [imageDirDest+ '//' + 'val' + '//' + 'Patient'+ str(patientId) +'//'+ fileName, patientId, 'U', 'U', 'U', 'U', width, height, '0', '1',
                '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']            
                
        #Pneumonia class
        RowData[26] = '1'
        
        listRowData.append(RowData)

    #Increment the file index
    pathNormal = DirSrc + 'NORMAL\\'

    # List of the training image files
    fileList = GetFileList(pathNormal)

    patientIdList = []

    for fileName in fileList:

        #Find the first index in the filename
        x=fileName.split('-')

        if x[1].isdigit() == True:
            patientId = x[1]
        else:
            patientId = x[2]
    
        #The image size
        width = GetJpgsize(pathNormal+'\\' + fileName)[0]
        height = GetJpgsize(pathNormal+'\\' + fileName)[1]

        #List of images of the same patient
        path = os.path.join(pathNormal, 'Patient'+ patientId)

        if GetPatientIdFromList(patientId, patientIdList) == True:
            shutil.copyfile (pathNormal+'\\' + fileName, path + '\\' + fileName)
        else:
            #Remove the directory if it is exist
            shutil.rmtree(path, ignore_errors=True, onerror=None)
            os.mkdir(path)
            shutil.copyfile (pathNormal+'\\' + fileName, path + '\\' + fileName)

        #Build training record for validation or training set
        if DirSrc.find('train') != -1:
            RowData = [imageDirDest+'//' + 'train' + '//' + 'Patient'+ str(patientId) + fileName, patientId, 'U', 'U', 'U', 'U', width, height, '1', '0',
                '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        else:
            RowData = [imageDirDest+'//' + 'val' + '//' + 'Patient'+ str(patientId) + fileName, patientId, 'U', 'U', 'U', 'U', width, height, '0', '1',
                '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
    
        listRowData.append(RowData)

    return listRowData   

def main():
    listRowData = []
    rows = [];

    # Remove the previous report csv file if it exists
    if os.path.exists(imageDirTrainSrc+'ChexXRayPneumonia.csv'):
        os.remove(imageDirTrainSrc+'ChexXRayPneumonia.csv')

    #Start with a header defining the content of the columns
    RowData = column_names + classes_name;
    listRowData.append(RowData)

    print("Hello World!")

    #ALL files with Training status
    SetFile(listRowData, imageDirTrainSrc)

    #ALL files with validation  status
    SetFile(listRowData, imageDirValidationSrc)

    #Write the result in a csv file
    with open(imageDir+'ChexXRayPneumonia.csv', 'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(listRowData)

    print ("End")

if __name__ == "__main__":
    main()