import csv
import os
import shutil
import random
import math
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pydicom as dicom
from pydicom import dcmread
from pydicom.data import get_testdata_file

from pathlib import Path
from PIL import Image, ImageOps
from PIL.Image import fromarray



classes_name = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Lung Opacity', 'Lung Lesion', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                'Consolidation', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',
                'Enlarged Cardiomediastinum', 'Opacity', 'Pleural', 'Lesion',
                'No Finding', 'Normal']

column_names = ['Path', 'Patient ID', 'Sex', 'Age', 'Frontal/Latéral', 'AP/PA', 'Image Width', 'Image Height', 'Train', 'Valid']


ValidationRange = 0.15
IndexClasse = 10

imageDir = 'Y:\\Images_CIUSSS'
imageDirSrc= 'Y:\Images_CIUSSS'
imageDirValidationSrc= 'Y:\Images_CIUSSS'
imageDirDest= '\data\public_data\\Images_CIUSSS\\'

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

def GetDcmFileList(dirPath):
    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(dirPath):
        # check if current path is a file
        if os.path.isfile(os.path.join(dirPath, path)):
            if path.find('view') != -1 :
                if path.endswith('.dcm'):
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
    
def ConvertDicomToJpg(dcmSrc, jpgDest):
    
    ds = dcmread(dcmSrc)
            
    img = ds.pixel_array # get image array
    
    #Rescale it
    scaled_img = (np.maximum(img,0) / img.max()) * 255.0
    #Change from array to image
    img_mem = Image.fromarray(scaled_img)
    img_mem.convert('RGB').save(jpgDest,"JPEG")
    img_mem.close()

   
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

def main():
    listRowData = []
    rows = [];

    # Remove the previous report csv file if it exists
    if os.path.exists(imageDirSrc+'ChexCiusss.csv'):
        os.remove(imageDirSrc+'ChexCiusss.csv')

    #Start with a header defining the content of the columns
    RowData = column_names + classes_name;
    listRowData.append(RowData)

    print("Hello World!")

    #Read the content of the CIUSSS image directory
    # List of the image files
    SubDirList = GetSubDirList(imageDirSrc)

    for subDir in SubDirList:

        patientId = subDir.split('\\')[-1];

        #Remove all jpg files
        jpgFiles = glob.glob(subDir + '*.jpg')
        for jpg in jpgFiles:
            os.remove(jpg)
        
        #Collect the images
        dicomList = GetDcmFileList(subDir)
        

        for i,dcm in enumerate(dicomList) :
            strDir = subDir + '\\' + dcm
            jpgDest = subDir+'\\'+patientId+'_'+str(i)+'.jpg'
            
            #Read the Dicom
            ds = dcmread(strDir)
            
            img = ds.pixel_array # get image array  float(img.max())
            
            #Convert the array into an array of float
            y = np.maximum(img,0).astype(float)
            #Change for 8 bits
            scaled_img = (y) * (255.0/float(img.max()))
            #Build an image from the array
            img_mem = Image.fromarray(scaled_img)
            
            img_save = img_mem.convert('RGB')
            
            #Exception in the CIUSSS Image for which no explanation  about the inverted grey scale
            if(patientId == 1003409796371265) or (patientId == 2263864282359354):
                img_invert = ImageOps.invert(img_save)
                #Save with the max of quality as possible
                img_invert.save(jpgDest,"JPEG", quality=100)
                img_invert.close()
            else:
                #Save with the max of quality as possible
                img_save.save(jpgDest,"JPEG", quality=100)
                img_save.close()
            
            #Close everything
            img_mem.close()
            

            print(ds)
                



    #Write the result in a csv file
    if(len(listRowData) > 0):
        with open(imageDir+'ChexCiusss.csv', 'w', newline='') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerows(listRowData)

    print ("End")

if __name__ == "__main__":
    main()