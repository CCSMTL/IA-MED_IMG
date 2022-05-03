import torch
from torch.utils.data import Dataset
import os

import numpy as np
import cv2 as cv
from PIL import Image



class CustomImageDataset(Dataset):
    """
    This is the dataloader for our classification models. It returns the image and the corresponding class
    """
    def __init__(self,img_dir,num_classes, transform=None):

        self.img_dir = img_dir
        self.transform = transform
        self.length=0
        self.files=[]
        self.annotation_files={}
        self.num_classes=num_classes

        for file in os.listdir(img_dir+"/images") :
                self.files.append(f"{self.img_dir}/images/{file}")


    def __len__(self):
        return len(self.files)

    def label_transform(self,label_ids): # encode one_hot
        one_hot=torch.zeros((self.num_classes))
        for label_id in label_ids :
            if int(label_id)==self.num_classes : # the empty class!
                pass # we do nothing
            else :
                one_hot[int(label_id)]=1
        return one_hot

    def __getitem__(self, idx):
        img_path=self.files[idx]

        patterns=img_path.split("/")[::-1]

        keyname = patterns[0]

        label_file=f"{self.img_dir}/labels/{keyname[:-3]}txt"

        category_ids=[]

        if os.path.exists(label_file) :
            with open(label_file) as f:
                line=f.readlines()
                if len(line)>0 : # if file no empty

                    category_ids.append(line[0])
                else :
                    category_ids.append(14) # if the txt file is missing, we presume empty image
        else :
            category_ids.append(14)  # the image is empty



        image = cv.imread(img_path)
        if self.transform:
            image=Image.fromarray(np.uint8(image))
            image = self.transform(image)

        label=self.label_transform(category_ids)


        return image.float(), label.float()