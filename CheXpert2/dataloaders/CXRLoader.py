#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30$

@author: Jonathan Beaulieu-Emond
"""

import copy
import os

import cv2 as cv
import numpy as np

import torch

import logging

from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from CheXpert2 import custom_Transforms
from CheXpert2.dataloaders.MongoDB import MongoDB
from CheXpert2 import names

# classes = [
#     "Cardiomegaly","Emphysema","Effusion","Lung Opacity",
#     "Lung Lesion"",Pleural Effusion","Pleural Other","Fracture",
#     "Consolidation","Hernia","Infiltration","Mass","Nodule","Atelectasis",
#     "Pneumothorax","Pleural_Thickening","Fibrosis","Edema","Enlarged Cardiomediastinum",
#     "Opacity","Lesion","Normal"




class CXRLoader(Dataset):
    """
    This is the dataloader for our classification models. It returns the image and the corresponding class
    """

    def __init__(
            self,
            split="Train",
            img_dir = "data",
            img_size=240,
            prob=None,
            intensity=0,
            label_smoothing=0,
            channels=1,
            use_frontal=False,
            datasets=None,
    ) :

        # ----- Variable definition ------------------------------------------------------

        assert datasets is not None, "You must specify the datasets to use"
        self.classes = names
        self.img_dir = img_dir
        self.annotation_files = {}
        self.label_smoothing = label_smoothing
        self.prob = prob if prob else [0, ] * 6
        self.intensity = intensity
        self.img_size = img_size
        self.channels = channels
        self.split = split

        # ----- Transform definition ------------------------------------------------------


        self.transform = self.get_transform(self.prob)
        self.advanced_transform = self.get_advanced_transform(self.prob, intensity)

        # ------- Caching & Reading -----------------------------------------------------------
        classnames = []#["Lung Opacity", "Enlarged Cardiomediastinum"] if pretrain else []


        self.files = MongoDB("10.128.107.212", 27017, datasets,use_frontal=use_frontal).dataset(split, classnames=classnames)




        self.files[self.classes] = self.files[self.classes].astype(int)




        paths=self.files.groupby("Exam ID")["Path"].apply(list)
        frontal_lateral = self.files.groupby("Exam ID")["Frontal/Lateral"].apply(list)
        self.files=self.files[self.classes+["Exam ID"]].groupby("Exam ID").mean().round(0)
        self.files["Path"]=paths
        self.files["Frontal/Lateral"]=frontal_lateral

        self.read_img = lambda idx : self.read_img_from_disk(paths=self.files.iloc[idx]['Path'],views=self.files.iloc[idx]['Frontal/Lateral'])



        self.weights = self.samples_weights()


        self.files.reset_index(inplace=True)


    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_transform(prob):  # for transform that would require pil images
        return A.Compose(
            [



                A.augmentations.geometric.transforms.Affine(scale=(0.95,1.05),translate_percent=(0.05,0.05),rotate=(-15,15),shear=None,cval=0,keep_ratio=True,p=prob[1]),


                A.augmentations.HorizontalFlip(p=prob[2]),
                A.augmentations.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, always_apply=False,
                                                       p=prob[3]),
                A.GridDistortion(num_steps=5,distort_limit=0.3,interpolation=1,border_mode=0,value=None,mask_value=None,always_apply=False,p=prob[4]),

                A.ElasticTransform(alpha=0.2, sigma=25, alpha_affine=50, interpolation=1, value=None,p=prob[5], border_mode=cv.BORDER_CONSTANT),

                #A.augmentations.transforms.RandomGamma()
                #A.augmentations.PixelDropout(dropout_prob=0.05,p=0.5),
                #gaussian blur?


            ]
        )
    @staticmethod
    def get_advanced_transform(prob, intensity):
        return transforms.Compose(
            [  # advanced/custom
            #    custom_Transforms.RandAugment(prob=prob[0], N=N, M=M),  # p=0.5 by default
                 custom_Transforms.Mixing(prob[0], intensity),
            #    custom_Transforms.CutMix(prob[2], intensity),

            ]
        )

    def get_label(self, idx):
        """
        This function returns the labels as a vector of probabilities. The input vectors are taken as is from
        the chexpert dataset, with 0,1,-1 corresponding to negative, positive, and uncertain, respectively.
        """

        vector, label_smoothing = self.files[self.classes].iloc[idx, :].to_numpy(), self.label_smoothing


        # we will use the  U-Ones method to convert the vector to a probability vector TODO : explore other method
        # source : https://arxiv.org/pdf/1911.06475.pdf
        labels = np.zeros((len(vector),))
        labels[vector == 1] = 1 - label_smoothing
        labels[vector == 0] = label_smoothing

        if self.split == "Train" :
            labels[vector == -1] = torch.rand(size=(len(vector[vector == -1]),)) * (0.85 - 0.55) + 0.55
        else :
            labels[vector == -1] = 1 # we only output binary for validation #TODO : verify that

        labels = torch.from_numpy(labels)
        labels[-1] = 1 - labels[-1] #lets predict the presence of a disease instead of the absence




        return labels


    def samples_weights(self):
        """
        This function returns weights 1/class_count for each image in the dataset such that each class
        is seen in similar amount
        """
        data = copy.copy(self.files).fillna(0)
        data = data.replace(-1,0.5)
        data=data.groupby("Exam ID").mean().round(0)
        data = data[self.classes]
        data = data.astype(int)

        count = data.sum().to_numpy()
        self.count = count
        for name,cat_count in zip(self.classes,count) :
            if cat_count == 0:
                logging.warning(f"Careful! The category {name} has 0 images!")

        if self.split != "Train" :
            return None
        weights = np.zeros((len(data)))
        ex=0
        for i, line in data.iterrows() :
            vector = line.to_numpy()
            a = np.where(vector == 1)[0]
            if len(a) > 0:
                category = np.random.choice(a, 1)
            else:
                category = len(self.classes) - 1  # assumes last class is the empty class


            weights[ex] = 1 / (count[category])
            ex+=1

        return weights


    def step(self,idxs,pseudo_labels):#moving avg

        labels=self.files.loc[idxs,self.classes].to_numpy()
        new_labels = 0.9*labels+0.1*pseudo_labels
        self.files.loc[idxs,self.classes] = new_labels

    def read_img_from_disk(self, paths,views):
        # views=np.array(views)
        #
        # frontal_views=np.where(views=="F")[0]
        # lateral_views=np.where(views=="L")[0]
        # for path in paths :
        #     assert os.path.exists(self.img_dir+path) ,f"path does not exists : {self.img_dir}{path}"
        # if len(frontal_views)>0 :
        #     frontal_path=paths[np.random.permutation(frontal_views)[0]]
        #
        #     frontal = cv.imread(f"{self.img_dir}{frontal_path}", cv.IMREAD_GRAYSCALE)
        #     frontal = cv.resize(
        #         frontal,
        #         (int(self.img_size), int(self.img_size)), cv.INTER_CUBIC,  # removed center crop
        #     ).squeeze()
        #
        # else :
        #     frontal=np.zeros((self.img_size,self.img_size))
        #
        #
        # if len(lateral_views) > 0:
        #     lateral_path = paths[np.random.permutation(lateral_views)[0]]
        #     lateral = cv.imread(f"{self.img_dir}{lateral_path}", cv.IMREAD_GRAYSCALE)
        #     lateral = cv.resize(
        #         lateral,
        #         (int(self.img_size), int(self.img_size)), cv.INTER_CUBIC,  # removed center crop
        #     ).squeeze()
        # else :
        #     lateral=np.zeros((self.img_size,self.img_size))
        # assert len(lateral_views)+len(frontal_views)>0
        images=np.zeros((2,self.img_size,self.img_size),dtype=np.uint8)
        for i,path in enumerate(np.random.permutation(paths)) :
            images[i,:,:]=cv.resize(cv.imread(f"{self.img_dir}{path}", cv.IMREAD_GRAYSCALE),(self.img_size,self.img_size))
            if i==1 :
                break


        return images

    def __getitem__(self, idx) :

        images= self.read_img(idx)
        label = self.get_label(idx)

        if self.split == "Train" :
            for i,image in enumerate(images) :
                images[i,:,:] = self.transform(image=image)["image"]

        images = torch.tensor(
            images,
            dtype=torch.uint8,
        )


        return images, label.float(),idx



if __name__ == "__main__" :
    os.environ["DEBUG"] = "True"
    img_dir = os.environ["img_dir"]
    train = CXRLoader(split="Train", img_dir=img_dir, img_size=240, prob=None, intensity=0, label_smoothing=0,
                      cache=False, num_worker=0, channels=1, unet=False, N=0, M=0, datasets = ["ChexPert"])
    valid = CXRLoader(split="Valid", img_dir=img_dir, img_size=240, prob=None, intensity=0, label_smoothing=0,
                      cache=False, num_worker=0, channels=1, unet=False, N=0, M=0,  datasets = ["ChexPert"])
    print(len(train))
    print(len(valid))
    print(len(train.weights))
    print(len(valid.weights))
    i = 0
    for dataset in [train, valid]:
        for image, label in dataset:
            i += 1
            if i == 100:
                break
