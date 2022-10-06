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
import pandas as pd
import torch
import tqdm

from joblib import Parallel, delayed, parallel_backend
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


cv.setNumThreads(0)
cv.ocl.setUseOpenCL(False)

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
            cache=False,
            num_worker=0,
            channels=1,
            unet=False,
            N=0,
            M=0,
            pretrain=False,
            datasets = ["ChexPert"],
    ):

        # ----- Variable definition ------------------------------------------------------


        self.classes = names

        self.img_dir = img_dir
        self.annotation_files = {}

        self.label_smoothing = label_smoothing

        self.prob = prob if prob else [0, ] * 6


        self.intensity = intensity
        self.img_size = img_size
        self.cache = cache
        self.channels = channels
        self.unet = unet
        self.split = split

        # ----- Transform definition ------------------------------------------------------

        self.preprocess = self.get_preprocess(channels, img_size)
        self.transform = self.get_transform(self.prob, intensity)
        self.advanced_transform = self.get_advanced_transform(self.prob, intensity, N, M)

        # ------- Caching & Reading -----------------------------------------------------------
        classnames = []#["Lung Opacity", "Enlarged Cardiomediastinum"] if pretrain else []

        if split == "test_chexpert" :
            self.files = MongoDB("10.128.107.212", 27017, datasets).dataset("Train", classnames=classnames)
            self.files = self.files.loc[self.files['Path'].str.contains("valid", case=False)]

        else :
            self.files = MongoDB("10.128.107.212", 27017, datasets).dataset(split, classnames=classnames)

        self.files[self.classes] = self.files[self.classes].astype(int)

        #mask = self.files[self.classes].values.sum(axis=1)>0
        #self.files = self.files.loc[mask]
        self.img_dir = img_dir




        self.img_dir = img_dir
        weights = self.samples_weights()
        paths=self.files.groupby("Exam ID")["Path"].apply(list)
        frontal_lateral = self.files.groupby("Exam ID")["Frontal/Lateral"].apply(list)
        self.files=self.files[self.classes+["Exam ID"]].groupby("Exam ID").mean().round(0)
        self.files["Path"]=paths
        self.files["Frontal/Lateral"]=frontal_lateral

        if self.cache: #if images are stored in RAM : CAREFUL! VERY RAM intensive
            with parallel_backend('threading', n_jobs=num_worker):
                self.images = Parallel()(
                    delayed(self.read_img_from_disk)(paths=self.files.iloc[idx]['Path'],views=self.files.iloc[idx]['Frontal/Lateral']) for idx in
                    tqdm.tqdm(range(0, len(self.files))))
            self.read_img = lambda idx : self.images[idx]
        else :
            self.read_img = lambda idx : self.read_img_from_disk(paths=self.files.iloc[idx]['Path'],views=self.files.iloc[idx]['Frontal/Lateral'])


        if split == "Train" and not pretrain:
            self.weights = weights
        else:
            self.weights = None

        self.files.reset_index(inplace=True)
    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_transform(prob, intensity):  # for transform that would require pil images
        return A.Compose(
            [



                A.augmentations.geometric.transforms.Affine(translate_percent=15,rotate=45,shear=5,cval=0,keep_ratio=True,p=prob[1]),
                A.augmentations.CropAndPad(percent=(-0.1,0.1),p=prob[2]),

                A.augmentations.HorizontalFlip(p=prob[3]),

                A.GridDistortion(num_steps=5,distort_limit=3,interpolation=1,border_mode=4,value=None,mask_value=None,always_apply=False,p=prob[5]),

                A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.4,contrast_limit=0.4,always_apply=False,p=prob[4]),
                A.augmentations.transforms.RandomGamma()
                #A.augmentations.PixelDropout(dropout_prob=0.05,p=0.5),
                #gaussian blur?


            ]
        )
    @staticmethod
    def get_advanced_transform(prob, intensity, N, M):
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

    @staticmethod
    def get_preprocess(channels, img_size):
        """
        Pre-processing for the model . This WILL be applied before inference
        """
        if channels == 1:
            normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        else :
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        return transforms.Compose(
            [

            #   transforms.CenterCrop(img_size),
                transforms.Resize(img_size),
                transforms.ConvertImageDtype(torch.float32),
                normalize,
            ]
        )

    def samples_weights(self):
        """
        This function returns weights 1/class_count for each image in the dataset such that each class
        is seen in similar amount
        """
        data = copy.copy(self.files).fillna(0)
        data = data.replace(-1,0.75)
        data=data.groupby("Exam ID").mean().round(0)
        data = data[self.classes]
        data = data.astype(int)

        count = data.sum().to_numpy()

        for name,cat_count in zip(self.classes,count) :
            if cat_count == 0:
                print(f"Careful! The category {name} has 0 images!")
        count[-1] /=2 #lets double the number of empty images we will give to the model
        self.count = count
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
        views=np.array(views)

        frontal_views=np.where(views=="F")[0]
        lateral_views=np.where(views=="L")[0]
        if len(frontal_views)>0 :
            frontal_path=paths[np.random.permutation(frontal_views)[0]]

            frontal = cv.imread(f"{self.img_dir}{frontal_path}", cv.IMREAD_GRAYSCALE)
            frontal = cv.resize(
                frontal,
                (int(self.img_size), int(self.img_size)), cv.INTER_CUBIC,  # removed center crop
            )

        else :
            frontal=np.zeros((self.img_size,self.img_size))
        if len(lateral_views) > 0:
            lateral_path = paths[np.random.permutation(lateral_views)[0]]
            lateral = cv.imread(f"{self.img_dir}{lateral_path}", cv.IMREAD_GRAYSCALE)
            lateral = cv.resize(
                lateral,
                (int(self.img_size), int(self.img_size)), cv.INTER_CUBIC,  # removed center crop
            )
        else :
            lateral=np.zeros((self.img_size,self.img_size))
        assert len(lateral_views)+len(frontal_views)>0







        return frontal,lateral

    def __getitem__(self, idx) :

        frontal,lateral= self.read_img(idx)
        label = self.get_label(idx)

        if self.split == "Train" :
            images = self.transform(image=np.concatenate([frontal[None,:,:],lateral[None,:,:]],axis=0).astype(np.float32))["image"]
            frontal,lateral=images[0],images[1]




        frontal = torch.tensor(
            frontal,
            dtype=torch.uint8,
        )[None, :, :]
        lateral = torch.tensor(
            lateral,
            dtype=torch.uint8,
        )[None, :, :]

        # if self.channels == 3:
        #     image = image.repeat((3, 1, 1))

        return frontal,lateral, label.float(),idx



if __name__ == "__main__" :
    os.environ["DEBUG"] = "False"
    img_dir = os.environ["img_dir"]
    train = CXRLoader(split="Train", img_dir=img_dir, img_size=240, prob=None, intensity=0, label_smoothing=0,
                      cache=False, num_worker=0, channels=1, unet=False, N=0, M=0, pretrain=False, datasets = ["ChexPert"])
    valid = CXRLoader(split="Valid", img_dir=img_dir, img_size=240, prob=None, intensity=0, label_smoothing=0,
                      cache=False, num_worker=0, channels=1, unet=False, N=0, M=0, pretrain=False, datasets = ["ChexPert"])
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
