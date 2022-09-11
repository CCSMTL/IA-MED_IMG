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
import yaml
from joblib import Parallel, delayed, parallel_backend
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from CheXpert2 import custom_Transforms
from CheXpert2.dataloaders.MongoDB import MongoDB


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
            datasets = ["ChexPert", "ChexNet"],
    ):
        # ----- Variable definition ------------------------------------------------------

        with open("data/data.yaml", "r") as stream:
            self.classes = yaml.safe_load(stream)["names"]

        self.img_dir = img_dir
        self.annotation_files = {}

        self.label_smoothing = label_smoothing

        self.prob = prob if prob else [0, ] * 5
        if len(self.prob) == 1:
            self.prob = self.prob * 5

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




        if os.environ["DEBUG"] == "True" :
            #read local csv instead of contacting the database
            self.files = pd.read_csv(f"{img_dir}/data/public_data/ChexPert/ChexPert.csv").loc[0:100]
        else :
            self.files = MongoDB("10.128.107.212", 27017, datasets).dataset(split,classnames=classnames)
        self.files[self.classes] = self.files[self.classes].astype(int)
        self.img_dir = img_dir



        if self.cache: #if images are stored in RAM : CAREFUL! VERY RAM intensive
            with parallel_backend('threading', n_jobs=num_worker):
                self.images = Parallel()(
                    delayed(self.read_img_from_disk)(f"{self.img_dir}/{self.files.iloc[idx]['Path']}") for idx in
                    tqdm.tqdm(range(0, len(self.files))))
            self.read_img = lambda idx : self.images[idx]
        else :
            self.read_img = lambda idx : self.read_img_from_disk(f"{self.img_dir}{self.files.iloc[idx]['Path']}")

        weights=self.samples_weights()
        if split == "Train" and not pretrain:
            self.weights = weights
        else:
            self.weights = None


    def __len__(self):
        return len(self.files)

    # @staticmethod
    # def get_transform(prob, intensity):  # for transform that would require pil images
    #     return transforms.Compose(
    #         [
    #             transforms.RandomErasing(prob[3], (intensity, intensity)),
    #             transforms.RandomHorizontalFlip(p=prob[4]),
    #         #    transforms.GaussianBlur(3, sigma=(0.1, 2.0))  # hyperparam kernel size
    #         ]
    #     )
    @staticmethod
    def get_transform(prob, intensity):  # for transform that would require pil images
        return A.Compose(
            [
                #    transforms.RandomErasing(prob[3], (intensity, intensity)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation(degrees=90),
                # transforms.RandomAffine(degrees=45,translate=(0.2,0.2),shear=(-15,15,-15,15)),

                A.augmentations.geometric.transforms.Affine(translate_percent=20,rotate=25,shear=15,cval=0,keep_ratio=True,p=prob[0]),
                A.augmentations.geometric.transforms.ElasticTransform(alpha=1,sigma=50,approximate=True,p=prob[2]),
                #A.augmentations.crops.transforms.RandomResizedCrop(self.img_size,self.img_size,p=1),
                A.augmentations.transforms.VerticalFlip(p=prob[3]),
                A.augmentations.transforms.GridDistortion(num_steps=5,distort_limit=3,interpolation=1,border_mode=4,value=None,mask_value=None,always_apply=False,p=prob[4]),
                #A.augmentations.Superpixels(),
                A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,always_apply=False,p=prob[5]),
                #A.augmentations.PixelDropout(dropout_prob=0.05,p=0.5),


            ]
        )
    @staticmethod
    def get_advanced_transform(prob, intensity, N, M):
        return transforms.Compose(
            [  # advanced/custom
            #    custom_Transforms.RandAugment(prob=prob[0], N=N, M=M),  # p=0.5 by default
                 custom_Transforms.Mixing(prob[1], intensity),
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
            normalize = transforms.Normalize(mean=[0.456], std=[0.224])
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        return transforms.Compose(
            [

                transforms.CenterCrop(img_size),
                transforms.ConvertImageDtype(torch.float32),
                normalize,
            ]
        )

    def samples_weights(self):
        """
        This function returns weights 1/class_count for each image in the dataset such that each class
        is seen in similar amount
        """
        data = copy.copy(self.files[self.classes]).fillna(0)
        data = data.astype(int)
        data = data.replace(-1, 1)
        count = data.sum().to_numpy()
        self.count = count
        weights = np.zeros((len(data)))
        for i, line in data[self.classes].iterrows():
            vector = line.to_numpy()[5:19]
            a = np.where(vector == 1)[0]
            if len(a) > 0:
                category = np.random.choice(a, 1)
            else:
                category = len(self.classes) - 1  # assumes last class is the empty class
            weights[i] = 1 / (count[category])

        weights = np.nan_to_num(weights,nan=0,posinf=0,neginf=0)
        return weights

    def read_img_from_disk(self, file):

        image = cv.imread(file, cv.IMREAD_GRAYSCALE)

        if image is None:
            raise Exception("Image not found by cv.imread: " + file)

        image = cv.resize(
            image,
            (int(self.img_size* 1.14), int(self.img_size* 1.14)),cv.INTER_CUBIC ,  # 256/224 ratio
        )



        return image

    def __getitem__(self, idx) :

        image = self.read_img(idx)
        label = self.get_label(idx)

        if self.split == "Train" :
            image = self.transform(image=image)["image"]

        image = torch.tensor(
            image,
            dtype=torch.uint8,
        )[None, :, :]

        if self.channels == 3:
            image = image.repeat((3, 1, 1))
        return image, label.float()



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
