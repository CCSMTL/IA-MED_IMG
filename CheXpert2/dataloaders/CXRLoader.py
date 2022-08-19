#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30$

@author: Jonathan Beaulieu-Emond
"""

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from joblib import Parallel, delayed, parallel_backend
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
from CheXpert2 import custom_Transforms
from CheXpert2.dataloaders.MongoDB import MongoDB


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
            dataset="Train",
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
    ):
        # ----- Variable definition ------------------------------------------------------

        with open("data/data.yaml", "r") as stream:
            self.classes = yaml.safe_load(stream)["names"]

        self.length = 0
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

        # ----- Transform definition ------------------------------------------------------

        self.preprocess = self.get_preprocess(channels, img_size)

        self.transform = self.get_transform(self.prob, intensity)
        self.advanced_transform = self.get_advanced_transform(self.prob, intensity, N, M)

        # ------- Caching & Reading -----------------------------------------------------------
        classnames = ["Lung Opacity", "Enlarged Cardiomediastinum", "Pneumonia"] if pretrain else []
        # ,"ChexNet","ChexXRay"
        try :
            100/0
            self.files = MongoDB("10.128.107.212", 27017, ["ChexPert"]).dataset(dataset,classnames=classnames)
            self.img_dir=""
        except :
            # TODO : remove in future version
            self.files = pd.read_csv(f"{img_dir}{dataset.lower()}.csv") #backup for compatiility
            warnings.warn("Using deprecated dataset ; will be removed in next version")

            if dataset == "Train" and pretrain:
                for classname in classnames :
                    self.files = self.files[self.files[classname] == 1]




        if self.cache:
            with parallel_backend('threading', n_jobs=num_worker):
                self.images = Parallel()(
                    delayed(self.read_img)(f"{self.img_dir}/{self.files.iloc[idx]['Path']}") for idx in
                    tqdm.tqdm(range(0, len(self.files))))

    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_transform(prob, intensity):  # for transform that would require pil images
        return transforms.Compose(
            [
                transforms.RandomErasing(prob[3], (intensity, intensity)),
                transforms.RandomHorizontalFlip(p=prob[4]),
                transforms.GaussianBlur(3, sigma=(0.1, 2.0))  # hyperparam kernel size
            ]
        )

    @staticmethod
    def get_advanced_transform(prob, intensity, N, M):
        return transforms.Compose(
            [  # advanced/custom
                custom_Transforms.RandAugment(prob=prob[0], N=N, M=M),  # p=0.5 by default
                custom_Transforms.Mixing(prob[1], intensity),
                custom_Transforms.CutMix(prob[2], intensity),

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
        labels[vector == -1] = torch.rand(size=(len(vector[vector == -1]),)) * (0.85 - 0.55) + 0.55

        return torch.from_numpy(labels)

    @staticmethod
    def get_preprocess(channels, img_size):
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

    def read_img(self, file):

        image = cv.imread(file, cv.IMREAD_GRAYSCALE)
        if image is None:
            raise Exception("Image not found by cv.imread: " + file)



        image = cv.resize(
            image,
            (int(self.img_size* 1.14), int(self.img_size* 1.14)),cv.INTER_AREA ,  # 256/224 ratio
        )

        image = torch.tensor(
            image * 255,
            dtype=torch.uint8,
        )[None, :, :]

        if self.channels == 3:
            image = image.repeat((3, 1, 1))

        return image

    def __getitem__(self, idx):
        if self.cache:
            image = self.images[idx]
        else:
            image = self.read_img(f"{self.img_dir}{self.files.iloc[idx]['Path']}")
        label = self.get_label(idx)

        return image, label.float()



if __name__ == "__main__" :

    train = CXRLoader(dataset="Train",img_dir="data/", img_size=240, prob=None, intensity=0, label_smoothing=0, cache=False, num_worker=0, channels=1, unet=False, N=0, M=0, pretrain=True)
    valid = CXRLoader(dataset="Valid",img_dir="data/", img_size=240, prob=None, intensity=0, label_smoothing=0, cache=False, num_worker=0, channels=1, unet=False, N=0, M=0, pretrain=False)

    for dataset in [train,valid] :
        for image,label in dataset :
            print(image.shape, label.shape)
            break

