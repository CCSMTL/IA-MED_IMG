#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30$

@author: Jonathan Beaulieu-Emond
"""
import os
import warnings

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from CheXpert2 import Transforms


# TODO : ADD PROBABILITY PER AUGMENT CATEGORY


class chexpertloader(Dataset):
    """
    This is the dataloader for our classification models. It returns the image and the corresponding class
    """

    def __init__(
            self,
            img_file,

            img_size=240,
            prob=0,
            intensity=0,
            label_smoothing=0,
            cache=False,
            num_worker=0,
            channels=3,
            unet=False,
            N=2,
            M=9,
    ):
        # ----- Variable definition ------------------------------------------------------
        self.img_file = img_file

        self.length = 0
        self.files = []
        self.annotation_files = {}

        self.label_smoothing = label_smoothing
        self.prob = prob
        self.intensity = intensity
        self.img_size = img_size
        self.cache = cache
        self.channels = channels
        self.labels = []
        self.unet = unet

        # ----- Transform definition ------------------------------------------------------

        self.preprocess = self.get_preprocess(channels, img_size)

        self.transform = self.get_transform(prob)
        self.advanced_transform = self.get_advanced_transform(prob, intensity, N, M)

        # ------- Caching & Reading -----------------------------------------------------------

        self.files = pd.read_csv(img_file).fillna(0)

        if os.environ["DEBUG"] == "True":
            self.files = self.files[0:100]

    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_transform(prob):
        return transforms.Compose(
            [
                transforms.RandomErasing(p=prob),  # TODO intensity to add
            ]
        )

    @staticmethod
    def get_advanced_transform(prob, intensity, N, M):
        return transforms.Compose(
            [  # advanced/custom
                Transforms.RandAugment(prob=prob, N=N, M=M),  # p=0.5 by default
                Transforms.Mixing(prob, intensity),
                Transforms.CutMix(prob),
                Transforms.RandomErasing(prob),
            ]
        )

    @staticmethod
    def get_label(vector):
        """
        This function returns the labels as a vector of probabilities. The input vectors are taken as is from
        the chexpert dataset, with 0,1,-1 corresponding to negative, positive, and uncertain, respectively.
        """

        vector = vector.to_numpy()
        # we will use the  U-Ones method to convert the vector to a probability vector TODO : explore other method
        # source : https://arxiv.org/pdf/1911.06475.pdf
        labels = np.zeros((len(vector),))
        labels[vector == 1] = 1
        labels[vector == -1] = torch.rand(size=(len(vector[vector == -1]),)) * (0.85 - 0.55) + 0.55

        return labels

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
                transforms.Resize(int(img_size * 1.14)),  # 256/224 ratio
                transforms.CenterCrop(img_size),
                normalize,
            ]
        )

    def read_img(self, file):

        image = cv.resize(
            cv.imread(file, cv.IMREAD_GRAYSCALE),
            (self.img_size, self.img_size),
        )
        image = torch.tensor(
            image * 255,
            dtype=torch.uint8,
        )[None, :, :]

        if self.channels == 3:
            image = image.repeat((3, 1, 1))

        return image

    def __getitem__(self, idx):

        image = self.read_img("data/" + self.files.iloc[idx]["Path"])
        label = self.get_label(self.files.iloc[idx, 5:19])
        image = self.transform(image)

        if self.prob > 0:
            idx = torch.randint(0, len(self), (1,))
            image2, label2 = self.read_img("data/" + self.files.iloc[idx]["Path"])
            image2 = self.transform(image2)

            samples = {
                "image": image,
                "landmarks": label,
                "image2": image2,
                "landmarks2": label2,
            }
            image, label, image2, label2 = (self.advanced_transform(samples)).values()
            del samples, image2, label2
        image = self.preprocess(image.float())

        if self.unet:
            return image, image
        return image, label


if __name__ == "__main__":

    cxraydataloader = CxrayDataloader(
        img_dir="../data/test", num_classes=14, channels=3
    )  # TODO : build test repertory with 1 or 2 test image/labels

    # testing
    x = np.uint8(np.random.random((224, 224, 3)) * 255)
    to = transforms.ToTensor()
    for i in range(5):
        img = Image.fromarray(x)
        cxraydataloader.transform(img)
        samples = {
            "image": to(img),
            "landmarks": torch.zeros((14,)),
            "image2": to(img),
            "landmarks2": torch.zeros((14,)),
        }

        cxraydataloader.advanced_transform(samples)
        out = cxraydataloader[0]
        stop = 1
