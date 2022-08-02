#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30$

@author: Jonathan Beaulieu-Emond
"""
import os

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import tqdm
from joblib import Parallel, delayed, parallel_backend
from torch.utils.data import Dataset
from torchvision import transforms

from CheXpert2 import custom_Transforms


class Chexpertloader(Dataset):
    """
    This is the dataloader for our classification models. It returns the image and the corresponding class
    """

    def __init__(
            self,
            img_file,
            img_dir="",
            img_size=240,
            prob=None,
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
        self.img_dir = img_dir
        self.length = 0

        self.annotation_files = {}

        self.label_smoothing = label_smoothing

        self.prob = prob if prob else [0, ] * 5
        if len(self.prob) == 1:
            self.prob = self.prob * 5

        self.intensity = intensity
        self.img_size = img_size

        self.cache = cache
        self.channels = channels
        self.labels = []
        self.unet = unet

        # ----- Transform definition ------------------------------------------------------

        self.preprocess = self.get_preprocess(channels, img_size)

        self.transform = self.get_transform(self.prob, intensity)
        self.advanced_transform = self.get_advanced_transform(self.prob, intensity, N, M)

        # ------- Caching & Reading -----------------------------------------------------------

        self.files = pd.read_csv(img_file).fillna(0)

        if os.environ["DEBUG"] == "True":
            self.files = self.files[0:100]
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

    @staticmethod
    def get_label(vector, label_smoothing):
        """
        This function returns the labels as a vector of probabilities. The input vectors are taken as is from
        the chexpert dataset, with 0,1,-1 corresponding to negative, positive, and uncertain, respectively.
        """

        vector = vector
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
            image = self.read_img(f"{self.img_dir}/{self.files.iloc[idx]['Path']}")
        label = self.get_label(self.files.iloc[idx, 6:19].to_numpy(), self.label_smoothing)
        # image = self.transform(image)

        # if sum(self.prob) > 0:
        #     idx = torch.randint(0, len(self), (1,)).item()
        #     image2 = self.read_img(f"{self.img_dir}/{self.files.iloc[idx]['Path']}")
        #     label2 = self.get_label(self.files.iloc[idx, 6:19].to_numpy(), self.label_smoothing)
        #     image2 = self.transform(image2)
        #
        #     samples = (image, image2, label, label2)
        #
        #     image, image2, label, label2 = self.advanced_transform(samples)
        #     del samples, image2, label2

        # image = self.preprocess(image)

        if self.unet:
            return image, image
        return image, label
