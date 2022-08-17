#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30$

@author: Jonathan Beaulieu-Emond
"""

import cv2 as cv
import numpy as np
import pandas as pd
import pymongo

import torch
import tqdm
from joblib import Parallel, delayed, parallel_backend
from torch.utils.data import Dataset
from torchvision import transforms

from CheXpert2 import custom_Transforms


def GetDocPerClassAndCollection(ClassName, collectionName, query):
    listResult = []
    queryRequest = {}
    query1 = {ClassName: "1"}

    if (query == 'Train'):
        query2 = {"Train": {'$in': ["1", "-1"]}}
    else:
        query2 = {query: '1'}

    # Merge the query dictionaries
    queryRequest = dictMerge(query1, query2)

    for collect in CollectionList:
        print(collect.name)

        if collect.name == collectionName:
            # The find() returns a cursor on the device but not a regular list
            res = collect.find(queryRequest)

            # Collect data and store the records into a list
            for x in res:
                listResult.append(x)
                print(x)

    return listResult


class CXRLoader(Dataset):
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
            N=0,
            M=0,
            pretrain=True
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

        if self.cache:
            with parallel_backend('threading', n_jobs=num_worker):
                self.images = Parallel()(
                    delayed(self.read_img)(f"{self.img_dir}/{self.files.iloc[idx]['Path']}") for idx in
                    tqdm.tqdm(range(0, len(self.files))))

        self.pretrain = pretrain
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

        convert = {
            "Male": 0,
            "Female": 1,
            "Frontal": 0,
            "Lateral": 1,
            "AP": 1,
            "PA": 0,
            0: 0.5,
            "LL": 0.5,
            "Unknown": 0.5,
            "RL": 0.5,

        }
        if not self.pretrain:
            vector, label_smoothing = self.files.iloc[idx, 5:19].to_numpy(), self.label_smoothing

            # we will use the  U-Ones method to convert the vector to a probability vector TODO : explore other method
            # source : https://arxiv.org/pdf/1911.06475.pdf
            labels = np.zeros((len(vector),))
            labels[vector == 1] = 1 - label_smoothing
            labels[vector == 0] = label_smoothing
            labels[vector == -1] = torch.rand(size=(len(vector[vector == -1]),)) * (0.85 - 0.55) + 0.55

        else:
            data = self.files.iloc[idx, 1:5].iloc
            labels = np.array([convert[data[0]], int(data[1]) / 100, convert[data[2]], convert[data[3]]])
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
        label = self.get_label(idx)
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
        return image, label.float()
