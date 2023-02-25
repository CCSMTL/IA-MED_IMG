#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on 2023-0119$

@author: Jonathan Beaulieu-Emond
"""

import copy
import os

import cv2 as cv

import numpy as np

import torch

import logging
import imageio as iio
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

from radia.dataloaders.MongoDB import MongoDB
from radia import names, hierarchy

for key in hierarchy.keys():
    if key not in names:
        names.insert(0, key)
from radia.custom_utils import (
    truncation_normalization,
    clahe,
    get_LUT_value,
    crop_coords,
)


class CXRLoader(Dataset):
    """
    This is the dataloader for our classification models. It returns the image and the corresponding class
    """

    def __init__(
        self,
        split: str = "Train",
        img_dir: str = "data",
        img_size: int = 240,
        prob: [float] = [1, 0.5, 1, 1, 1],
        label_smoothing: float = 0,
        channels: int = 1,
        use_frontal: bool = False,
        datasets: [str] = None,
        debug: bool = False,
    ) -> Dataset:
        """

        Args:
            split: Whether this will be the training, validation or test set
            img_dir: The root directory of the images
            img_size: The size of the images
            prob: A list of probabilities for each image transformation. By default, the transformations are all always applied
            label_smoothing: A float to add to the labels to smooth them. E.g for 0.05 absent : 0.05, present : 0.95
            channels: The number of channel of the images. Either 1 or 3
            use_frontal: Boolean whether to use only frontal images. Default-False
            datasets: A list of datasets to use. Default-None, which means all datasets
            debug: Whether to use the debug configuration to be able to run locally
        Returns :
            Dataset
        """

        # ----- Assertion to validate inputs ---------------------------------------------
        assert datasets is not None, "You must specify the datasets to use"
        assert 224 <= img_size <= 1000, "Image size must be between 224 and 1000"
        assert 0 <= label_smoothing <= 1, "Label smoothing must be between 0 and 1"
        assert channels in [1, 3], "Channels must be either 1 or 3"
        assert split in [
            "Train",
            "Valid",
            "Test",
        ], "Split must be either Train, Valid or Test"
        assert (
            len(prob) == 5
        ), f"Probabilities must be a list of 5 floats. Currently is {len(prob)}"
        for pro in prob:
            assert 0 <= pro <= 1, "Probabilities must be between 0 and 1"

        # ----- Variable definition ------------------------------------------------------
        self.classes = names
        self.img_dir = img_dir
        self.annotation_files = {}
        self.label_smoothing = label_smoothing
        self.prob = (
            prob
            if prob
            else [
                0,
            ]
            * 6
        )
        self.img_size = img_size
        self.channels = channels
        self.split = split

        # ----- Transform definition ------------------------------------------------------

        self.transform = self.get_transform(self.prob)

        # ------- Caching & Reading -----------------------------------------------------------
        classnames = (
            []
        )  # ["Lung Opacity", "Enlarged Cardiomediastinum"] if pretrain else []

        self.files = MongoDB(
            "10.128.107.212",
            27017,
            datasets,
            use_frontal=use_frontal,
            img_dir=img_dir,
            debug=debug,
        ).dataset(split)

        self.files[self.classes] = self.files[self.classes].astype(int)

        # -------- Group files per patient's exam ---------------------------------------------
        paths = self.files.groupby("Exam ID")["Path"].apply(list)
        frontal_lateral = self.files.groupby("Exam ID")["Frontal/Lateral"].apply(list)
        self.files = (
            self.files[self.classes + ["Exam ID"]].groupby("Exam ID").mean().round(0)
        )
        self.files["Path"] = paths
        self.files["Frontal/Lateral"] = frontal_lateral
        # ------------------------------------------------------------------------------------------
        self.read_img = lambda idx: self.read_img_from_disk(
            paths=self.files.iloc[idx]["Path"]
        )
        self.preprocess = self.get_preprocess(channels)
        self.weights = self.samples_weights()

        self.files.reset_index(inplace=True)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_transform(prob):  # for transform that would require pil images
        """
        Get the transformation to apply to the images
        Args:
            prob: The probabilities of applying each transformation

        Returns: A function that will apply the transformation with such probability

        """
        return A.Compose(
            [
                # A.augmentations.geometric.transforms.Affine(scale=(0.85, 1.15), translate_percent=(0.15, 0.15),
                #                                             rotate=(-25, 25), shear=None, cval=0, keep_ratio=True,
                #                                             p=prob[0]),
                # A.augmentations.HorizontalFlip(p=prob[1]),
                # A.augmentations.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, always_apply=False,
                #                                        p=prob[2]),
                # A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=0, value=None,
                #                  mask_value=None, always_apply=False, p=prob[3]),
                #
                # A.ElasticTransform(alpha=0.2, sigma=25, alpha_affine=50, interpolation=1, value=None, p=prob[4],
                #                    border_mode=cv.BORDER_CONSTANT),
                A.augmentations.geometric.transforms.Affine(
                    scale=(0.90, 1.10),
                    rotate=(-15, 15),
                    shear=None,
                    cval=0,
                    keep_ratio=True,
                    p=prob[0],
                ),
                # A.augmentations.transforms.GaussNoise(var_limit=(0, 0.01), mean=0, p=prob[1]),
                A.augmentations.HorizontalFlip(p=prob[1]),
                A.augmentations.transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    always_apply=False,
                    p=prob[2],
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.2,
                    interpolation=1,
                    border_mode=0,
                    value=None,
                    mask_value=None,
                    always_apply=False,
                    p=prob[3],
                ),
                A.ElasticTransform(
                    alpha=0.1,
                    sigma=20,
                    alpha_affine=40,
                    interpolation=1,
                    value=None,
                    p=prob[4],
                    border_mode=cv.BORDER_CONSTANT,
                ),
            ]
        )

    def get_label(self, idx):
        """
        This function returns the labels as a vector of probabilities. The input vectors are taken as is from
        the chexpert dataset, with 0,1,-1 corresponding to negative, positive, and uncertain, respectively.
        """

        vector, label_smoothing = (
            self.files[self.classes].iloc[idx, :].to_numpy(),
            self.label_smoothing,
        )

        # we will use the  U-Ones method to convert the vector to a probability vector
        # source : https://arxiv.org/pdf/1911.06475.pdf
        labels = np.zeros((len(vector),))
        labels[vector == 1] = 1 - label_smoothing
        labels[vector == 0] = label_smoothing

        if self.split == "Train":
            labels[vector == -1] = (
                torch.rand(size=(len(vector[vector == -1]),)) * (0.85 - 0.55) + 0.55
            )
        else:
            labels[vector == -1] = 1  # we only output binary for validation

        labels = torch.from_numpy(labels)
        labels[-1] = (
            1 - labels[-1]
        )  # lets predict the presence of a disease instead of the absence

        return labels

    def samples_weights(self):
        """
        This function returns weights 1/class_count for each image in the dataset such that each class
        is seen in similar amount
        """
        data = copy.copy(self.files).fillna(0)
        data = data.replace(-1, 0.5)
        data = data.groupby("Exam ID").mean().round(0)
        data = data[self.classes]
        data = data.astype(int)

        count = data.sum().to_numpy()
        self.count = count  # used for positive class weights
        #
        #         # WAS USED TO CALCULATE SAMPLE WEIGHTS
        #         #
        #         # for name, cat_count in zip(self.classes, count):
        #         #     if cat_count == 0:
        #         #         logging.warning(f"Careful! The category {name} has 0 images!")
        #         #
        #         # if self.split != "Train":
        #         #     return None
        #         # weights = np.zeros((len(data)))
        #         # ex = 0
        #         # for i, line in data.iterrows():
        #         #     vector = line.to_numpy()
        #         #     a = np.where(vector == 1)[0]
        #         #     if len(a) > 0:
        #         #         category = np.random.choice(a, 1)
        #     else:
        #         category = len(self.classes) - 1  # assumes last class is the empty class
        #
        #     weights[ex] = 1 / (count[category])
        #     ex += 1
        weights = torch.ones(len(self.files))
        return weights

    def read_img_from_disk(self, paths):
        """
        This function reads the images from disk associated with an exams and returns them as a numpy array
        It also applies the data augmentation to the images. For now it will load up to 2 images per exam
        Args:
            paths: A list of paths associated with the images for a specific exam

        Returns: A numpy array of shape (channel,self.img_size,self.img_size)

        """

        images = np.zeros((self.img_size, self.img_size, 2 * self.channels))
        for i, path in enumerate(np.random.permutation(paths)):
            # images[i,:,:]=cv.resize(cv.imread(f"{self.img_dir}{path}", cv.IMREAD_GRAYSCALE),(self.img_size,self.img_size))

            # img = cv.imread(f"{self.img_dir}{path}", cv.IMREAD_GRAYSCALE)
            # from PIL import Image
            # with open(f"{self.img_dir}{path}", 'rb') as f:
            #     img = np.asarray(Image.open(f))
            #
            try:
                img = iio.v3.imread(f"{self.img_dir}{path}")
                img = img.astype(np.uint8)

            except:
                logging.critical(f"Could not read image {self.img_dir}{path}")
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

            if len(img.shape) > 2:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            h, w = img.shape
            img_cropped = cv.resize(
                img[int(0.1 * h) : int(0.9 * h), int(0.1 * w) : int(0.9 * w)],
                (self.img_size, self.img_size),
            )

            if self.split.lower() == "train":
                img_cropped = self.transform(image=img_cropped)["image"]

            cl1 = clahe(img_cropped, 2.0)

            # img_normalized = truncation_normalization(img_cropped)

            # img_normalized = (img_cropped - np.min(img_cropped)) / (np.max(img_cropped) - np.min(img_cropped))
            img_normalized = img_cropped
            if self.channels == 3:

                cl2 = clahe(img_normalized, 4.0)
                img_final = np.transpose(
                    np.array([img_normalized, cl1, cl2]), (1, 2, 0)
                )

            else:
                img_final = cl1[:, :, None]

            images[:, :, i * self.channels : (i + 1) * self.channels] = img_final[
                :, :, : self.channels
            ]
            if i == 1:
                break

        return images

    @staticmethod
    def get_preprocess(channels):
        """
        Pre-processing for the model . This WILL be applied before inference

        Args:
            channels: The number of channels of the images. Either 1 or 3
        """
        if channels == 1:
            normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                normalize,
            ]
        )

    def __getitem__(self, idx):

        images = self.read_img(idx) / 255

        h, w, c = images.shape

        tensor_images = torch.zeros((c, h, w))
        for i in range(0, 2):
            image = images[:, :, i * self.channels : (i + 1) * self.channels]
            tensor_images[
                i * self.channels : (i + 1) * self.channels, :, :
            ] = self.preprocess(image)

        assert not torch.isnan(tensor_images).any()
        label = self.get_label(idx)

        return tensor_images.float(), label.float(), idx


if __name__ == "__main__":

    img_dir = os.environ["img_dir"]
    train = CXRLoader(
        split="Train",
        img_dir=img_dir,
        img_size=240,
        prob=[1, 1, 1, 1, 1],
        label_smoothing=0,
        channels=3,
        datasets=["CIUSSS"],
        debug=False,
    )
    valid = CXRLoader(
        split="Valid",
        img_dir=img_dir,
        img_size=240,
        prob=[1, 1, 1, 1, 1],
        label_smoothing=0,
        channels=1,
        datasets=["CIUSSS"],
        debug=False,
    )

    print(len(train))
    print(len(valid))
    i = 0
    import time

    start = time.time()
    for dataset in [train, valid]:
        for image, label, idx in dataset:
            i += 1
            if i == 100:
                break

    print("time : ", time.time() - start)
