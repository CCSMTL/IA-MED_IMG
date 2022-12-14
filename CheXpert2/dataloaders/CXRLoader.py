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
import imageio as iio
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

from CheXpert2.dataloaders.MongoDB import MongoDB
from CheXpert2 import names



def get_LUT_value(data, window, level):
    """Apply the RGB Look-Up Table for the given
       data and window/level value."""

    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) /
                                               (window - 1) + 0.5) * (255 - 0)])


def crop_coords(img):
    """
    Crop ROI from image.
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    cnts, _ = cv.findContours(breast_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)
    return (x, y, w, h)


def truncation_normalization(img):
    """
    Clip and normalize pixels in the breast ROI.
    @img : numpy array image
    return: numpy array of the normalized image
    """
    Pmin = np.percentile(img[img != 0], 5)
    Pmax = np.percentile(img[img != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[img == 0] = 0
    return normalized


def clahe(img, clip):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img * 255, dtype=np.uint8))
    return cl


class CXRLoader(Dataset):
    """
    This is the dataloader for our classification models. It returns the image and the corresponding class
    """

    def __init__(
            self,
            split="Train",
            img_dir="data",
            img_size=240,
            prob=None,
            intensity=0,
            label_smoothing=0,
            channels=1,
            use_frontal=False,
            datasets=None,
            debug=False
    ):

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

        # ------- Caching & Reading -----------------------------------------------------------
        classnames = []  # ["Lung Opacity", "Enlarged Cardiomediastinum"] if pretrain else []

        self.files = MongoDB("10.128.107.212", 27017, datasets, use_frontal=use_frontal,img_dir=img_dir,debug=debug).dataset(split,
                                                                                                 classnames=classnames)

        self.files[self.classes] = self.files[self.classes].astype(int)

        paths = self.files.groupby("Exam ID")["Path"].apply(list)
        frontal_lateral = self.files.groupby("Exam ID")["Frontal/Lateral"].apply(list)
        self.files = self.files[self.classes + ["Exam ID"]].groupby("Exam ID").mean().round(0)
        self.files["Path"] = paths
        self.files["Frontal/Lateral"] = frontal_lateral

        self.read_img = lambda idx: self.read_img_from_disk(paths=self.files.iloc[idx]['Path'],
                                                            views=self.files.iloc[idx]['Frontal/Lateral'])
        self.preprocess = self.get_preprocess(channels)
        self.weights = self.samples_weights()

        self.files.reset_index(inplace=True)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_transform(prob):  # for transform that would require pil images
        return A.Compose(
            [

                A.augmentations.geometric.transforms.Affine(scale=(0.85, 1.15), translate_percent=(0.15, 0.15),
                                                            rotate=(-25, 25), shear=None, cval=0, keep_ratio=True,
                                                            p=prob[1]),

                A.augmentations.HorizontalFlip(p=prob[2]),
                A.augmentations.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, always_apply=False,
                                                       p=prob[3]),
                A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=0, value=None,
                                 mask_value=None, always_apply=False, p=prob[4]),

                A.ElasticTransform(alpha=0.2, sigma=25, alpha_affine=50, interpolation=1, value=None, p=prob[5],
                                   border_mode=cv.BORDER_CONSTANT),

                # A.augmentations.transforms.RandomGamma()
                # A.augmentations.PixelDropout(dropout_prob=0.05,p=0.5),
                # gaussian blur?
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

        if self.split == "Train":
            labels[vector == -1] = torch.rand(size=(len(vector[vector == -1]),)) * (0.85 - 0.55) + 0.55
        else:
            labels[vector == -1] = 1  # we only output binary for validation #TODO : verify that

        labels = torch.from_numpy(labels)
        labels[-1] = 1 - labels[-1]  # lets predict the presence of a disease instead of the absence

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
        self.count = count
        for name, cat_count in zip(self.classes, count):
            if cat_count == 0:
                logging.warning(f"Careful! The category {name} has 0 images!")

        if self.split != "Train":
            return None
        weights = np.zeros((len(data)))
        ex = 0
        for i, line in data.iterrows():
            vector = line.to_numpy()
            a = np.where(vector == 1)[0]
            if len(a) > 0:
                category = np.random.choice(a, 1)
            else:
                category = len(self.classes) - 1  # assumes last class is the empty class

            weights[ex] = 1 / (count[category])
            ex += 1

        return weights

    def step(self, idxs, pseudo_labels):  # moving avg ; not yet fully implemented

        labels = self.files.loc[idxs, self.classes].to_numpy()
        new_labels = 0.999 * labels + 0.001 * pseudo_labels
        self.files.loc[idxs, self.classes] = new_labels

    def read_img_from_disk(self, paths, views):

        images = torch.zeros((2*self.channels,self.img_size, self.img_size))
        for i, path in enumerate(np.random.permutation(paths)):
            # images[i,:,:]=cv.resize(cv.imread(f"{self.img_dir}{path}", cv.IMREAD_GRAYSCALE),(self.img_size,self.img_size))

            # img = cv.imread(f"{self.img_dir}{path}", cv.IMREAD_GRAYSCALE)
            # from PIL import Image
            # with open(f"{self.img_dir}{path}", 'rb') as f:
            #     img = np.asarray(Image.open(f))
            #
            img = cv.resize(iio.v3.imread(f"{self.img_dir}{path}"), (int(self.img_size * 1.2), int(self.img_size * 1.2)))

            # h, w = image.shape

            # crop_img = image[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)]
            # images[i, :, :] = cv.resize(crop_img, (self.img_size, self.img_size))
            if len(img.shape) > 2:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            (x, y, w, h) = crop_coords(img)
            img_cropped = img[y:y + h, x:x + w]
            img_normalized = truncation_normalization(img_cropped)


            if self.channels==3 :
                cl1 = clahe(img_normalized, 1.0)
                cl2 = clahe(img_normalized, 2.0)
                img_final = cv.merge((np.array(img_normalized * 255, dtype=np.uint8), cl1, cl2))
                img_final = cv.resize(img_final, (int(self.img_size), int(self.img_size)))
                if self.split.lower() == "train" :
                    img_final = self.transform(image=img_final)["image"]
                images[i * 3:(i + 1) * 3,:,:] = self.preprocess(img_final)
            else :
                img_final = cv.resize(np.array(img_normalized * 255, dtype=np.uint8), (int(self.img_size), int(self.img_size)))
                if self.split.lower() == "train":
                    img_final = self.transform(image=img_final)["image"]
                images[i,:, :] = self.preprocess(img_final)


            if i == 1:
                break

        return images

    @staticmethod
    def get_preprocess(channels):
        """
        Pre-processing for the model . This WILL be applied before inference
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

        images = self.read_img(idx)
        assert not torch.isnan(images).any()
        label = self.get_label(idx)

        return images, label.float(), idx


if __name__ == "__main__":

    img_dir = os.environ["img_dir"]
    train = CXRLoader(split="Train", img_dir=img_dir, img_size=240, prob=None, intensity=0, label_smoothing=0,
                      channels=3, datasets=["ChexPert"],debug=True)
    valid = CXRLoader(split="Valid", img_dir=img_dir, img_size=240, prob=None, intensity=0, label_smoothing=0,
                      channels=3, datasets=["ChexPert"],debug=True)
    print(len(train))
    print(len(valid))
    i = 0
    for dataset in [train, valid]:
        for image, label, idx in dataset:
            i += 1
            if i == 100:
                break
