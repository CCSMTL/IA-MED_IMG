import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
import cv2 as cv
from PIL import Image
from CheXpert2 import Transforms
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import warnings


# TODO : ADD PROBABILITY PER AUGMENT CATEGORY


class CxrayDataloader(Dataset):
    """
    This is the dataloader for our classification models. It returns the image and the corresponding class
    """

    def __init__(
        self,
        img_dir,
        num_classes,
        img_size=240,
        prob=0,
        intensity=0,
        label_smoothing=0,
        cache=False,
        num_worker=0,
        channels=3,
        unet=False,
    ):
        # ----- Variable definition ------------------------------------------------------
        self.img_dir = img_dir

        self.length = 0
        self.files = []
        self.annotation_files = {}
        self.num_classes = num_classes
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
        self.advanced_transform = self.get_advanced_transform(prob, intensity)

        # ------- Caching & Reading -----------------------------------------------------------

        a = 100 if __debug__ else len(os.listdir(img_dir + "/images"))
        num_worker = max(num_worker, 1)

        self.filename = sorted(os.listdir(img_dir + "/images")[0:a])

        if self.cache:
            self.files, self.labels = map(
                list,
                zip(
                    *Parallel(n_jobs=num_worker)(
                        delayed(self.read_img)(i) for i in tqdm(self.filename)
                    )
                ),
            )

    def __len__(self):
        return len(self.filename)

    @staticmethod
    def get_transform(prob):
        return transforms.Compose(
            [
                transforms.RandomErasing(p=prob),  # TODO intensity to add
            ]
        )

    @staticmethod
    def get_advanced_transform(prob, intensity):
        return transforms.Compose(
            [  # advanced/custom
                Transforms.RandAugment(
                    prob=prob, intensity=intensity
                ),  # p=0.5 by default
                Transforms.Mixing(prob, intensity),
                Transforms.CutMix(prob),
                Transforms.RandomErasing(prob),
            ]
        )

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

        image = torch.tensor(
            cv.imread(f"{self.img_dir}/images/{file}", cv.IMREAD_GRAYSCALE) * 255,
            dtype=torch.uint8,
        )[None, :, :]

        label = self.label_transform(
            self.retrieve_cat(f"{self.img_dir}/labels/{file[:-4]}.txt")
        ).float()
        if self.channels == 3:
            image = image.repeat((3, 1, 1))

        return image, label

    def label_transform(self, label_ids):  # encode one_hot
        one_hot = torch.zeros((self.num_classes)) + self.label_smoothing

        for label_id in label_ids:

            assert (
                label_id <= self.num_classes
            ), f"Please verify your class ID's! You have ID={label_id} with #class={self.num_classes}"
            if label_id < self.num_classes:  # the empty class!
                one_hot[label_id] = 1 - self.label_smoothing

        return one_hot

    @staticmethod
    def retrieve_cat(label_file):
        category_ids = []

        if os.path.exists(label_file):
            with open(label_file) as f:
                lines = f.readlines()

                if len(lines) > 0:  # if file no empty
                    for line in lines:
                        line = line.split(" ")
                        category_ids.append(int(line[0]))

                else:
                    raise Exception("File empty!!")
        else:
            raise Exception("File not found!")

        return category_ids

    def get_image(self, idx):
        if self.cache:
            image, label = self.files[idx], self.labels[idx]

        else:
            image, label = self.read_img(self.filename[idx])

        return image, label

    def __getitem__(self, idx):

        image, label = self.get_image(idx)

        image = self.transform(image)

        if self.prob > 0:
            idx = torch.randint(0, len(self), (1,))
            image2, label2 = self.get_image(idx)
            image2 = self.transform(image2)

            samples = {
                "image": image,
                "landmarks": label,
                "image2": image2,
                "landmarks2": label2,
            }
            image, label, image2, label2 = (self.advanced_transform(samples)).values()

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
