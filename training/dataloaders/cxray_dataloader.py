import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
import cv2 as cv
from PIL import Image
import Transforms
from tqdm.auto import tqdm
from joblib import Parallel, delayed


class CustomImageDataset(Dataset):
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
    ):

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

        self.labels = []

        def caching(file):
            if self.cache:
                file = transforms.Resize(self.img_size)(
                    Image.fromarray(
                        np.uint8(
                            cv.imread(
                                f"{self.img_dir}/images/{file}", cv.IMREAD_GRAYSCALE
                            )
                            * 255
                        )
                    )
                )

                label = self.retrieve_cat(f"{self.img_dir}/labels/{file}")
            else:
                file = f"{self.img_dir}/images/{file}"
                label = self.retrieve_cat(f"{self.img_dir}/labels/{file}")

            return file, label
        a = 100 if os.environ["DEBUG"]=="True" else len(os.listdir(img_dir + "/images"))
        self.files, self.labels = map(
            list,
            zip(
                *Parallel(n_jobs=8)(
                    delayed(caching)(i) for i in tqdm(os.listdir(img_dir + "/images")[0:a])
                )
            ),
        )

    def __len__(self):
        if os.environ["DEBUG"] == "True":
            return 100
        return len(self.files)

    def transform(self, samples):

        samples["image"] = transforms.Resize(self.img_size)(samples["image"])
        samples["image2"] = transforms.Resize(self.img_size)(samples["image2"])
        # transforms.CenterCrop(self.img_size)(image) #redundant?

        # samples["image"] = transforms.RandomHorizontalFlip()(
        #    samples["image"]
        # )  # default 0.5 prob
        samples["image"] = transforms.RandAugment(
            14, magnitude=int(10 * self.intensity)
        )(samples["image"])
        samples["image2"] = transforms.RandAugment(
            14, magnitude=int(10 * self.intensity)
        )(samples["image2"])
        samples["image"] = transforms.ToTensor()(samples["image"])
        samples["image2"] = transforms.ToTensor()(samples["image2"])
        # samples = Transforms.Mixing(self.prob, self.intensity)(samples)
        # samples = Transforms.CutMix(self.prob)(samples)
        # samples = Transforms.RandomErasing(self.prob)(samples)

        samples["image"] = transforms.Normalize(mean=[0.456], std=[0.224])(
            samples["image"]
        )

        return samples["image"], samples["landmarks"]

    def label_transform(self, label_ids):  # encode one_hot
        one_hot = torch.zeros((self.num_classes)) + self.label_smoothing
        for label_id in label_ids:
            if int(label_id) == self.num_classes:  # the empty class!
                pass  # we do nothing
            else:
                one_hot[int(label_id)] = 1 - 2 * self.label_smoothing
        return one_hot

    def retrieve_cat(self, keyname):
        category_ids = []
        label_file = f"{self.img_dir}/labels/{keyname[:-3]}txt"
        if os.path.exists(label_file):
            with open(label_file) as f:
                lines = f.readlines()
                if len(lines) > 0:  # if file no empty
                    for line in lines:
                        line = line.split(" ")
                        category_ids.append(line[0])
                else:
                    category_ids.append(
                        14
                    )  # if the txt file is missing, we presume empty image
        else:
            category_ids.append(14)  # the image is empty

        return self.label_transform(category_ids)

    def __getitem__(self, idx):

        if self.cache:
            image = self.files[idx]
            label = self.labels[idx]
        else:
            img_path = self.files[idx]
            patterns = img_path.split("/")[::-1]
            keyname = patterns[0]
            label = self.retrieve_cat(keyname)
            image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            image = Image.fromarray(np.uint8(image * 255))

        if torch.rand((1,)) < self.prob:
            if self.cache:
                idx = torch.randint(0, len(self), (1,))
                image2 = self.files[idx]
                random_label = self.labels[idx]
            else:
                random_image = self.files[torch.randint(0, len(self), (1,))]
                random_label = self.retrieve_cat(random_image.split("/")[::-1][0])
                image2 = Image.fromarray(
                    np.uint8(cv.imread(random_image * 255, cv.IMREAD_GRAYSCALE))
                )

            sample = {
                "image": image,
                "landmarks": label,
                "image2": image2,
                "landmarks2": random_label,
            }

            image, label = self.transform(sample)
        else:  # basic tranformation

            image = transforms.Resize(self.img_size)(image)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.456], std=[0.224])(image)

        return image.float(), label.float()
