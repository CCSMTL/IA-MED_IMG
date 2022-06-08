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

#TODO : ADD PROBABILITY PER AUGMENT CATEGORY

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
        unet=False
    ):
        #----- Variable definition ------------------------------------------------------
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
        self.channels=channels
        self.labels = []
        self.unet=unet

        # ----- Transform definition ------------------------------------------------------
        if self.channels==1 :
            normalize = transforms.Normalize(mean=[0.456], std=[0.224])
        else :
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.preprocess= transforms.Compose([
            transforms.Resize(int(self.img_size*1.14)),  # 256/224 ratio
            transforms.CenterCrop(self.img_size),


            normalize
        ])


        self.transform=transforms.Compose([


            transforms.RandomErasing(p=self.prob),#TODO intensity to add

        ])
        self.advanced_transform=transforms.Compose([ #advanced/custom
            Transforms.RandAugment(prob=self.prob, intensity=self.intensity),  # p=0.5 by default
            Transforms.Mixing(self.prob, self.intensity),
            Transforms.CutMix(self.prob),
            Transforms.RandomErasing(self.prob),

        ])
        #------- Caching & Reading -----------------------------------------------------------
        a = 100 if __debug__ else len(os.listdir(img_dir + "/images"))
        num_worker=max(num_worker,1)

        self.filename= tqdm(os.listdir(img_dir + "/images")[0:a])

        if self.cache :
            self.files,self.labels = map(
                list,
                zip(
                    *Parallel(n_jobs=num_worker)(
                        delayed(self.read_img)(i) for i in self.filename
                    )
                ),
            )
        else :
            self.files = [lambda : self.read_img(i) for i in self.filename] # lambda allows to create anonymous function
                                                                            #here, i use it to "precall" "i" without actually loading the data!



    def __len__(self):
        if __debug__ :
            return 100
        return len(self.files)

    def read_img(self,file):
        totensor=transforms.PILToTensor() #TODO fix ugly
        image =  Image.fromarray(
                np.uint8(
                    cv.imread(
                        f"{self.img_dir}/images/{file}", cv.IMREAD_GRAYSCALE
                    )
                    * 255
                )
            )



        label = self.retrieve_cat(f"{self.img_dir}/labels/{file}")
        if self.channels == 3:
            image = image.convert('RGB')
        image=totensor(image)
        image=image.type(torch.uint8)
        return image, label

    def transform(self, samples):




        # samples["image"] = transforms.RandomHorizontalFlip()(
        #    samples["image"]
        # )  # default 0.5 prob
        samples["image"]=self.preprocess(samples["image"])
        samples["image2"] = self.preprocess(samples["image2"])
        samples["image"] = transforms.RandAugment(
            14, magnitude=int(10 * self.intensity)
        )(samples["image"])
        samples["image2"] = transforms.RandAugment(
            14, magnitude=int(10 * self.intensity)
        )(samples["image2"])



        return samples["image"], samples["landmarks"]

    def label_transform(self, label_ids):  # encode one_hot
        one_hot = torch.zeros((self.num_classes)) + self.label_smoothing

        for label_id in label_ids:
            assert label_id<=self.num_classes , f"Please verify your class ID's! You have ID={label_id} with #class={self.num_classes}"
            if int(label_id) < self.num_classes:  # the empty class!
                one_hot[int(label_id)] = 1 - 2 * self.label_smoothing


        return one_hot

    def retrieve_cat(self, keyname):
        category_ids = []
        label_file = f"{self.img_dir}/labels/{keyname[:-4]}.txt"
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

    def get_image(self,idx):
        if self.cache :
            image, label = self.files[idx], self.labels[idx]


        else :
            (image, label) = self.files[idx]()


        return image,label
        
    def __getitem__(self, idx):

        image,label=self.get_image(idx)

        image = self.transform(image)


        if  self.prob>0:

            idx = torch.randint(0, len(self), (1,))
            image2,label2=self.get_image(idx)
            image2 = self.transform(image2)

            samples = {
                "image": image,
                "landmarks": label,
                "image2": image2,
                "landmarks2": label2,
            }
            image,label,image2,label2 = self.advanced_transform(samples).values


        image=self.preprocess(image.float())

        if self.unet :
            return image,image
        return image, label.float()


if __name__=="__main__" :

    cxraydataloader=CxrayDataloader(img_dir="../data/test", num_classes=14, channels=3) #TODO : build test repertory with 1 or 2 test image/labels

    #testing
    x=np.uint8(np.random.random((224,224,3))*255)
    to=transforms.ToTensor()
    for i in range(5) :
        img=Image.fromarray(x)
        cxraydataloader.transform(img)
        samples = {
            "image": to(img),
            "landmarks": torch.zeros((14,)),
            "image2": to(img),
            "landmarks2": torch.zeros((14,)),
        }

        cxraydataloader.advanced_transform(samples)
        out=cxraydataloader[0]
        stop=1


