import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from dataloaders.CxrayDataloader import CxrayDataloader
import os
from custom_utils import dummy_context_mgr
from models.CNN import CNN

def test_dataloader_retrieve_categories():

    GT = [[14], [14], [14], [14], [13, 5, 7], [14], [5], [5], [5], [13]]
    img_dir = "tests/data_test"
    filename = os.listdir(img_dir + "/images")
    for file, true in zip(filename, GT):
        label = CxrayDataloader.retrieve_cat(f"{img_dir}/labels/{file[:-4]}.txt")
        assert true == label

def test_dataloader_categories_2_vector():

    self=dummy_context_mgr()
    self.num_classes=14
    self.label_smoothing=0
    label_transform=CxrayDataloader.label_transform

    GT = [[14], [14], [14], [14], [13, 5, 7], [14], [5], [5], [5], [13]]

    img_dir = "tests/data_test"
    filename = os.listdir(img_dir + "/images")
    for file, true in zip(filename, GT):
        label=label_transform(self,true).tolist()
        vector=np.zeros((15))
        vector[true]=1
        vector=vector[0:14].tolist()
        assert vector==label


def test_dataloader_init() :
    cxraydataloader = CxrayDataloader(img_dir="tests/data_test", num_classes=14, channels=3)

def test_dataloader_grayscale() :

    GT=[[14],[14],[14],[14],[13,5,7],[14],[5],[5],[5],[13]]
    vectors=[]
    for true in GT :
        vector = np.zeros((15))
        vector[true] = 1
        vector = vector[0:14].tolist()
        vectors.append(vector)

    cxraydataloader = CxrayDataloader(img_dir="tests/data_test", num_classes=14, channels=1)
    for i,true in enumerate(vectors) :
        image,label=cxraydataloader[i]
        assert image.shape[1]==1 , "images don't have the right number of channels!"
        label=label.tolist()
        i+=1

        assert (true==label), "labels are not equal to ground truth!"


def test_dataloader_RGB():
    GT = [[14], [14], [14], [14], [13, 5, 7], [14], [5], [5], [5], [13]]
    vectors = []
    for true in GT:
        vector = np.zeros((15))
        vector[true] = 1
        vector = vector[0:14].tolist()
        vectors.append(vector)

    cxraydataloader = CxrayDataloader(img_dir="tests/data_test", num_classes=14, channels=3)
    for i, true in enumerate(vectors):
        image, label = cxraydataloader[i]
        assert image.shape[1] == 3, "images don't have the right number of channels!"
        label = label.tolist()
        i += 1

        assert (true == label), "labels are not equal to ground truth!"


def test_dataloader_transform() :

        transform=CxrayDataloader.get_transform(0.2)
        # testing outputs
        x = torch.randint(0,255,(224, 224, 3),dtype=torch.uint8)
        to = transforms.ToTensor()
        for i in range(5):

            img2=transform(x)

            assert x.shape == img2.shape


def test_dataloader_advanced_transform():
    # testing outputs
    x = np.uint8(np.random.random((224, 224, 3)) * 255)
    to = transforms.ToTensor()
    transform=CxrayDataloader.get_advanced_transform(0.2,0.1)
    for i in range(5):
        img = to(Image.fromarray(x))

        samples = {
            "image": img,
            "landmarks": torch.zeros((14,)),
            "image2": img,
            "landmarks2": torch.zeros((14,)),
        }

        img2=transform(samples)
        assert img2["image"].shape==img.shape , "images are not the same shape!"
        assert len(img2["landmarks"])==14


def test_cnn_grayscale() :



    x = torch.randn((2,1, 320, 320))
    for name in ["resnet18","densenet121"]:#, "inception_v3"]: #inception outputs differs
        print(name)
        cnn=CNN(name,14,channels=1)
        y = cnn(x)  # test forward loop


def test_cnn_RGB():


    x = torch.randn((2, 3, 320, 320))
    for name in ["resnet18", "densenet121"]:  # , "inception_v3"]: #inception outputs differs
        print(name)
        cnn = CNN(name, 14, channels=3)
        y = cnn(x)  # test forward loop


def test_unet_RGB() :
    from models.Unet import Unet

    x = torch.zeros((2, 3, 320, 320))
    for name in ["resnet18", "densenet121"] : #,"inception_v3"]: #inception outputs differs
        print(name)
        unet = Unet(name, 14, channels=3)
        y = unet(x)  # test forward loop


def test_unet_grayscale(): # still in developpment
    from models.Unet import Unet

    x = torch.zeros((2, 1, 320, 320))
    for name in ["resnet18", "densenet121"]:  # ,"inception_v3"]: #inception outputs differs
        print(name)
        unet = Unet(name, 14, channels=1)
        y = unet(x)  # test forward loop


def test_sampler() :
    from Sampler import Sampler
    sampler=Sampler()
    samples=sampler.sampler()#probably gonna break?

if __name__=="__main__" :
    test_dataloader_init()
    test_dataloader_retrieve_categories()
    test_dataloader_RGB()
    test_dataloader_grayscale()
    test_sampler()
    test_unet_grayscale()
    test_unet_RGB()
    test_cnn_grayscale()
    test_cnn_RGB()