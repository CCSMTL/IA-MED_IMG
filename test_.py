import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from dataloaders.CxrayDataloader import CxrayDataloader


def test_dataloader_retrieve_categories():

    cxraydataloader = CxrayDataloader(img_dir="tests/data_test", num_classes=14, channels=3)
    import os
    GT = [[14], [14], [14], [14], [13, 5, 7], [14], [5], [5], [5], [13]]
    img_dir = "tests/data_test"
    filename = os.listdir(img_dir + "/images")
    for file, true in zip(filename, GT):
        label = cxraydataloader.retrieve_cat(f"{img_dir}/labels/{file[:-4]}.txt")
        assert true == label

def test_dataloader_categories_2_vector():

    cxraydataloader = CxrayDataloader(img_dir="tests/data_test", num_classes=14, channels=1)
    import os
    GT = [[14], [14], [14], [14], [13, 5, 7], [14], [5], [5], [5], [13]]

    img_dir = "tests/data_test"
    filename = os.listdir(img_dir + "/images")
    for file, true in zip(filename, GT):
        label=cxraydataloader.label_transform(true).tolist()
        vector=np.zeros((15))
        vector[true]=1
        vector=vector[0:14].tolist()
        assert vector==label

def test_dataloader2() : #TODO : split in different test

    GT=[[14],[14],[14],[14],[13,5,7],[14],[5],[5],[5],[13]]
    vectors=[]
    for true in GT :
        vector = np.zeros((15))
        vector[true] = 1
        vector = vector[0:14].tolist()
        vectors.append(vector)
    for channels in [1,3] :
        cxraydataloader = CxrayDataloader(img_dir="tests/data_test", num_classes=14, channels=channels)

        # testing inputs



        for i,true in enumerate(vectors) :
            image,label=cxraydataloader[i]
            label=label.tolist()

            i+=1

            assert (true==label), "labels are not equal to ground truth!"

        # testing outputs
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




def test_cnn_grayscale() :

    from models.CNN import CNN

    x = torch.randn((2,1, 320, 320))
    for name in ["resnet18","densenet121"]:#, "inception_v3"]: #inception outputs differs
        print(name)
        cnn=CNN(name,14,channels=1)
        y = cnn(x)  # test forward loop


def test_cnn_RGB():
    from models.CNN import CNN

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


def test_unet_grayscale():
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

    test_dataloader_retrieve_categories()
    test_dataloader2()

    test_sampler()
    test_unet_grayscale()
    test_unet_RGB()
    test_cnn_grayscale()
    test_cnn_RGB()