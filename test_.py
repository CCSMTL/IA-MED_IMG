import torch
import numpy as np
from torchvision import transforms
from PIL import Image


def test_dataloader() :
    from dataloaders.CxrayDataloader import CxrayDataloader
    GT=[[14],[14],[14],[14],[13,5,7],[14],[5],[5],[5],[13]]
    for channels in [1,3] :
        cxraydataloader = CxrayDataloader(img_dir="tests/data_test", num_classes=14, channels=channels)

        # testing inputs
        loader = torch.utils.data.DataLoader(
            cxraydataloader,
            batch_size=1,
            num_workers=0,
        )
        labels=[]
        for image,label in loader :
            label=label.numpy()
            labels.append(np.where(label==1))

        assert GT==labels , "labels are not equal to ground truth!"
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





def test_cnn() :

    from models.CNN import CNN
    for channels in [1,3] :
        x = torch.zeros((2, channels, 320, 320))
        for name in ["resnet18","densenet121","inception_v3"] :

            cnn=CNN(name,14,channels=channels)
            y = cnn(x)  # test forward loop


def test_unet() :
    from models.Unet import Unet
    for channels in [1, 3]:
        x = torch.zeros((2, channels, 320, 320))
        for name in ["resnet18", "densenet121", "inception_v3"]:
            unet = Unet(name, 14, channels=channels)
            y = unet(x)  # test forward loop

def test_sampler() :
    from Sampler import Sampler
    sampler=Sampler()
    samples=sampler.sampler()#probably gonna break?

def test_parser() :
    from parser import init_parser
    parser=init_parser()



if __name__=="__main__" :
    test_dataloader()
    test_parser()
    test_sampler()
    test_unet()
    test_cnn()