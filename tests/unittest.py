import unittest
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


class Test(unittest.TestCase) :


    def dataloader_test(self):
        from dataloaders import CxrayDataloader

        for channels in [1, 3]:
            cxraydataloader = CxrayDataloader(img_dir="/data_test", num_classes=14, channels=channels)

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

    def cnn_test(self):

        from models.CNN import CNN
        for channels in [1, 3]:
            x = torch.zeros((2, channels, 320, 320))
            for name in ["resnet18", "densenet121", "inception_v3"]:
                cnn = CNN(name, 14, channels=channels)
                y = cnn(x)  # test forward loop

    def unet_test(self):
        from models.Unet import Unet
        for channels in [1, 3]:
            x = torch.zeros((2, channels, 320, 320))
            for name in ["resnet18", "densenet121", "inception_v3"]:
                unet = Unet(name, 14, channels=channels)
                y = unet(x)  # test forward loop

    def sampler_test(self):
        from Sampler import Sampler
        sampler = Sampler()
        samples = sampler.sampler()  # probably gonna break?

    def parser_test(self):
        from parser import init_parser
        parser = init_parser()


if __name__=="__main__" :
    unittest.main()