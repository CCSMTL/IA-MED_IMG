import os

import numpy as np
import torch
from PIL import Image
# -------- proxy config ---------------------------
from six.moves import urllib
from torchvision import transforms

from CheXpert2.custom_utils import dummy_context_mgr
from CheXpert2.dataloaders.CxrayDataloader import CxrayDataloader
from CheXpert2.models.CNN import CNN

proxy = urllib.request.ProxyHandler(
    {
        "https": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
        "http": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
    }
)
os.environ["HTTPS_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
os.environ["HTTP_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
# construct a new opener using your proxy settings
opener = urllib.request.build_opener(proxy)
# install the openen on the module-level
urllib.request.install_opener(opener)

os.environ["DEBUG"] = "False"


def test_cnn_grayscale():
    x = torch.randn((2, 1, 320, 320))
    for name in [
        "resnet18",
        "densenet121",
    ]:  # , "inception_v3"]: #inception outputs differs
        print(name)
        cnn = CNN(name, 14, channels=1)
        y = cnn(x)  # test forward loop


def test_cnn_RGB():

    x = torch.randn((2, 3, 320, 320))
    for name in [
        "resnet18",
        "densenet121",
    ]:  # , "inception_v3"]: #inception outputs differs
        print(name)
        cnn = CNN(name, 14, channels=3)
        y = cnn(x)  # test forward loop




if __name__ == "__main__":
    test_dataloader_init()
    test_dataloader_retrieve_categories()
    test_dataloader_RGB()
    test_dataloader_grayscale()
    test_sampler()
    test_unet_grayscale()
    test_unet_RGB()
    test_cnn_grayscale()
    test_cnn_RGB()
