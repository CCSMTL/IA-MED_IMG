import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


from CheXpert2.dataloaders.CxrayDataloader import CxrayDataloader
from CheXpert2.custom_utils import dummy_context_mgr
from CheXpert2.models.CNN import CNN

# -------- proxy config ---------------------------
from six.moves import urllib

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



def test_unet_RGB():
    from CheXpert2.models.Unet import Unet

    x = torch.zeros((2, 3, 320, 320))
    for name in [
        "resnet18",
        "densenet121",
    ]:  # ,"inception_v3"]: #inception outputs differs
        print(name)
        unet = Unet(name, 14, channels=3)
        y = unet(x)  # test forward loop


def test_unet_grayscale():  # still in developpment
    from CheXpert2.models.Unet import Unet

    x = torch.zeros((2, 1, 320, 320))
    for name in [
        "resnet18",
        "densenet121",
    ]:  # ,"inception_v3"]: #inception outputs differs
        print(name)
        unet = Unet(name, 14, channels=1)
        y = unet(x)  # test forward loop


if __name__ == "__main__":


    test_unet_grayscale()
    test_unet_RGB()
