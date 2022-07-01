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

def test_sampler():
    from CheXpert2.Sampler import Sampler

    sampler = Sampler("tests/data_test")
    samples = sampler.sampler()  # probably gonna break?


if __name__ == "__main__":

    test_sampler()
