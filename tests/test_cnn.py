import os

import torch
# -------- proxy config ---------------------------
from six.moves import urllib

from radia.models.CNN import CNN

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

def test_cnn_grayscale():
    os.environ["DEBUG"] = "True"
    x = torch.randn((2, 2, 320, 320))
    cnn = CNN("convnext_tiny", 14, channels=1)
    y = cnn(x)  # test forward loop


def test_cnn_RGB():
    os.environ["DEBUG"] = "True"
    x = torch.randn((2, 6, 320, 320))


    cnn = CNN("convnext_tiny", 14, channels=3)
    y = cnn(x)  # test forward loop


def test_cnn_hierarchical():
    os.environ["DEBUG"] = "True"
    x = torch.randn((2, 6, 320, 320))


    cnn = CNN("convnext_tiny", 14, channels=3,hierarchical=True)
    y = cnn(x)  # test forward loop


def test_cnn_weighted_pooling():
    os.environ["DEBUG"] = "True"
    x = torch.randn((2, 6, 320, 320))
    cnn = CNN("convnext_tiny", 14, channels=3,global_pool="weighted")
    y = cnn(x)  # test forward loop


if __name__ == "__main__":

    test_cnn_grayscale()
    test_cnn_RGB()
