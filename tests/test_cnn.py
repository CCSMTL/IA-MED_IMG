import os

import torch
# -------- proxy config ---------------------------
from six.moves import urllib

from radia.models.CNN import CNN
from radia.models.Hierarchical import Hierarchical
from radia.models.Weighted import Weighted
from radia.models.Weighted_hierarchical import Weighted_hierarchical

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


    cnn = Hierarchical("convnext_tiny", 14, channels=3)
    y = cnn(x)  # test forward loop

def test_cnn_weighted():
    os.environ["DEBUG"] = "True"
    x = torch.randn((2, 6, 320, 320))


    cnn = Weighted("convnext_tiny", 14, channels=3)
    y = cnn(x)  # test forward loop
def test_cnn_weighted_hierarchical():
    os.environ["DEBUG"] = "True"
    x = torch.randn((2, 6, 320, 320))


    cnn =Weighted_hierarchical("convnext_tiny", 14, channels=3)
    y = cnn(x)  # test forward loop



if __name__ == "__main__":

    test_cnn_grayscale()
    test_cnn_RGB()
