import os

import torch
import numpy as np
from radia.dataloaders.CXRLoader import CXRLoader
from radia import names

# -------- proxy config ---------------------------

# proxy = urllib.request.ProxyHandler(
#     {
#         "https": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
#         "http": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
#     }
# )
# os.environ["HTTPS_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
# os.environ["HTTP_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
# # construct a new opener using your proxy settings
# opener = urllib.request.build_opener(proxy)
# # install the openen on the module-level
# urllib.request.install_opener(opener)

try :
    img_dir = os.environ["img_dir"]
except :
    img_dir = ""

def test_dataloader_get_item_1channel():

    train = CXRLoader(
            split="Train",
            img_dir = img_dir,
            img_size=224,
            datasets=["ChexPert"],
            debug=True,
            channels=1
            )
    print(len(train))
    images, label,idx = train[4]
    frontal = images[0,:,:]
    assert frontal.shape == (224,224)
    assert label.shape == (len(names),)

def test_dataloader_get_item_3channel():

    train = CXRLoader(
            split="Train",
            img_dir = img_dir,
            img_size=224,
            datasets=["ChexPert"],
            debug=True,
            channels=3
            )
    print(len(train))
    images, label,idx = train[4]
    frontal = images[0,:,:]
    assert frontal.shape == (224,224)
    assert label.shape == (len(names),)


def test_dataloader_transform():

    transform = CXRLoader.get_transform([0.2, ] * 6)
    # testing outputs
    x = np.random.randint(0, 255, (224,224,1), dtype=np.uint8)

    for i in range(5):
        img2 = transform(image=x)["image"]

        assert x.shape == img2.shape

def test_dataloader_sampler():

    train = CXRLoader("Train",datasets=["ChexPert"],debug=True,img_dir=img_dir)
    assert len(train.weights) == len(train)


if __name__ == "__main__":
    test_dataloader_transform()

