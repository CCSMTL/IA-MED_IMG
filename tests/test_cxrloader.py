import os

import torch
import numpy as np
from CheXpert2.dataloaders.CXRLoader import CXRLoader
from CheXpert2 import names

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

def test_dataloader_get_item():
    os.environ["DEBUG"] = "True"
    train = CXRLoader(
            split="Train",
            img_dir = img_dir,
            img_size=224,
            datasets=["ChexPert"])
    frontal,lateral, label,idx = train[4]
    assert frontal.shape == (1, int(224), int(224))
    assert label.shape == (len(names),)


def test_dataloader_transform():
    os.environ["DEBUG"] = "True"
    transform = CXRLoader.get_transform([0.2, ] * 6, 0.1)
    # testing outputs
    x = np.random.randint(0, 255, (224,224,1), dtype=np.uint8)

    for i in range(5):
        img2 = transform(image=x)["image"]

        assert x.shape == img2.shape


def test_dataloader_advanced_transform():
    # testing outputs
    os.environ["DEBUG"] = "True"
    img = torch.randint(0, 255, (16, 3, 224, 224), dtype=torch.uint8)
    transform = CXRLoader.get_advanced_transform([0.2, ] * 5, 0.1, 2, 9)
    label = torch.randint(0, 2, (16, 14), dtype=torch.float32)
    for i in range(5):
        img2, label2 = transform((img, label))

        assert img2.shape == img.shape, "images are not the same shape!"
        assert label2.shape[1] == 14


def test_dataloader_sampler():
    os.environ["DEBUG"] = "False"
    train = CXRLoader("Train",datasets=["ChexPert"])
    assert len(train.weights) == len(train)


if __name__ == "__main__":
    test_dataloader_transform()
    test_dataloader_advanced_transform()
