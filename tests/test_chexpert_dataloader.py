import os

import numpy as np
import torch

from CheXpert2.dataloaders.Chexpertloader import Chexpertloader


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



def test_dataloader_get_label():
    os.environ["DEBUG"] = "True"
    vectors = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, -1, -1])
    labels = Chexpertloader.get_label(vectors, label_smoothing=0.05).tolist()

    assert labels[0:13] == [0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.05, 0.05, 0.95, 0.05, 0.05, 0.05, 0.05]
    assert 0 < labels[13] < 1
    assert 0 < labels[14] < 1


def test_dataloader_get_item():
    os.environ["DEBUG"] = "True"
    img_file = os.path.join(os.getcwd(), "tests/data_test/valid.csv")
    cxraydataloader = Chexpertloader(
        img_file=img_file, channels=3, img_size=224, img_dir="tests/data_test",
    )
    image, label = cxraydataloader[4]
    assert image.shape == (3, int(224 * 1.14), int(224 * 1.14))
    assert label.shape == (13,)


def test_dataloader_transform():
    os.environ["DEBUG"] = "True"
    transform = Chexpertloader.get_transform([0.2, ] * 5, 0.1)
    # testing outputs
    x = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)

    for i in range(5):
        img2 = transform(x)

        assert x.shape == img2.shape


def test_dataloader_advanced_transform():
    # testing outputs
    os.environ["DEBUG"] = "True"
    img = torch.randint(0, 255, (16, 3, 224, 224), dtype=torch.uint8).to(memory_format=torch.channels_last)
    transform = Chexpertloader.get_advanced_transform([0.2, ] * 5, 0.1, 2, 9)
    label = torch.randint(0, 2, (16, 13), dtype=torch.float32)
    for i in range(5):
        img2, label2 = transform((img, label))

        assert img2.shape == img.shape, "images are not the same shape!"
        assert label2.shape[1] == 13


if __name__ == "__main__":
    test_dataloader_transform()
    test_dataloader_advanced_transform()
