import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


from CheXpert2.dataloaders.CxrayDataloader import CxrayDataloader
from CheXpert2.custom_utils import dummy_context_mgr


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


def test_dataloader_retrieve_categories():
    img_dir = os.path.join(os.getcwd(),"data_test")

    GT = [[14], [14], [14], [14], [13, 5, 7], [14], [5], [5], [5], [13]]

    filename = sorted(os.listdir(img_dir + "/images"))

    answers=[]
    for file, true in zip(filename, GT):
        label = CxrayDataloader.retrieve_cat(f"{img_dir}/labels/{file[:-4]}.txt")
        answers.append(label)

    assert GT == answers

def test_dataloader_categories_2_vector():
    img_dir = os.path.join(os.getcwd(),"data_test")

    self = dummy_context_mgr()
    self.num_classes = 14
    self.label_smoothing = 0
    label_transform = CxrayDataloader.label_transform

    GT = [[14], [14], [14], [14], [13, 5, 7], [14], [5], [5], [5], [13]]


    filename = os.listdir(img_dir + "/images")
    for file, true in zip(filename, GT):
        label = label_transform(self, true).tolist()
        vector = np.zeros((15))
        vector[true] = 1
        vector = vector[0:14].tolist()
        assert vector == label


def test_dataloader_init():
    img_dir = os.path.join(os.getcwd(),"data_test")
    cxraydataloader = CxrayDataloader(
        img_dir=img_dir, num_classes=14, channels=3
    )


def test_dataloader_grayscale():
    img_dir = os.path.join(os.getcwd(), "data_test")
    GT = [[14], [14], [14], [14], [13, 5, 7], [14], [5], [5], [5], [13]]
    vectors = []
    for true in GT:
        vector = np.zeros((15))
        vector[true] = 1
        vector = vector[0:14].tolist()
        vectors.append(vector)

    cxraydataloader = CxrayDataloader(
        img_dir=img_dir, num_classes=14, channels=1
    )
    for i, true in enumerate(vectors):
        image, label = cxraydataloader[i]
        assert image.shape[0] == 1, "images don't have the right number of channels!"
        label = label.tolist()
        i += 1

        assert true == label, "labels are not equal to ground truth!"


def test_dataloader_RGB():
    img_dir = os.path.join(os.getcwd(), "data_test")
    GT = [[14], [14], [14], [14], [13, 5, 7], [14], [5], [5], [5], [13]]
    vectors = []
    for true in GT:
        vector = np.zeros((15))
        vector[true] = 1
        vector = vector[0:14].tolist()
        vectors.append(vector)

    cxraydataloader = CxrayDataloader(
        img_dir=img_dir, num_classes=14, channels=3
    )
    for i, true in enumerate(vectors):
        image, label = cxraydataloader[i]
        assert image.shape[0] == 3, "images don't have the right number of channels!"
        label = label.tolist()
        i += 1

        assert true == label, "labels are not equal to ground truth!"


def test_dataloader_transform():

    transform = CxrayDataloader.get_transform(0.2)
    # testing outputs
    x = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
    to = transforms.ToTensor()
    for i in range(5):

        img2 = transform(x)

        assert x.shape == img2.shape


def test_dataloader_advanced_transform():
    # testing outputs
    x = np.uint8(np.random.random((224, 224, 3)) * 255)
    to = transforms.ToTensor()
    transform = CxrayDataloader.get_advanced_transform(0.2, 0.1)
    for i in range(5):
        img = to(Image.fromarray(x))

        samples = {
            "image": img,
            "landmarks": torch.zeros((14,)),
            "image2": img,
            "landmarks2": torch.zeros((14,)),
        }

        img2 = transform(samples)
        assert img2["image"].shape == img.shape, "images are not the same shape!"
        assert len(img2["landmarks"]) == 14



if __name__ == "__main__":
    test_dataloader_init()
    test_dataloader_retrieve_categories()
    test_dataloader_RGB()
    test_dataloader_grayscale()
    test_dataloader_transform()
    test_dataloader_advanced_transform()
