# ------python import------------------------------------
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
# -------- proxy config ---------------------------
from six.moves import urllib
from sklearn import metrics

import wandb
from CheXpert2.dataloaders.Chexpertloader import Chexpertloader
from CheXpert2.models.CNN import CNN
from polycam.polycam.polycam import PCAMp

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

os.environ["DEBUG"] = "True"


@torch.no_grad()
def infer_loop(model, loader, criterion, device):
    """

    :param model: model to evaluate
    :param loader: dataset loader
    :param criterion: criterion to evaluate the loss
    :param device: device to do the computation on
    :return: val_loss for the N epoch, tensor of concatenated labels and predictions
    """
    running_loss = 0
    results = [torch.tensor([]), torch.tensor([])]

    for inputs, labels in loader:
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = (
            inputs.to(device, non_blocking=True, memory_format=torch.channels_last),
            labels.to(device, non_blocking=True),
        )
        inputs = loader.dataset.preprocess(inputs)
        # forward + backward + optimize

        outputs, heatmaps = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.detach()

        if inputs.shape != labels.shape:  # prevent storing images if training unets
            results[1] = torch.cat(
                (results[1], torch.sigmoid(outputs).detach().cpu()), dim=0
            )
            results[0] = torch.cat((results[0], labels.cpu()), dim=0)

        del (
            inputs,
            labels,
            outputs,
            loss,
        )  # garbage management sometimes fails with cuda
        break

    return running_loss, results, heatmaps


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch testing for a specific model")

    parser.add_argument(
        "--dataset",
        default="valid",
        type=str,
        choices=["training", "validation"],
        required=False,
        help="Choice of the test set",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Choice of the model",
    )

    return parser


def find_thresholds(true, pred):

    true, pred = true.numpy(), pred.numpy()

    def f1(thresholds, true, pred):
        for ex, item in enumerate(pred):
            item = np.where(item > thresholds, 1, 0)
            pred[ex] = item

        return -metrics.f1_score(true, pred, average="macro", zero_division=0)

    thresholds = scipy.optimize.minimize(f1, args=(true, pred))
    return thresholds


def main():
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    # ----- parsing arguments --------------------------------------
    parser = init_argparse()
    args = parser.parse_args()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # ------loading test set --------------------------------------
    img_dir = os.environ["img_dir"]
    test_dataset = Chexpertloader(f"{img_dir}/{args.dataset}.csv", img_dir, img_size=320)

    # ----------------loading model -------------------------------

    model = CNN("convnext_base", 13)
    # model =  torch.nn.parallel.DistributedDataParallel(model)

    api = wandb.Api()

    # run = api.run(f"ccsmtl2/Chestxray/{args.run_id}")
    # run.file("models_weights/convnext_base/DistributedDataParallel.pt").download(replace=True)
    state_dict = torch.load("models_weights/convnext_base/DistributedDataParallel.pt")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()
    model = PCAMp(model)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    start.record()
    running_loss, results, heatmaps = infer_loop(
        model=model, loader=test_loader, criterion=criterion, device=device
    )
    end.record()
    torch.cuda.synchronize()
    print("time : ", start.elapsed_time(end))
    plt.imshow(np.sum(heatmaps[0][0].detach().cpu().numpy(), axis=0))
    plt.savefig("heatmaps.png")

if __name__ == "__main__":
    main()
