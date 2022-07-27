# ------python import------------------------------------
import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import yaml
# -------- proxy config ---------------------------
from six.moves import urllib
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import wandb
from CheXpert2.dataloaders.Chexpertloader import Chexpertloader
from CheXpert2.training.training import validation_loop
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


#
# def find_thresholds_1(tpr, fpr):
#     gmeans= []
#     thresholds= []
#     # calculate the g-mean for each threshold
#     gmeans = sqrt(tpr * (1 - fpr))
#
#     ...
#     # locate the index of the largest g-mean
#     thresholds = argmax(gmeans)
#     print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
#     return thresholds


def convert(array, thresholds):
    array = array.numpy()
    answers = []
    for item in array:
        item = np.where(item > thresholds, 1, 0)
        if np.max(item) == 0:
            answers.append(14)
        else:
            answers.append(np.argmax(item))
    return answers


def create_confusion_matrix(results):
    from CheXpert2.Metrics import Metrics  # had to reimport due to bug

    metrics = Metrics(14)
    metrics = metrics.metrics()
    for metric in metrics.keys():
        print(metric + " : ", metrics[metric](results[0].numpy(), results[1].numpy()))
    thresholds = find_thresholds(results[0], results[1])

    y_true, y_pred = convert(results[0]), convert(results[1], thresholds)
    m = (
        confusion_matrix(np.int32(y_true), np.int32(y_pred), normalize="pred") * 100
    ).round(0)

    return m


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
    num_classes = 15

    # ------loading test set --------------------------------------
    img_dir = os.environ["img_dir"]
    test_dataset = Chexpertloader(f"{img_dir}/{args.dataset}.csv", img_dir, img_size=320)

    # ----------------loading model -------------------------------
    from CheXpert2.models.CNN import CNN

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

    a = time.time()
    running_loss, results, heatmaps = validation_loop(
        model=model, loader=test_loader, criterion=criterion, device=device
    )
    print("time :", (time.time() - a) / len(test_dataset))
    plt.plot(np.sum(heatmaps[0][0].detach().cpu().numpy(), axis=0))
    plt.savefig("heatmaps.png")
    # m = create_confusion_matrix(results)
    # # -----------------------------------------------------------------------------------
    #
    # with open("data/data.yaml", "r") as stream :
    #     names = yaml.safe_load(stream)["names"]
    #
    # # np.savetxt(f"{model._get_name()}_confusion_matrix.txt",m)
    # print("avg class : ", np.mean(np.diag(m)))
    #
    # z_text = [[str(y) for y in x] for x in m]
    #
    # import plotly.figure_factory as ff
    #
    # fig = ff.create_annotated_heatmap(
    #     m, x=names, y=names, annotation_text=z_text, colorscale="Blues"
    # )
    #
    # fig.update_layout(
    #     margin=dict(t=50, l=200),
    #     # title="ResNext50 3.0",
    #     xaxis_title="True labels",
    #     yaxis_title="Predictions",
    # )
    #
    # fig["data"][0]["showscale"] = True
    # import plotly.io as pio
    #
    # pio.write_image(fig, f"{model._get_name()}_conf_mat.png", width=1920, height=1080)
    # fig.show()

    from CheXpert2.results_visualization import plot_polar_chart
    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]
    from CheXpert2.Metrics import Metrics
    metrics = Metrics(num_classes=13, names=names).metrics()
    summary = {}
    for key, value in metrics.items():
        summary[key] = value(results[0].detach().cpu().numpy(), results[1].detach().cpu().numpy())

    plot_polar_chart(summary)


if __name__ == "__main__":
    main()
