# ------python import------------------------------------
import argparse
import time
import warnings

import numpy as np
import torch
import tqdm
import yaml
from sklearn.metrics import confusion_matrix

from CheXpert2.dataloaders.CxrayDataloader import CxrayDataloader
from CheXpert2.training.training import validation_loop
from sklearn import metrics
import scipy
import os
import wandb
from sklearn.metrics import roc_curve, auc

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


def init_argparse():
    parser = argparse.ArgumentParser(description="Launch testing for a specific model")

    parser.add_argument(
        "-t",
        "--dataset",
        default="unseen",
        type=str,
        choices=["training", "validation"],
        required=True,
        help="Choice of the test set",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="alexnet",
        type=str,
        choices=["alexnet", "resnext50_32x4d", "vgg19", "densenet201"],
        required=True,
        help="Choice of the model",
    )

    return parser


def find_thresholds(true, pred):

    true, pred = true.numpy(), pred.numpy()

    def f1(thresholds, true, pred):
        pred2 = np.zeros_like(pred)
        for ex, item in enumerate(pred):
            item2 = np.where(item > thresholds, 1, 0)
            pred2[ex] = item2

        return -metrics.f1_score(true, pred2, average="macro", zero_division=0)

    x0 = np.zeros((14)) + 0.5
    thresholds = scipy.optimize.minimize(f1, x0, args=(true, pred))
    return thresholds.x


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

    y_true, y_pred = convert(results[0], thresholds), convert(results[1], thresholds)
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

    test_dataset = CxrayDataloader(f"data/{args.dataset}", num_classes=14)

    # ----------------loading model -------------------------------
    from CheXpert2.models.CNN import CNN

    model = CNN(args.model, 14)
    model = torch.nn.DataParallel(model)
    if not os.path.exists(f"models/models_weights/{args.model}/DataParallel.pt"):
        wandb.restore(
            f"models/models_weights/{args.model}/DataParallel.pt",
            run_path="ai-chexnet/test-project/1oc66oio",
        )
    model.load_state_dict(
        torch.load(
            f"models/models_weights/{args.model}/DataParallel.pt"  # TODO : load .pt and check name for if dataparallel
            #    ,map_location=torch.device('cpu')
        )
    )
    model = model.to(device)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=80,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    a = time.time()
    running_loss, results = validation_loop(
        model=model, loader=tqdm.tqdm(test_loader), criterion=criterion, device=device
    )
    print("time :", (time.time() - a) / len(test_dataset))

    m = create_confusion_matrix(results)
    # -----------------------------------------------------------------------------------

    with open("data/data.yaml", "r") as stream:  # TODO : remove hardcode
        names = yaml.safe_load(stream)["names"]

    # np.savetxt(f"{model._get_name()}_confusion_matrix.txt",m)
    print("avg class : ", np.mean(np.diag(m)))

    z_text = [[str(y) for y in x] for x in m]

    import plotly.figure_factory as ff

    fig = ff.create_annotated_heatmap(
        m, x=names, y=names, annotation_text=z_text, colorscale="Blues"
    )

    fig.update_layout(
        margin=dict(t=50, l=200),
        # title="ResNext50 3.0",
        xaxis_title="True labels",
        yaxis_title="Predictions",
    )

    fig["data"][0]["showscale"] = True
    import plotly.io as pio

    pio.write_image(fig, f"{model._get_name()}_conf_mat.png", width=1920, height=1080)
    fig.show()


if __name__ == "__main__":
    main()
