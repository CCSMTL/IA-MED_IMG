# ------python import------------------------------------
import argparse
import time
import warnings
import numpy as np
import torch
import tqdm
from sklearn.metrics import multilabel_confusion_matrix
import yaml

from CheXpert2.training.training import validation_loop
from CheXpert2.dataloaders.CxrayDataloader import CxrayDataloader


def init_argparse():
    parser = argparse.ArgumentParser(description="Launch testing for a specific model")

    parser.add_argument(
        "-t",
        "--testset",
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
        choices=["alexnet", "resnext50_32x4d", "vgg19","densenet201"],
        required=True,
        help="Choice of the model",
    )

    return parser


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
    preprocess=CxrayDataloader.get_preprocess(3, 320)

    test_dataset = CxrayDataloader(
        f"data/{args.dataset}/images", transform=preprocess
    )

    # ----------------loading model -------------------------------
    from CheXpert2.models.CNN import CNN
    model=CNN(args.model,14)

    model.load_state_dict(
        torch.load(
            f"models/models_weights/{args.model}/{args.model}.pt" #?
        )
    )
    model = model.to(device)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )  # keep

    a = time.time()
    running_loss, results = validation_loop(
        model=model, loader=tqdm.tqdm(test_loader), criterion=criterion, device=device
    )
    print("time :", (time.time() - a) / len(test_dataset))

    from CheXpert2.Metrics import metrics  # had to reimport due to bug



    y_true, y_pred = results[0].numpy().round(0), results[1].numpy().round(0)

    # -----------------------------------------------------------------------------------
    metric = metrics(num_classes=14)
    metrics = metric.metrics()

    for metric in metrics.keys():
        print(metric + " : ", metrics[metric](y_true, y_pred))

    with open("data/data.yaml", "r") as stream:  # TODO : remove hardcode
        names = yaml.safe_load(stream)["names"]

    names += ["No Finding"]
    m = multilabel_confusion_matrix(y_true, y_pred, labels=names).round(2)
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
