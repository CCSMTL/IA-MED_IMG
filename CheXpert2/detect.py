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
from CheXpert2.models.Ensemble import Ensemble
#from polycam.polycam.polycam import PCAMp
from CheXpert2.Metrics import Metrics
import yaml
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

os.environ["DEBUG"] = "False"


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

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.detach()

        if inputs.shape != labels.shape:  # prevent storing images if training unets
            results[1] = torch.cat(
                (results[1], outputs.detach().cpu()), dim=0
            )
            results[0] = torch.cat((results[0], labels.cpu()), dim=0)

        del (
            inputs,
            labels,
            outputs,
            loss,
        )  # garbage management sometimes fails with cuda


    return running_loss, results



def set_thresholds(self, true, pred):
    best_threshold = np.zeros((self.num_classes))
    for i in range(self.num_classes):
        max_score = 0
        for threshold in np.arange(0.01, 1, 0.01):
            pred2 = np.where(np.copy(pred[:, i]) > threshold, 1, 0)
            score = metrics.f1_score(
                true[:, i], pred2, average="macro", zero_division=0
            )  # weighted??
            if score > max_score:
                max_score = score
                best_threshold[i] = threshold

    return best_threshold


def main():
    criterion = torch.nn.BCELoss()
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    # ----- parsing arguments --------------------------------------


    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)
    # ------loading test set --------------------------------------
    img_dir = os.environ["img_dir"]
    #img_dir = "data"
    test_dataset = Chexpertloader(f"{img_dir}/valid.csv", img_dir, img_size=384,channels=1,N=0,M=0,pretrain=False)


    # ----------------loading model -------------------------------

    models = [
        CNN("convnext_base", img_size=384, channels=1, num_classes=14, pretrained=False, pretraining=False),
        CNN("convnext_base", img_size=384, channels=1, num_classes=14, pretrained=False, pretraining=False),
        CNN("densenet201", img_size=384, channels=1, num_classes=14, pretrained=False, pretraining=False),
        CNN("densenet201", img_size=384, channels=1, num_classes=14, pretrained=False, pretraining=False),
    ]
    # model =  torch.nn.parallel.DistributedDataParallel(model)

    #api = wandb.Api()
    # run = api.run(f"ccsmtl2/Chestxray/{args.run_id}")
    # run.file("models_weights/convnext_base/DistributedDataParallel.pt").download(replace=True)
    weights = [
        "/data/home/jonathan/IA-MED_IMG/models_weights/convnext_base.pt",
        "/data/home/jonathan/IA-MED_IMG/models_weights/convnext_base_2.pt",
        "/data/home/jonathan/IA-MED_IMG/models_weights/densenet201.pt",
        "/data/home/jonathan/IA-MED_IMG/models_weights/densenet201_2.pt",
    ]

    for model,weight in zip (models, weights) :
        # state_dict = torch.load(weight)
        #
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        #     new_state_dict[name] = v

        #model.load_state_dict(new_state_dict)
        model.load_state_dict(torch.load(weight,map_location=torch.device(device)))
        #model = model.to(device)
        model.eval()
    ensemble = Ensemble(models,14)
    ensemble.to(device)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    #start.record()
    import time
    start = time.time()
    running_loss, results = infer_loop(model=model, loader=test_loader, criterion=criterion, device=device)
    #end.record()
    end=time.time()
    #torch.cuda.synchronize()
    #print("time : ", start.elapsed_time(end))
    print(end-start)
    #plt.imshow(np.sum(heatmaps[0][0].detach().cpu().numpy(), axis=0))
    #plt.savefig("heatmaps.png")

    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]
    metric = Metrics(num_classes=14, names=names, threshold=np.zeros((14)) + 0.5)
    metrics = metric.metrics()
    metrics_results = {}
    for key in metrics:
        pred = results[1].numpy()
        true = results[0].numpy().round(0)
        metric_result = metrics[key](true, pred)
        metrics_results[key] = metric_result

    print(metrics_results)

if __name__ == "__main__":
    main()
