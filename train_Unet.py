# ------python import------------------------------------
import warnings

import pandas as pd
import torch
import wandb
import os
import argparse
import torchvision
import numpy as np
import copy
from models.Unet import Unet
# -----local imports---------------------------------------

from training.training import training
from training.dataloaders.CxrayDataloader import CxrayDataloader
from custom_utils import Experiment, set_parameter_requires_grad
from parser import init_parser

# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
# torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
# torch.backends.cudnn.benchmark = True

# ----------- parse arguments----------------------------------





def main():
    parser = init_parser()
    args = parser.parse_args()
    os.environ["DEBUG"] = str(args.debug)
    max_batch_size = args.batch_size  # defines the maximum batch_size supported by your gpu for a specific model.
    accumulate = args.accumulate
    # ----------- hyperparameters-------------------------------------
    config = {
        #   "beta1"
        #   "beta2"
        "optimizer": torch.optim.AdamW,
        "criterion": torch.nn.CrossEntropyLoss(),#reduction='none'),
        "augment prob": 0,
        "augment intensity": 0,
        "label smoothing": 0,
        # "sampler"
        "gradient accum": accumulate,
        "num_worker": args.num_worker,
    }
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

    # ---------- Sampler -------------------------------------------
    from Sampler import Sampler

    Sampler = Sampler()

    # -----------model initialisation------------------------------
    model = Unet(args.model)
    if args.device == "parallel":
        model = torch.nn.DataParallel(model)
    print(
        f"mini batch size : {max_batch_size}. The gradient will be accumulated {accumulate} times"
    )
    if torch.cuda.is_available():
        device = f"cuda:{args.device}" if args.device != "parallel" else "cuda:0"

    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")



    print("The model has now been successfully loaded into memory")

    # -------data initialisation-------------------------------
    # os.environ["WANDB_MODE"] = "offline"

    from Metrics import Metrics

    train_dataset = CxrayDataloader(
        f"data/training",
        num_classes=14,
        img_size=args.img_size,
        prob=config["augment prob"],
        intensity=config["augment intensity"],
        label_smoothing=config["label smoothing"],
    )
    val_dataset = CxrayDataloader(
        f"data/validation", num_classes=14, img_size=args.img_size
    )

    # rule of thumb : num_worker = 4 * number of gpu ; on windows leave =0
    # batch_size : maximum possible without crashing

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=max_batch_size,
        num_workers=config["num_worker"],
        pin_memory=True,
        sampler=Sampler.sampler(),
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(max_batch_size * 2), num_workers=config["num_worker"], pin_memory=True
    )
    print("The data has now been loaded successfully into memory")

    # ------------training--------------------------------------------
    print("Starting training now")

    # send model to gpu
    model = model.to(device)

    # initialize metrics loggers
    optimizer = config["optimizer"](model.parameters())

    config = config | vars(args)

    if args.wandb:
        wandb.init(
            project="test-project", entity="ai-chexnet", config=copy.copy(config)
        )

        wandb.watch(model)

    experiment = Experiment(
        "Unet", is_wandb=args.wandb, tags=[args.model], config=copy.copy(config)
    )

    metric = Metrics(num_classes=14, threshold=np.zeros((14)) + 0.5)
    metrics = metric.metrics()
    training(
        model,
        optimizer,
        config["criterion"],
        training_loader,
        validation_loader,
        device,
        minibatch_accumulate=accumulate,
        epoch_max=args.epoch,
        patience=5,
        experiment=experiment,
        metrics=[],
    )


if __name__ == "__main__":
    main()
