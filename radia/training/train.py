#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on 2023-0119$

@author: Jonathan Beaulieu-Emond
"""


# ------python import------------------------------------
import os
import sys
import warnings


import torch
import torch.distributed as dist
import logging
import wandb

# -----local imports---------------------------------------
from radia.models.CNN import CNN
from radia.models.Hierarchical import Hierarchical
from radia.models.Weighted_hierarchical import Weighted_hierarchical
from radia.models.Weighted import Weighted
from radia.Experiment import Experiment
from radia.Parser import init_parser
from radia import names, hierarchy

for key in hierarchy.keys():
    if key not in names:
        names.insert(0, key)

# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
# torch.autograd.set_detect_anomaly(True)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True

# torch.set_float32_matmul_precision('high')
# --------load local variable ----------------------------

try:
    img_dir = os.environ["img_dir"]
except:
    os.environ["img_dir"] = ""


def initialize_config(args):
    # -------- proxy config ---------------------------
    # import urllib
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

    # ------------ parsing & Debug -------------------------------------

    if args.debug:
        os.environ["WANDB_MODE"] = "offline"
    # 2) load from env.variable the data repository location
    try:
        img_dir = os.environ["img_dir"]
    except:
        img_dir = "data"

    # ---------- Device Selection ----------------------------------------
    if torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"
        logging.critical("No gpu is available for the computation")

    # ----------- hyperparameters-------------------------------------<

    config = vars(args)
    experiment = Experiment(
        f"{args.model}",
        names=names,
        tag=None,
        config=config,
        epoch_max=args.pretraining,
        patience=40,
    )

    config = wandb.config
    torch.set_num_threads(max(config["num_worker"], 1))

    return config, img_dir, experiment, device


def main():

    # -------------- set up logging -----------------------------------
    logging.basicConfig(filename="RADIA.log", level=logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    # ------------ parse argument and get config -------------------
    parser = init_parser()
    args = parser.parse_args()
    config, img_dir, experiment, device = initialize_config(args)
    num_classes = len(names)

    # -----------model initialisation------------------------------

    model = CNN(
        config["model"],
        num_classes,
        pretrained=config["pretrained"],
        channels=config["channels"],
        drop_rate=config["drop_rate"],
        # global_pool=config["global_pool"],
        # hierarchical=config["hierarchical"],
    )

    if config["global_pool"]=="weighted" and config["hierarchical"] :
        Model = Weighted_hierarchical
    elif config["global_pool"]=="weighted" :
        Model = Weighted
    elif config["hierarchical"] :
        Model = Hierarchical
    else :
        Model = CNN

    model = Model(
            backbone_name=config["model"],
            num_classes=num_classes,
            pretrained=config["pretrained"],
            channels=config["channels"],
            drop_rate=config["drop_rate"],
            # global_pool=config["global_pool"],
            # hierarchical=config["hierarchical"],
        )

    # model = torch.compile(model)
    # send model to gpu

    logging.info("The model has now been successfully loaded into memory")

    # ------------pre-training--------------------------------------
    if config["pretraining"] != 0:

        experiment.compile(
            model,
            optimizer="AdamW",
            criterion="BCEWithLogitsLoss",
            train_datasets=["CIUSSS", "PadChest"],
            val_datasets=["vinBigData", "MimicCxrJpg"],
            config=config,
            device=device,
        )
        results = experiment.train()
        model.reset_classifier()
        model = model.to(device)

    # ------------training--------------------------------------

    # setting up for the training

    config["train_dataset"] = ["ChexPert"]
    config["val_dataset"] = ["ChexPert"]

    experiment = Experiment(
        f"{config['model']}",
        names=names,
        tag=config["tag"],
        config=config,
        epoch_max=config["epoch"],
        patience=40,
    )

    experiment.compile(
        model=model,
        optimizer="AdamW",
        criterion="BCEWithLogitsLoss",
        train_datasets=config["train_dataset"],
        val_datasets=config["val_dataset"],
        config=config,
        device=device,
    )

    results = experiment.train()
    experiment.end(results)


if __name__ == "__main__":
    main()
