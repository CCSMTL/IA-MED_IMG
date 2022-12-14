#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-28$

@author: Jonathan Beaulieu-Emond
"""
import os

import torch
import yaml
import logging
from CheXpert2.Experiment import Experiment
from CheXpert2.models.CNN import CNN
from CheXpert2.training.train import main
from CheXpert2 import names,debug_config

def test_train():
    #-------- proxy config ---------------------------
    import urllib
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
    verbose=5
    logging.basicConfig(filename='RADIA.log', level=verbose * 10)
    try:
        img_dir = os.environ["img_dir"]
    except:
        os.environ["img_dir"] = ""

    #torch.cuda.is_available = lambda: False
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""

    os.environ["WANDB_MODE"] = "offline"



    experiment = Experiment(
        f"{debug_config['model']}", names=names, tag=None, config=debug_config, epoch_max=1, patience=1
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = CNN(debug_config["model"], len(names),
                pretrained=debug_config["pretrained"], channels=debug_config["channels"])

    experiment.compile(
        model=model,
        optimizer="AdamW",
        criterion="BCEWithLogitsLoss",
        train_datasets=["ChexPert"],
        val_datasets=["ChexPert"],
        config=debug_config,
        device=device
    )
    results = experiment.train()
    assert experiment.best_loss != 0



if __name__ == "__main__":
    test_train()
