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
    verbose=5
    logging.basicConfig(filename='RADIA.log', level=verbose * 10)
    try:
        img_dir = os.environ["img_dir"]
    except:
        img_dir = ""

    torch.cuda.is_available = lambda: False
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DEBUG"] = "True"
    os.environ["WANDB_MODE"] = "offline"



    experiment = Experiment(
        f"{debug_config['model']}", names=names, tag=None, config=debug_config, epoch_max=1, patience=5
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = CNN(debug_config["model"], len(names), img_size=debug_config["img_size"], freeze_backbone=debug_config["freeze"],
                pretrained=debug_config["pretrained"], channels=debug_config["channels"], pretraining=False)

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
