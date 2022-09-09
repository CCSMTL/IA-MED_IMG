#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-28$

@author: Jonathan Beaulieu-Emond
"""
import os

import torch
import yaml

from CheXpert2.Experiment import Experiment
from CheXpert2.models.CNN import CNN
from CheXpert2.training.train import main


def test_train():
    try:
        img_dir = os.environ["img_dir"]
    except:
        img_dir = ""

    torch.cuda.is_available = lambda: False
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DEBUG"] = "True"
    os.environ["WANDB_MODE"] = "offline"

    config = {
        "model": "densenet121",
        "batch_size": 100,
        "img_size": 223,
        "num_worker": 0,
        "augment_intensity": 0,
        "cache": False,
        "N": 0,
        "M": 2,
        "clip_norm": 100,
        "label_smoothing": 0,
        "lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "freeze": False,
        "pretrained": False,
        "pretraining": 0,
        "channels": 1,
        "autocast": True,
        "pos_weight": 1,
    }
    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]

    experiment = Experiment(
        f"{config['model']}", names=names, tags=None, config=config, epoch_max=1, patience=5
    )
    optimizer = torch.optim.AdamW
    criterion = torch.nn.BCEWithLogitsLoss
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prob = [0, ] * 5
    model = CNN(config["model"], 15, img_size=config["img_size"], freeze_backbone=config["freeze"],
                pretrained=config["pretrained"], channels=config["channels"], pretraining=False)
    main(config, img_dir, model, experiment, optimizer, criterion, device, prob, metrics=None, pretrain=False)
    assert experiment.best_loss != 0


if __name__ == "__main__":
    test_train()
