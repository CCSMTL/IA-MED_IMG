#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-19$

@author: Jonathan Beaulieu-Emond
"""
import os

import numpy as np
import timm

from CheXpert2.Experiment import Experiment
from CheXpert2 import debug_config
from CheXpert2.models.CNN import CNN
os.environ["WANDB_MODE"] = "offline"
experiment = Experiment(directory="/debug", names=np.arange(0, 13).astype(str))

def test_experiment_compile():
    os.environ["DEBUG"]="True"
    experiment.compile(
        model=CNN("convnext_tiny", num_classes=13, img_size=224, freeze_backbone=False, pretrained=False, channels=3),
        optimizer="AdamW",
        criterion="BCEWithLogitsLoss",
        train_datasets=["ChexPert"],
        val_datasets=["ChexPert"],
        config=debug_config,
        device="cpu"
    )


def test_experiment_log_metric():
    os.environ["DEBUG"] = "True"
    experiment.log_metric("auc", {"banana": 0})


def test_experiment_log_metrics():
    os.environ["DEBUG"] = "True"
    experiment.log_metrics({"apple": 3})


def test_experiment_next_epoch():
    os.environ["DEBUG"] = "True"
    experiment.next_epoch(val_loss=0)


# def test_experiment_end_result() :
#     os.environ["DEBUG"] = "True"
#     results = [torch.randint(0, 2, size=(100,13)), torch.rand(size=(100,13))]
#     experiment.end(results)



if __name__ == "__main__":
    test_experiment_log_metric()
    test_experiment_log_metrics()
    test_experiment_next_epoch()
    # test_experiment_end_result()
