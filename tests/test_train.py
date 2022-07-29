#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-28$

@author: Jonathan Beaulieu-Emond
"""
import os

import pandas as pd
import torch

from CheXpert2.Experiment import Experiment
from CheXpert2.train import main

#
# def test_train():
#     torch.cuda.is_available = lambda: False
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     os.environ["DEBUG"] = "True"
#     os.environ["WANDB_MODE"] = "offline"
#
#     config = {
#         "model": "densenet201",
#         "batch_size": 2,
#         "img_size": 320,
#         "num_worker": 0,
#         "augment_intensity": 0,
#         "cache": False,
#         "N": 0,
#         "M": 2,
#         "clip_norm": 100,
#         "label_smoothing": 0,
#         "lr": 0.001,
#         "beta1": 0.9,
#         "beta2": 0.999,
#         "weight_decay": 0.01,
#     }
#
#     img_dir = "tests/data_test"
#     names = pd.read_csv("tests/data_test/valid.csv").columns[5:19]
#     names = names + ["No Findings"]
#     experiment = Experiment(
#         f"{config['model']}", names=names, tags=None, config=config, epoch_max=1, patience=5
#     )
#     optimizer = torch.optim.AdamW
#     criterion = torch.nn.BCEWithLogitsLoss()
#     device = "cpu"
#     prob = [0, ] * 5
#     main(config, img_dir, experiment, optimizer, criterion, device, prob, sampler=None)
#     assert experiment.best_loss != 0
#

if __name__ == "__main__":
    test_train()
