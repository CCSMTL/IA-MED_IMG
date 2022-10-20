#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-13$

@author: Jonathan Beaulieu-Emond
"""
import os

import numpy as np
import torch
import torch.distributed as dist

from CheXpert2.Experiment import Experiment
from CheXpert2.custom_utils import set_parameter_requires_grad
from CheXpert2.models.CNN import CNN
from CheXpert2.training.train import initialize_config
from CheXpert2.Parser import init_parser
from  CheXpert2 import names
def cleanup():
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    parser = init_parser()
    args = parser.parse_args()
    config, img_dir, experiment, device = initialize_config(args)
    num_classes = len(names)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    # -----------model initialisation------------------------------

    model = CNN(config["model"], num_classes=num_classes, img_size=config["img_size"], freeze_backbone=config["freeze"],
                pretrained=config["pretrained"], channels=config["channels"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )


    print("The model has now been successfully loaded into memory")

    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ---training-------------------------------------

    experiment.compile(
        model=model,
        optimizer="AdamW",
        criterion="BCEWithLogitsLoss",
        train_datasets=["ChexPert", "CIUSSS"],
        val_datasets=["CIUSSS"],
        config=config,
        device=device
    )

    results = experiment.train()

    experiment.end(results)
    cleanup()
