#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-13$

@author: Jonathan Beaulieu-Emond
"""
import os
import logging
import numpy as np
import torch
import torch.distributed as dist

from CheXpert2.Experiment import Experiment
from CheXpert2.custom_utils import set_parameter_requires_grad
from CheXpert2.models.CNN import CNN
from CheXpert2.training.train import initialize_config
from CheXpert2.Parser import init_parser
from CheXpert2 import names


def cleanup():
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(filename="RADIA.log", level=logging.DEBUG)
    parser = init_parser()
    args = parser.parse_args()
    config, img_dir, _, device = initialize_config(args)

    if torch.cuda.is_available():
        logging.debug("Using GPU with nccl backend")
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
    else:
        assert (
            os.environ["DEBUG"] == "True"
        ), "DEBUG is set to False but NO GPU is available"
        dist.init_process_group("gloo")
        device = "cpu"

    num_classes = len(names)

    # -----------model initialisation------------------------------

    model = CNN(
        config["model"],
        num_classes=num_classes,
        img_size=config["img_size"],
        freeze_backbone=config["freeze"],
        pretrained=config["pretrained"],
        channels=config["channels"],
    )
    model = model.to(device)
    local_rank = [int(os.environ["LOCAL_RANK"])] if torch.cuda.is_available() else None
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logging.debug("The model has now been successfully loaded into memory")
    # ---training-------------------------------------
    experiment = Experiment(
        f"{config['model']}",
        names=names,
        tag=config["tag"],
        config=config,
        epoch_max=config["epoch"],
        patience=20,
    )
    experiment.compile(
        model=model,
        optimizer="AdamW",
        criterion="BCEWithLogitsLoss",
        train_datasets=["ChexPert", "CIUSSS"],
        val_datasets=["CIUSSS"],
        config=config,
        device=device,
    )

    results = experiment.train()

    experiment.end(results)
    cleanup()
