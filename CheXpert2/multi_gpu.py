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
from torch.utils.data.sampler import SequentialSampler

from CheXpert2.Experiment import Experiment
from CheXpert2.custom_utils import set_parameter_requires_grad
from CheXpert2.models.CNN import CNN
from CheXpert2.train import main, initialize_config


def cleanup():
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    config, img_dir, experiment, device, prob, sampler, names = initialize_config()
    from CheXpert2.Sampler import Sampler

    Sampler2 = Sampler(f"{img_dir}/train.csv")
    sampler2= Sampler2.sampler()
    sampler2.weights = 1 / sampler2.weights
    sampler2 = torch.utils.data.DistributedSampler(SequentialSampler(sampler2))
    sampler = torch.utils.data.DistributedSampler(SequentialSampler(sampler))
    optimizer = torch.optim.AdamW
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    # -----------model initialisation------------------------------

    model = CNN(config["model"], 14, img_size=config["img_size"], freeze_backbone=config["freeze"],
                pretrained=config["pretrained"], channels=config["channels"])
    # send model to gpu
    model = model.to(device)
    print("The model has now been successfully loaded into memory")

    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ---pretraining-------------------------------------
    if config["pretraining"] != 0:
        experiment2 = Experiment(
            f"{config['model']}", names=names, tags=None, config=config, epoch_max=5, patience=5
        )
        results = main(config, img_dir, model, experiment2, optimizer, torch.nn.MSELoss(), device, prob, sampler2,
                       metrics=None, pretrain=True)

    # -----setting up training-------------------------------------
    dist.barrier()
    set_parameter_requires_grad(model.module.features)
    from CheXpert2.Metrics import Metrics  # sklearn f**ks my debug
    metric = Metrics(num_classes=14, names=experiment.names, threshold=np.zeros((14)) + 0.5)
    metrics = metric.metrics()

    # -----training-------------------------------------------


    results = main(config, img_dir, model, experiment, optimizer, torch.nn.MSELoss(), device, prob, sampler,
                   metrics=metrics, pretrain=False)
    experiment.end(results)
    cleanup()
