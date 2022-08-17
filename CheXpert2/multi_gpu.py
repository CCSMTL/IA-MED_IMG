#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-13$

@author: Jonathan Beaulieu-Emond
"""
import os
import copy
import numpy as np
import torch
import torch.distributed as dist

from CheXpert2.Experiment import Experiment
from CheXpert2.models.CNN import CNN
from CheXpert2.train import main, initialize_config


def cleanup():
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    config, img_dir, experiment, device, prob, sampler, names = initialize_config()
    sampler2 = copy.copy(sampler)
    sampler2.weights = 1 / sampler2.weights
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    # -----------model initialisation------------------------------

    model = CNN(config["model"], 4, img_size=config["img_size"], freeze_backbone=config["freeze"],
                pretrained=config["pretrained"], channels=config["channels"])
    # send model to gpu
    model = model.to(device)
    print("The model has now been successfully loaded into memory")
    # ---pretraining-------------------------------------
    experiment2 = Experiment(
        f"{config['model']}", names=names, tags=None, config=config, epoch_max=5, patience=5
    )
    from torch.utils.data.sampler import SequentialSampler

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    sampler = torch.utils.data.DistributedSampler(SequentialSampler(sampler2))
    optimizer = torch.optim.AdamW
    results = main(config, img_dir, model, experiment2, optimizer, torch.nn.BCEWithLogitsLoss(), device, prob, sampler2,
                   metrics=None, pretrain=True)

    # -----setting up training-------------------------------------
    dist.barrier()
    model.module.backbone.reset_classifier(14)
    model2 = CNN(config["model"], 14, img_size=config["img_size"], freeze_backbone=config["freeze"],
                 pretrained=False, channels=config["channels"])
    model2.load_state_dict(model.module.state_dict())

    model2.pretrain = False
    model = model2.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)
    sampler = torch.utils.data.DistributedSampler(SequentialSampler(sampler))
    from CheXpert2.Metrics import Metrics  # sklearn f**ks my debug

    metric = Metrics(num_classes=14, names=experiment.names, threshold=np.zeros((14)) + 0.5)
    metrics = metric.metrics()

    # -----training-------------------------------------------

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    results = main(config, img_dir, model, experiment, optimizer, torch.nn.BCEWithLogitsLoss(), device, prob, sampler,
                   metrics=metrics, pretrain=False)
    experiment.end(results)
    cleanup()
