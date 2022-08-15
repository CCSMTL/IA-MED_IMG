#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-13$

@author: Jonathan Beaulieu-Emond
"""
import copy
import os

import numpy as np
import torch
import torch.distributed as dist

from CheXpert2.models.CNN import CNN
from CheXpert2.train import main, initialize_config


def cleanup():
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    config, img_dir, experiment, experiment2, device, prob, sampler = initialize_config()
    # -----------model initialisation------------------------------

    model = CNN(config["model"], 4, img_size=config["img_size"], freeze_backbone=config["freeze"],
                pretrained=config["pretrained"], channels=config["channels"])
    # send model to gpu

    print("The model has now been successfully loaded into memory")
    # ---pretraining-------------------------------------
    experiment2 = copy.copy(experiment)
    experiment2.max_epoch = 5
    from torch.utils.data.sampler import SequentialSampler

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    sampler = torch.utils.data.DistributedSampler(SequentialSampler(sampler))
    main(config, img_dir, model, experiment2, device, prob, sampler, None, pretrain=True)

    # -----setting up training-------------------------------------
    dist.barrier()
    model.module.backbone.reset_classifier(14)
    model2 = CNN(config["model"], 14, img_size=config["img_size"], freeze_backbone=config["freeze"],
                 pretrained=False, channels=config["channels"])
    model2.load_state_dict(model.module.state_dict())

    model2.pretrain = False
    model = model2.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model)
    from CheXpert2.Metrics import Metrics  # sklearn f**ks my debug

    metric = Metrics(num_classes=14, names=experiment.names, threshold=np.zeros((14)) + 0.5)
    metrics = metric.metrics()

    # -----training-------------------------------------------

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    main(config, img_dir, model, experiment, device, prob, sampler, metrics, pretrain=False)
    cleanup()
