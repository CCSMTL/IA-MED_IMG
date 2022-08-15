#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-13$

@author: Jonathan Beaulieu-Emond
"""
import torch
import torch.distributed as dist

from CheXpert2.train import main, initialize_config


def cleanup():
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    config, img_dir, experiment, experiment2, device, prob, sampler = initialize_config()

    #---pretraining-------------------------------------
    main(config, img_dir, experiment, experiment2, device, prob, sampler,pretrain=True)


    #-----setting up training-------------------------------------
    dist.barrier()
    model.module.backbone.reset_classifier(14)
    model2 = CNN(config["model"], 14, img_size=config["img_size"], freeze_backbone=config["freeze"],
                 pretrained=False, channels=config["channels"])
    model2.load_state_dict(model.module.state_dict())
    model = model2.to(device)
    model.pretrain = False
    model = torch.nn.parallel.DistributedDataParallel(model)


    #-----training-------------------------------------------
    main(config, img_dir, experiment, experiment2, device, prob, sampler,pretrain=False)
    cleanup()
