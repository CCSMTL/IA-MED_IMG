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
    config, img_dir, experiment, experiment2, optimizer, optimizer2, criterion, device, prob, sampler = initialize_config()
    main(config, img_dir, experiment, experiment2, optimizer, optimizer2, criterion, device, prob, sampler)
    cleanup()
