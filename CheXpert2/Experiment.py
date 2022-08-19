#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-19$

@author: Jonathan Beaulieu-Emond
"""
import os
import pathlib

import numpy as np
import torch
import tqdm

import wandb
from CheXpert2.custom_utils import convert
from CheXpert2.results_visualization import plot_polar_chart


class Experiment:
    def __init__(self, directory, names, tags=None, config=None, epoch_max=50, patience=5,no_log=False):
        self.names = names
        self.weight_dir = "models_weights/" + directory

        self.summary = {}
        self.metrics = {}
        self.pbar = tqdm.tqdm(total=epoch_max, position=0)
        self.best_loss = np.inf
        self.keep_training = True
        self.epoch = 0
        self.epoch_max = epoch_max
        self.max_patience = patience
        self.patience = patience
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # create directory if doesnt existe
        path = pathlib.Path(self.weight_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            wandb.init(project="Chestxray", entity="ccsmtl2", config=config)

        self.no_log = no_log
    def next_epoch(self, val_loss, model):

        if val_loss < self.best_loss or self.epoch == 0:
            self.best_loss = val_loss
            self.log_metric("best_loss", self.best_loss, epoch=self.epoch)
            self.patience = self.max_patience
            self.summarize()
            self.save_weights(model)
        else:
            self.patience -= 1
            print("patience has been reduced by 1")
            print(val_loss)
        self.pbar.update(1)
        self.epoch += 1
        if self.patience == 0 or self.epoch == self.epoch_max:
            self.keep_training = False
        print(self.summary)


    def log_metric(self, metric_name, value, epoch=None):
        if self.rank == 0 and not self.no_log:
            if epoch is not None:
                wandb.log({metric_name: value, "epoch": epoch})
            else:
                wandb.log({metric_name: value})
            self.metrics[metric_name] = value

    def log_metrics(self, metrics, epoch=None):
        if self.rank == 0 and not self.no_log:
            metrics["epoch"] = epoch
            wandb.log(metrics)
            self.metrics = self.metrics | metrics

    def save_weights(self, model):
        if self.rank == 0 and os.environ["DEBUG"] == "False" and not self.no_log:
            torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")
            wandb.save(f"{self.weight_dir}/{model._get_name()}.pt")

    def summarize(self):
        self.summary = self.metrics

    def watch(self, model):
        if self.rank == 0 and not self.no_log:
            wandb.watch(model)

    def end(self, results):

        if self.rank == 0 and not self.no_log:
            # 1) confusion matrix

            self.log_metric(
                "conf_mat",
                wandb.sklearn.plot_confusion_matrix(
                    convert(results[0]),
                    convert(results[1]),
                    self.names,
                ),
                epoch=None)
            plot_polar_chart(self.summary)


if __name__ == "__main__":
    experiment = Experiment(dir="/debug", names=["1", "2", "3"])
    os.environ["DEBUG"] = "True"

    experiment.log_metric("auc", {"banana": 0})

    experiment.log_metrics({"apple": 3})

    experiment.next_epoch(3, None)

    results = [torch.randint(0, 2, size=(10, 13)), torch.rand(size=(10, 13))]
    experiment.end(results)
