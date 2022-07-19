#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-19$

@author: Jonathan Beaulieu-Emond
"""
import pathlib

import numpy as np
import pandas as pd
import torch
import tqdm

import wandb
from CheXpert2.custom_utils import convert


class Experiment:
    def __init__(self, directory, names, tags=None, config=None, epoch_max=50, patience=5):
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
        # clean old files/previous weights
        # root, dir, files = list(os.walk(self.directory))[0]
        # for f in files:
        #     os.remove(root + "/" + f)

        wandb.init(project="Chestxray", entity="ccsmtl2", config=config)

    def next_epoch(self, val_loss, model):

        if self.rank == 0:
            if val_loss < self.best_loss or self.epoch == 0:
                self.best_loss = val_loss
                self.save_weights(model)
                self.patience = self.max_patience
                self.summarize()
            else:
                self.patience -= 1
                print("patience has been reduced by 1")
            self.pbar.update(1)
            self.epoch += 1
        if self.patience == 0 or self.epoch == self.epoch_max:
            self.keep_training = False

    def log_metric(self, metric_name, value, epoch=None):

        if epoch:
            wandb.log({metric_name: value, "epoch": epoch})
        else:
            wandb.log({metric_name: value})
        self.metrics[metric_name] = value

    def log_metrics(self, metrics, epoch=None):

        metrics["epoch"] = epoch
        wandb.log(metrics)
        self.metrics = self.metrics | metrics

    def save_weights(self, model):
        if self.rank == 0:
            torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")
            wandb.save(f"{self.weight_dir}/{model._get_name()}.pt")

    def summarize(self):
        self.summary = self.metrics

    def watch(self, model):
        wandb.watch(model)

    def end(self, results):
        if self.rank == 0:
            # 1) confusion matrix

            self.log_metric(
                "conf_mat",
                wandb.sklearn.plot_confusion_matrix(
                    convert(results[0]),
                    convert(results[1]),
                    self.names,
                ),
                epoch=None
            )

            df = pd.read_csv("data/chexnet_results.csv", index_col=0, na_values=0)
            # df.columns = ["chexnet", "chexpert"]
            self.summary["auc"]["No Finding"] = 0
            df["ours"] = pd.Series(self.summary["auc"])

            df.fillna(0, inplace=True)

            import plotly.graph_objects as go

            fig = go.Figure(
                data=[
                    go.Scatterpolar(r=(df["chexnet"] * 100).round(0), fill='toself', name='chexnet'),
                    go.Scatterpolar(r=(df["Chexpert"] * 100).round(0), fill='toself',
                                    name='Chexpert'),
                    go.Scatterpolar(r=(df["ours"] * 100).round(0), fill='toself', name='ours')
                ],
                layout=go.Layout(
                    title=go.layout.Title(text='Class AUC'),
                    polar={'radialaxis': {'visible': True}},
                    showlegend=True,
                    template="plotly_dark"
                )
            )
            # fig.update_polar(ticktext=names)
            fig.write_image("polar.png")
            wandb.log({"polar_chart": fig})
            for key, value in self.summary.items():
                wandb.run.summary[key] = value
