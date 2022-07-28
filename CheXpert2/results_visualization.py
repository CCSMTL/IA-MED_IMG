#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-20$

@author: Jonathan Beaulieu-Emond
"""
import os

import pandas as pd
import plotly.graph_objects as go


def plot_polar_chart(summary):
    df = pd.read_csv("data/chexnet_results.csv", index_col=0, na_values=0)
    # df.columns = ["chexnet", "chexpert"]
    summary["auc"]["No Finding"] = 0
    df["ours"] = pd.Series(summary["auc"])

    df.fillna(0, inplace=True)

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
    if os.environ["DEBUG"] == "False":
        fig.write_image("polar.png")

    # wandb.log({"polar_chart": fig})
    # for key, value in summary.items():
    #    wandb.run.summary[key] = value
