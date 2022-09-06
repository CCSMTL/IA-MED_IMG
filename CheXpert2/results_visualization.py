#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-20$

@author: Jonathan Beaulieu-Emond
"""
import os

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import wandb


def plot_polar_chart(summary):

    df = pd.read_csv("data/chexnet_results.csv", index_col=0, na_values=0).T
    df.fillna(0, inplace=True)
    values = np.array(list(summary["auc"].values())).squeeze()
    columns =np.array(list(summary["auc"].keys())).squeeze()

    ours = pd.DataFrame(values[None,:],columns=columns,index=["ours"])


    df=pd.concat([df,ours],join="outer").T
    df.fillna(0,inplace=True)
    print(df)
    columns=list(df.index)
    fig = go.Figure(
        data=[
            go.Scatterpolar(r=(df["chexnet"] * 100).round(0), fill='toself', name='chexnet',
                            theta=columns),
            go.Scatterpolar(r=(df["Chexpert"] * 100).round(0), fill='toself', theta=columns,
                            name='Chexpert'),
            go.Scatterpolar(r=(df["ours"] * 100).round(0), fill='toself', name='ours',
                            theta=columns)
        ],
        layout=go.Layout(
            title=go.layout.Title(text='Class AUC'),
            polar={'radialaxis': {'visible': True}},
            showlegend=True,
            template="plotly_dark"
        )
    )


    fig.write_image("polar.png")

    wandb.log({"polar_chart": fig})

