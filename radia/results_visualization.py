#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-20$

@author: Jonathan Beaulieu-Emond
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import wandb
import plotly.express as px


def plot_polar_chart(summary):

    # df = pd.read_csv("data/chexnet_results.csv", index_col=0, na_values=0).T
    df = pd.DataFrame([])
    ours = pd.DataFrame(summary["auc"], index=["ours"])

    df = pd.concat([df, ours], join="outer").T
    df.fillna(0, inplace=True)
    print(df)
    columns = list(df.index)
    fig = go.Figure(
        data=[
            #    go.Scatterpolar(r=(df["chexnet"] * 100).round(0), fill='toself', name='chexnet',
            #                    theta=columns),
            #    go.Scatterpolar(r=(df["Chexpert"] * 100).round(0), fill='toself', theta=columns,
            #                    name='Chexpert'),
            go.Scatterpolar(
                r=(df["ours"] * 100).round(0), fill="toself", name="ours", theta=columns
            )
        ],
        layout=go.Layout(
            # title=go.layout.Title(text='Class AUC'),
            polar={"radialaxis": {"visible": True}},
            showlegend=True,
            template="plotly_dark",
        ),
    )

    fig.show()
    # fig.write_image("polar.png")

    wandb.log({"polar_chart": fig})


def plot_bar_chart(summary):
    summary = summary["auc"]

    summary = pd.DataFrame(list(summary.items()), columns=["classes", "AUC"])
    print(summary)
    fig = px.bar(
        summary,
        x="classes",
        y="AUC",
        template="plotly_white",
        title="AUROC per category",
        color="AUC",
        color_continuous_scale=px.colors.sequential.Viridis,
    ).update_xaxes(categoryorder="total descending")

    fig.update_layout(
        xaxis_title="Pathologies",
        yaxis_title="Count",
    )

    fig.show()
    fig.write_image("histogram_mongodb_ciusss_train.png")


if __name__ == "__main__":

    api = wandb.Api()
    run = api.run("ccsmtl2/Chestxray/ehhnkxtj")

    wandb_summary = run.summary
    # summary = {"auc" : {}}
    # for key in wandb_summary.keys() :
    #     if "auc" in key :
    #         print(key)
    #         summary["auc"] = wandb_summary[key]
    print(wandb_summary["auc"])
    # plot_bar_chart(wandb_summary)

    summary = {"auc": dict(wandb_summary["auc"])}
    plot_polar_chart(summary)
