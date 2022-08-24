#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-28$

@author: Jonathan Beaulieu-Emond
"""
import os
import itertools
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mne_connectivity.viz import plot_connectivity_circle
from CheXpert2.dataloaders.MongoDB import MongoDB


os.environ["DEBUG"]="False"
with open("data/data.yaml", "r") as stream:
    names = yaml.safe_load(stream)["names"]

def chord_chexpert():
    data = MongoDB("10.128.107.212", 27017, ["ChexPert", "ChexNet", "ChexXRay"]).dataset("Train",[]).fillna(0)

    data = data[names].astype(int).replace(-1,1)
    n = len(names)
    conn = np.zeros((n, n))
    # for ex, line in data.iterrows():
    #     line = line.to_numpy()
    #     diseases = np.where(line == 1)[0].tolist()
    #     while diseases:
    #         disease = diseases.pop()
    #         for next_disease in diseases:
    #             conn[disease][next_disease] += 1

    for ex, line in data.iterrows():
        line = line.to_numpy()
        diseases = np.where(line == 1)[0]

        for disease in diseases :
            conn[disease,:]+=line
            conn[:,disease] +=line


    np.fill_diagonal(conn,0)
    conn = conn / np.sum(conn)
    fig, axes = plot_connectivity_circle(conn, data.columns)
    fig.savefig("chords_mongodb")
    plt.title("Correlation map between diseases in chexnet")
    fig.show()


def histogram_chexpert():
    data = MongoDB("10.128.107.212", 27017, ["ChexPert", "ChexNet", "ChexXRay"]).dataset("Train", []).fillna(0)

    data = data[names].astype(int).to_numpy()
    n = len(names)
    counts = np.zeros((3, n))
    counts[0, :] = np.sum(np.where(data == -1, 1, 0), axis=0)
    counts[1, :] = np.sum(np.where(data == 0, 1, 0), axis=0)
    counts[2, :] = np.sum(np.where(data == 1, 1, 0), axis=0)
    print(counts)

    # plt.xticks(rotation=45, fontsize=6)

    labels = ["-1", "1"]
    data = {"-1": counts[0, :], "1": counts[2, :]}
    df = pd.DataFrame(data, columns=labels, index=names)
    fig,ax=plt.subplots()
    df.plot.barh(ax=ax)
    plt.xlabel("Classes")  # , fontsize = 60)
    plt.ylabel("Count")  # , fontsize = 60)
    plt.legend()  # prop={'size':45})
    plt.title("Count of each class")
    fig.savefig("histogram_mongodb")
    fig.show()


if __name__ == "__main__":

    #histogram_chexpert()
    chord_chexpert()
