#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-28$

@author: Jonathan Beaulieu-Emond
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne_connectivity.viz import plot_connectivity_circle


def main():
    return "hello World"


mapping = {
    "Cardiomegaly": 0,
    "Emphysema": 1,
    "Effusion": 2,
    "Consolidation": 3,
    "Hernia": 4,
    "Infiltration": 5,
    "Mass": 6,
    "Nodule": 7,
    "Atelectasis": 8,
    "Pneumothorax": 9,
    "Pleural_Thickening": 10,
    "Pneumonia": 11,
    "Fibrosis": 12,
    "Edema": 13,
    "No Finding": 14,
}


def chord():
    data = pd.read_csv("data/data_table.csv")

    diseases = data["Finding Labels"].values

    diseases = [i.split("|") for i in diseases]

    conn = np.zeros((15, 15))
    for line in diseases:
        if len(line) > 1:
            prev_disease = [mapping[line[0]]]
            for disease in line[1::]:
                name = mapping[disease]
                for item in prev_disease:
                    conn[item][name] += 1

                prev_disease.append(name)
    conn = conn / np.sum(conn)
    fig, axes = plot_connectivity_circle(conn, list(mapping.keys()))
    fig.savefig("chords")
    fig.title("Correlation map between diseases in chexnet")
    fig.show()


def histogram():
    data = pd.read_csv("data/data_table.csv")

    diseases = data["Finding Labels"].values

    diseases = [i.split("|") for i in diseases]
    d_flat = np.array(list(itertools.chain(*diseases)))
    names = np.unique(d_flat)
    count = []
    for name in names:
        count.append(np.sum(np.where(d_flat == name, 1, 0)))
    fig, ax = plt.subplots()
    # plt.xticks(rotation=45, fontsize=6)
    plt.xlabel("Classes")  # , fontsize = 60)
    plt.ylabel("Count")  # , fontsize = 60)
    plt.legend()  # prop={'size':45})
    plt.title("Count of each class")
    ax.barh(names, count, align="center")

    fig.savefig("histogram")
    fig.show()


# ----------------------------------------------------------------------------------------------------------------------


def chord_chexpert():
    data = pd.read_csv("data/CheXpert-v1.0-small/train.csv").fillna(0)

    data = data.iloc[:, 5:19]

    conn = np.zeros((14, 14))
    for ex, line in data.iterrows():
        line = line.to_numpy()
        diseases = np.where(line == 1)[0].tolist()
        while diseases:
            disease = diseases.pop()
            for next_disease in diseases:
                conn[disease][next_disease] += 1

    conn = conn / np.sum(conn)
    fig, axes = plot_connectivity_circle(conn, data.columns)
    fig.savefig("chords_chexpert")
    plt.title("Correlation map between diseases in chexnet")
    fig.show()


def histogram_chexpert():
    data = pd.read_csv("data/CheXpert-v1.0-small/train.csv").fillna(0)
    names = data.iloc[:, 5:19].columns
    diseases = data.iloc[:, 5:19].to_numpy()
    counts = np.zeros((3, 14))
    counts[0, :] = np.sum(np.where(diseases == -1, 1, 0), axis=0)
    counts[1, :] = np.sum(np.where(diseases == 0, 1, 0), axis=0)
    counts[2, :] = np.sum(np.where(diseases == 1, 1, 0), axis=0)
    print(counts)

    # plt.xticks(rotation=45, fontsize=6)

    labels = ["-1", "0", "1"]
    data = {"-1": counts[0, :], "0": counts[1, :], "1": counts[2, :]}
    df = pd.DataFrame(data, columns=labels, index=names)
    fig,ax=plt.subplots()
    df.plot.barh(ax=ax)
    plt.xlabel("Classes")  # , fontsize = 60)
    plt.ylabel("Count")  # , fontsize = 60)
    plt.legend()  # prop={'size':45})
    plt.title("Count of each class")
    fig.savefig("histogram_chexpert")
    fig.show()


if __name__ == "__main__":
    # chexnet visualization
    # histogram()
    # chord()

    # chexpert visualization
    histogram_chexpert()
    chord_chexpert()
