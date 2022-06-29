#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-28$

@author: Jonathan Beaulieu-Emond
"""
import itertools

import matplotlib
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
    data = pd.read_csv("../data/data_table.csv")

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
    fig.show()


def histogram():
    data = pd.read_csv("../data/data_table.csv")

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

    ax.barh(names, count, align="center")

    fig.savefig("histogram")
    fig.show()


if __name__ == "__main__":
    histogram()
    chord()
