#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-28$

@author: Jonathan Beaulieu-Emond
"""
import os
#import itertools
import yaml
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd
from mne_connectivity.viz import plot_connectivity_circle
from CheXpert2.dataloaders.MongoDB import MongoDB


os.environ["DEBUG"]="False"
with open("data/data.yaml", "r") as stream:
    names = yaml.safe_load(stream)["names"]

def chord_chexpert():
    data = MongoDB("10.128.107.212", 27017, []).dataset("Valid",[])
    data[['Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pleural Other']] = 0
    data['Normal'] = data["No Finding"]
    data = data[names].astype(int).replace(-1,1)
    n = len(names)
    conn = np.zeros((n, n))
    for ex, line in data.iterrows():
        line = line.to_numpy()
        diseases = np.where(line == 1)[0]

        for disease in diseases :
            conn[disease,:]+=line
            conn[:,disease] +=line


    np.fill_diagonal(conn,0)
    #conn = conn / np.sum(conn) #normalize
    fig, axes = plot_connectivity_circle(conn, data.columns)
    fig.savefig("chords_mongodb_ciusss_valid")
    #plt.title("Correlation map between diseases in chexnet")
    #fig.show()


def histogram_chexpert():
    data = MongoDB("10.128.107.212", 27017, []).dataset("Valid", [])
    data[['Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pleural Other']]=0
    data['Normal'] = data["No Finding"]
    data = data[names].astype(int).to_numpy()
    n = len(names)
    counts = np.zeros((3, n))
    counts[0, :] = np.sum(np.where(data == -1, 1, 0), axis=0)
    counts[1, :] = np.sum(np.where(data == 0, 1, 0), axis=0)
    counts[2, :] = np.sum(np.where(data == 1, 1, 0), axis=0)


    # plt.xticks(rotation=45, fontsize=6)
    import plotly.express as px
    labels = ["-1", "1"]
    data = {"-1": counts[0, :], "1": counts[2, :]}
    df = pd.DataFrame(data, columns=labels, index=names)
    #fig,ax=plt.subplots()
    fig = px.bar(df, x=names, y=labels)
    fig.show()
    fig.write_image("histogram_mongodb_ciusss_valid.png")



if __name__ == "__main__":

    chord_chexpert()
    histogram_chexpert()
