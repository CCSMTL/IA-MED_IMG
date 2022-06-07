#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-05-23$

@author: Jonathan Beaulieu-Emond
"""
import os

import pandas as pd
import tqdm
import torch
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.preprocessing import label_binarize
from training.dataloaders.CxrayDataloader import CustomImageDataset
from sklearn.gaussian_process.kernels import RBF


import torchxrayvision as xrv
from models.Unet import Unet

RBF = RBF(1)


def accuracy(true, pred):

    return np.mean(np.where(pred == true, 1, 0))


def f1(true, pred):

    return sklearn.metrics.f1_score(true, pred, average="macro")  # weighted??


metrics = {"f1": f1, "acc": accuracy}


def load_model(backbone, pretrained=True):
    # torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")
    # model = Unet(backbone_name=backbone, pretrained=not pretrained)
    # PATH = f"../models/models_weights/Unet/{backbone}/Unet.pt"
    # if pretrained:
    #    model.load_state_dict(torch.load(PATH))
    repo = "pytorch/vision:v0.10.0"
    # model = torch.hub.load(repo, backbone, pretrained=True)
    model = xrv.models.DenseNet(weights="densenet121-res224-all")

    model.eval()
    try:
        encoder = model.features
    except:
        encoder = model

    return encoder


@torch.no_grad()
def main():
    os.environ["DEBUG"] = "True"
    os.environ["CLUSTERING"] = "True"
    results = pd.DataFrame()
    # 1) load Unet encoder
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = load_model("densenet201", pretrained=False).to(device)
    # 2) load dataset
    train_dataset = CustomImageDataset(
        f"../data/training", num_classes=14, img_size=320  # ?
    )
    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=10, num_workers=8, pin_memory=True
    )

    val_dataset = CustomImageDataset(
        f"../data/validation", num_classes=14, img_size=320
    )

    # rule of thumb : num_worker = 4 * number of gpu ; on windows leave =0
    # batch_size : maximum possible without crashing

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=10, num_workers=0, pin_memory=True
    )
    # 3) Perform clustering

    encodings = torch.tensor([])
    labels = torch.tensor([])

    for image, label in tqdm.tqdm(training_loader):
        image = image.to(device)
        x = encoder(image)
        # x = x.flatten(start_dim=2)
        encodings = torch.cat((x, encodings), dim=0)
        labels = torch.cat((label, labels), dim=0)

    encodings_val = torch.tensor([])
    labels_val = torch.tensor([])
    for image, label in tqdm.tqdm(val_loader):
        x = encoder(image)
        # x = x.flatten(start_dim=2)
        encodings_val = torch.cat((encodings_val, x), dim=0)  # feature 3?
        labels_val = torch.cat((labels_val, label), dim=0)
    # 4) Clustering
    labels = labels.numpy()
    labels_val = labels_val.numpy()
    # ----------------------------------------
    import umap
    from umap.parametric_umap import ParametricUMAP

    reducer = ParametricUMAP()

    # reducer = umap.UMAP()
    # pca = PCA(n_components=2)
    # pca_encodings = pca.fit_transform(encodings.flatten(start_dim=1).numpy())
    # pca_encodings_val = pca.transform(encodings_val.flatten(start_dim=1).numpy())
    pca_encodings = reducer.fit_transform(encodings.flatten(start_dim=1).numpy())
    pca_encodings_val = reducer.transform(encodings_val.flatten(start_dim=1).numpy())
    # ----------------------------------------
    # attempt 1 : SVM (supervised)
    clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    clf.fit(pca_encodings, labels)  # dim3?
    results["SVM"] = clf.predict(pca_encodings_val)
    # ----------------------------------------
    # attempt 2 : hierarchical clustering (unsupervised)
    clustering = AgglomerativeClustering(n_clusters=15, compute_full_tree=True)
    y2 = clustering.fit(pca_encodings).labels_

    # ----------------------------------------
    # attempt 3 : XGBoost / Adaboost
    dtrain = xgb.DMatrix(pca_encodings, label=labels)
    dtest = xgb.DMatrix(pca_encodings_val, label=labels_val)
    parameters = {
        "max_depth": 6,
        "nthread": 8,
        "eval_metric": "auc",
        "objective": "multi:softprob",
        #    "sampling_method": "gradient_based",
        "num_class": 15,
    }
    num_round = 10
    bst = xgb.train(parameters, dtrain, num_round)
    result = bst.predict(dtest)
    print(result)
    results["XGBoost"] = np.argmax(result, axis=1)
    # ----------------------------------------

    # 5) Analysis
    svm, xgboost = {}, {}
    for key, metric in metrics.items():
        svm[key] = metric(labels_val, results["SVM"])
        xgboost[key] = metric(labels_val, results["XGBoost"])

    print(svm, xgboost)

    import plotly.express as go

    # pca_encodings = RBF(pca_encodings)
    fig = go.scatter(x=pca_encodings[:, 0], y=pca_encodings[:, 1], color=labels)
    fig.show()

    fig = go.scatter(x=pca_encodings[:, 0], y=pca_encodings[:, 1], color=y2)
    fig.show()
    return "hello World"


if __name__ == "__main__":
    main()
