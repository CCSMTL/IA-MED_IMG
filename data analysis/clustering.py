#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-05-23$

@author: Jonathan Beaulieu-Emond
"""
import os

import torch
from training.dataloaders.cxray_dataloader import CustomImageDataset
from models.Unet import Unet

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_model(backbone) :
    # torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")
    model = Unet(backbone_name=backbone)
    PATH= f"models/models_weights/Unet/{backbone}/Unet.pt"
    if os.environ["DEBUG"]=="False" :
        model.load_state_dict(torch.load(PATH))
    encoder=model.forward_backbone()
    return encoder

def main():


    # 1) load Unet encoder


    encoder=load_model("resnet18")
    # 2) load dataset
    train_dataset = CustomImageDataset(
        f"data/training",
        num_classes=14,
        img_size=320 #?
    )
    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        num_workers=0,
        pin_memory=True,
    )
    # 3) Perform clustering

    encodings=torch.tensor([[],[],[],[]])
    labels=torch.tensor([[],[]])
    for image,label in training_loader :
        x,features=encoder(image)
        encodings=torch.cat((encodings,features))
        labels=torch.cat((labels,label))

    encodings_val = torch.tensor([[], [], [], []])
    labels_val = torch.tensor([[], []])
    for image, label in training_loader:
        x, features = encoder(image)
        encodings_val = torch.cat((encodings_val, features))
        labels_val = torch.cat((labels, label))
    # 4) Clustering
    #pca = PCA(n_components=10)
    pca = PCA(n_components="mle")
    encodings=pca.fit_transform(encodings)


    #attempt 1 : SVM (supervised)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(encodings,torch.argmax(labels))
    y1=clf.predict(encodings_val)
    #attempt 2 : hierarchical clustering (unsupervised)
    clustering = AgglomerativeClustering()
    y2=clustering.fit_predict(encodings)

    #attempt 3 : ?


    # 5) Analysis


    return "hello World"


if __name__ == "__main__":
    main()
