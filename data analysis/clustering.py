#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-05-23$

@author: Jonathan Beaulieu-Emond
"""
import os
import tqdm
import torch
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from training.dataloaders.cxray_dataloader import CustomImageDataset
from models.Unet import Unet


def load_model(backbone) :
    # torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")
    model = Unet(backbone_name=backbone,pretrained=False)
    PATH= f"../models/models_weights/Unet/{backbone}/Unet.pt"

    model.load_state_dict(torch.load(PATH))
    model.eval()
    encoder=model.forward_backbone
    return encoder
@torch.no_grad()
def main():
    os.environ["DEBUG"]="True"

    # 1) load Unet encoder


    encoder=load_model("resnet18")
    # 2) load dataset
    train_dataset = CustomImageDataset(
        f"../data/training",
        num_classes=14,
        img_size=320 #?
    )
    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        num_workers=8,
        pin_memory=True,
    )

    val_dataset = CustomImageDataset(
        f"../data/validation", num_classes=14, img_size=320
    )

    # rule of thumb : num_worker = 4 * number of gpu ; on windows leave =0
    # batch_size : maximum possible without crashing


    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, num_workers=0, pin_memory=True
    )
    # 3) Perform clustering

    encodings= torch.tensor([])
    labels= torch.tensor([])

    for image,label in tqdm.tqdm(training_loader) :
        x,features=encoder(image)
        x = x.flatten(start_dim=2)
        encodings=torch.cat((x,encodings),dim=0)
        labels=torch.cat((label,labels),dim=0)

    encodings_val =  torch.tensor([])
    labels_val =  torch.tensor([])
    for image, label in tqdm.tqdm(val_loader) :
        x, features = encoder(image)
        x=x.flatten(start_dim=2)
        encodings_val = torch.cat((encodings_val, x),dim=0) #feature 3?
        labels_val = torch.cat((labels_val, label),dim=0)
    # 4) Clustering
    labels=labels.numpy()
    labels_val=labels_val.numpy()

    #pca = PCA(n_components=10)
    pca = PCA(n_components=100)
    encodings=pca.fit_transform(encodings.flatten(start_dim=1).numpy())
    encodings_val=pca.transform(encodings_val.flatten(start_dim=1).numpy())

    #attempt 1 : SVM (supervised)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(encodings,np.argmax(labels==1,axis=1)) #dim3?
    y1=clf.predict(encodings_val)
    #attempt 2 : hierarchical clustering (unsupervised)
    clustering = AgglomerativeClustering(n_clusters=30,compute_full_tree=True)
    y2=clustering.fit_predict(encodings)



    #attempt 3 : XGBoost / Adaboost
    dtrain = xgb.DMatrix(encodings, label=np.argmax(labels,axis=1))
    parameters = {
        "max_depth" : 6,
        "nthread"   : 8,
        "eval_metric" : "auc",
        'objective': 'multi:softprob',
        "sampling_method" : "gradient_based",
        "num_class" : 14
    }
    num_round = 10
    bst = xgb.train(parameters, dtrain, num_round)

    stop=1
    # 5) Analysis


    return "hello World"


if __name__ == "__main__":
    main()
