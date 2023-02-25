# ------python import------------------------------------
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from six.moves import urllib
from sklearn import metrics

# -------- local import ---------------------------

from radia.dataloaders.CXRLoader import CXRLoader
from radia.models.CNN import CNN
from radia.Metrics import Metrics
from radia import names
import tqdm

# -------- proxy config ---------------------------
def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue

        own_state[name].copy_(param)


def load_model(weights, models):
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    for model, weight in zip(models, weights):
        state_dict = torch.load(weight, map_location=torch.device(device))

        # model.load_state_dict(state_dict)
        load_my_state_dict(model, state_dict)
        model.eval()
        model = model.to(device)
    return models


@torch.no_grad()
def infer_loop(model, loader, criterion, device):
    """

    :param model: model to evaluate
    :param loader: dataset loader
    :param criterion: criterion to evaluate the loss
    :param device: device to do the computation on
    :return: val_loss for the N epoch, tensor of concatenated labels and predictions
    """
    running_loss = 0
    results = [torch.tensor([]), torch.tensor([])]

    for inputs, labels, idx in tqdm.tqdm(loader):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = (
            inputs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        # inputs,labels = loader.dataset.advanced_transform((inputs, labels))
        # forward + backward + optimize

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        outputs = torch.sigmoid(outputs)
        running_loss += loss.detach()

        if inputs.shape != labels.shape:  # prevent storing images if training unets
            results[1] = torch.cat((results[1], outputs.detach().cpu()), dim=0)
            results[0] = torch.cat((results[0], labels.cpu()), dim=0)

        del (
            inputs,
            labels,
            outputs,
            loss,
        )  # garbage management sometimes fails with cuda

    return running_loss, results
