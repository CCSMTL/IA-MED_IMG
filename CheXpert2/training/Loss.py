#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-18$

@author: Jonathan Beaulieu-Emond
"""


# Libauc : not working for now
# from libauc.losses import AUCM_MultiLabel
# from libauc.optimizers import PESG
# #config.update({"lr": 0.0001}, allow_val_change=True)
# loss = AUCM_MultiLabel(device=device, num_classes=num_classes,
#                        imratio=np.array(experiment.training_loader.dataset.count).tolist())
# criterion = lambda outputs, preds: loss(torch.sigmoid(outputs), preds)
# results = experiment.train(optimizer=PESG(model,a=loss.a,b=loss.b,alpha=loss.alpha,imratio=np.array(experiment.training_loader.dataset.count).tolist(), device=device,lr=config["lr"],margin=1,weight_decay=config["weight_decay"]) , criterion=criterion,val_criterion=loss)
# criterion = lambda outputs, preds: torch.nn.functional.binary_cross_entropy(torch.sigmoid(outputs), preds)
def main():
    return "hello World"


if __name__ == "__main__":
    main()
