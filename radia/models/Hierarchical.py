#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-01-23$

@author: Jonathan Beaulieu-Emond
"""
import torch
from radia import names, hierarchy
from radia.models.CNN import CNN
import logging

for key in hierarchy.keys():
    if key not in names:
        names.insert(0, key)

class Hierarchical(CNN):
    def __init__(self,*args,**kwargs) :
        super(Hierarchical,self).__init__(*args,**kwargs)



        self.hierarchy = {}
        for parent, children in hierarchy.items():
            self.hierarchy[names.index(parent)] = [
                names.index(child) for child in children
            ]

    def forward(self, images):
        outputs = torch.zeros((images.shape[0], self.num_classes)).to(images.device)

        if images.shape[1] == self.channels:
            outputs += self.backbone(images)

        else:
            for i in range(0, 2):
                # iterate through the two images for one patient
                image = images[:, i * self.channels: (i + 1) * self.channels, :, :]

                # if not torch.round(torch.min(image),decimals=2)==torch.round(torch.mean(image),decimals=2) : #if the image is not empty
                outputs += self.backbone(image)


        if not self.training:
            # we apply the sigmoid function if in the inference phase
            # outputs = torch.sigmoid(outputs)


            # using conditional probability
            for parent_class, children in self.hierarchy.items():
                prob_parent = outputs[:, parent_class]
                outputs[:, children] = outputs[:, children] * prob_parent[:, None]

            # prob_sick = outputs[:, -1]
            # outputs[:,0:-1] = outputs[:,0:-1] * prob_sick[:, None]

        return outputs


