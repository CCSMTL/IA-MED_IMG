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


class Weighted(CNN) :
    def __init__(self,*args,**kwargs) :
        super(Weighted,self).__init__(*args,**kwargs)



        self.final_convolution = torch.nn.Conv2d(
            in_channels=self.backbone.feature_info[-1]["num_chs"],
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fc = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.backbone.feature_info[-1]["num_chs"], 1)
                for i in range(self.num_classes)
            ]
        )
        self.dropout = torch.nn.Dropout(p=self.drop_rate)
    def weighted_forward(self, x):
        """
        Forward pass with weighted pooling as described in https://arxiv.org/pdf/2005.14480.pdf

        Args:
            x: The image to forward pass

        Returns: The logits of the image

        """
        logits = torch.zeros((x.shape[0], self.num_classes)).to(x.device)
        features = self.backbone.forward_features(x)
        for i in range(self.num_classes):
            prob_map = torch.sigmoid(self.final_convolution(features))

            weight_map = prob_map / prob_map.sum(dim=2, keepdim=True).sum(
                dim=3, keepdim=True
            )
            feat = (
                (features * weight_map)
                .sum(dim=2, keepdim=True)
                .sum(dim=3, keepdim=True)
            )

            feat = feat.flatten(start_dim=1)

            classifier = self.fc[i]
            feat = self.dropout(feat)
            logit = classifier(feat)

            logits[:, i] = logit[:, 0]

        return logits

    def forward(self, images):
        outputs = torch.zeros((images.shape[0], self.num_classes)).to(images.device)

        if images.shape[1] == self.channels:
            outputs += self.weighted_forward(images)

        else:
            for i in range(0, 2):
                # iterate through the two images for one patient
                image = images[:, i * self.channels: (i + 1) * self.channels, :, :]

                # if not torch.round(torch.min(image),decimals=2)==torch.round(torch.mean(image),decimals=2) : #if the image is not empty
                outputs += self.weighted_forward(image)



        return outputs

    def reset_classifier(self):
        if self.prob_pool:
            self.fc = torch.nn.ModuleList(
                [
                    torch.nn.Linear(self.backbone.feature_info[-1]["num_chs"], 1)
                    for i in range(self.num_classes)
                ]
            )
        else:
            self.backbone.reset_classifier(
                self.num_classes, self.drop_rate, self.global_pool
            )