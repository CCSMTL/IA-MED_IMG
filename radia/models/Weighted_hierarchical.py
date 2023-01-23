import functools

import torch
from torchvision import transforms
from radia.custom_utils import channels321
from radia.models.CNN import CNN
import timm
from radia import names, hierarchy
import logging

for key in hierarchy.keys():
    if key not in names:
        names.insert(0, key)


class Weighted_hierarchical(CNN):
    """
       This is a super class of the CNN model implemented before. It overrides
       the forward and reset classifier method to implement the weighted pooling
       as described in https://arxiv.org/pdf/2005.14480.pdf . It also implement the conditional
       learning process as described in https://arxiv.org/abs/1911.06475
       """

    def __init__(self, *args, **kwargs):
        super(Weighted_hierarchical, self).__init__(*args, **kwargs)

        self.hierarchical = True

        self.hierarchy = {}
        for parent, children in hierarchy.items():
            self.hierarchy[names.index(parent)] = [
                names.index(child) for child in children
            ]



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
            outputs +=self.weighted_forward(images)

        else:
            for i in range(0, 2):
                # iterate through the two images for one patient
                image = images[:, i * self.channels : (i + 1) * self.channels, :, :]

                # if not torch.round(torch.min(image),decimals=2)==torch.round(torch.mean(image),decimals=2) : #if the image is not empty
                outputs += self.weighted_forward(image)


        if not self.training:
            # we apply the sigmoid function if in the inference phase
            # outputs = torch.sigmoid(outputs)

            if self.hierarchical:
                # using conditional probability
                for parent_class, children in self.hierarchy.items():
                    prob_parent = outputs[:, parent_class]
                    outputs[:, children] = outputs[:, children] * prob_parent[:, None]

                # prob_sick = outputs[:, -1]
                # outputs[:,0:-1] = outputs[:,0:-1] * prob_sick[:, None]

        return outputs

    def reset_classifier(self):

        self.fc = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.backbone.feature_info[-1]["num_chs"], 1)
                for i in range(self.num_classes)
            ]
        )


if __name__ == "__main__":  # for debugging purpose

    for name in ["densenet121", "resnet18", "convnext_small"]:
        for channels in [1, 3]:
            x = torch.zeros((4, 2 * channels, 320, 320))
            for hierarchical in [True, False]:
                for prob_pool in [True, False]:
                    print(name, channels, hierarchical, prob_pool)
                    model = CNN(
                        name,
                        14,
                        channels,
                        hierarchical,
                        False,
                        0,
                        "weighted" if prob_pool else "avg",
                    )
                    y = model(x)
                    print(y.shape)
