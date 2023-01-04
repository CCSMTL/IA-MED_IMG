import functools

import torch
from torchvision import transforms
from CheXpert2.custom_utils import channels321
import timm
from CheXpert2 import names,hierarchy
import logging


class CNN(torch.nn.Module):
    def __init__(self,
                 backbone_name: str,
                 num_classes: int,
                 channels: int = 3,
                 hierarchical: bool = True,
                 pretrained: bool = True,
                 drop_rate: float = 0,
                 global_pool: str = "avg"
                 ) -> object:
        """

        @rtype: object
        @param backbone_name: The name of the backbone . See timm.list_models() for the complete list
        @param num_classes: The number of class returned as the last layer of the model
        @param channels: The number of channels of the input image. Either 1 or 3
        @param hierarchical: Whether to use the defined hierarchy of the pathologies as described in https://arxiv.org/abs/1911.06475
        @param pretrained: Whether to use the pretrained weights from ImageNet
        @param drop_rate: The drop rate of the linear layer
        @param global_pool: The type of pooling used to merge the features extracted
        """
        super().__init__()






        self.backbone_name  = backbone_name
        self.pretrained     = pretrained
        self.drop_rate      = drop_rate
        self.prob_pool      = True if global_pool == "weighted" else False
        self.global_pool    = "avg" if self.prob_pool else global_pool
        self.channels       = channels
        self.names          = names
        self.num_classes    = num_classes
        self.backbone       = self.get_backbone()

        self.hierarchical   = hierarchical
        if self.hierarchical :
            self.hierarchy = {}
            for parent,children in hierarchy.items() :
                self.hierarchy[names.index(parent)] = [names.index(child) for child in children]

        self.backbone = self.get_backbone()

        self.final_convolution = torch.nn.Conv2d(in_channels = self.backbone.feature_info[-1]["num_chs"],out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc = torch.nn.ModuleList([torch.nn.Linear(self.backbone.feature_info[-1]["num_chs"],1) for i in range(self.num_classes)])
        self.dropout = torch.nn.Dropout(p=self.drop_rate)
    def load_backbone(self, path: str) -> None:
        """
        Load the backbone from a given path
        @param path: The path to the saved model
        """
        self.backbone.load_state_dict(torch.load(path))

    def get_backbone(self):
        if "yolo" in self.backbone_name:
            raise NotImplementedError("The Yolo model is not functionnal for now.")
            backbone = torch.hub.load('ultralytics/yolov5', "_create",
                                      f'{backbone_name}-cls.pt',
                                      device="cpu")  # ,classes=num_classes,channels=channels)

            classifier = list(backbone.named_modules())[-1]
            #
            if channels == 1:
                channels321(backbone)

            setattr(backbone, classifier[0],
                    torch.nn.Linear(classifier[1].in_features, num_classes, bias=True, device="cpu"))
        else:
            assert self.backbone_name in timm.list_models(), print("This model is not supported")
            backbone = timm.create_model(self.backbone_name, pretrained=self.pretrained, in_chans=self.channels,
                                         num_classes=self.num_classes, drop_rate=self.drop_rate, global_pool=self.global_pool)


        return backbone

    def weighted_forward(self, x):
        """
        Forward pass with weighted pooling as described in https://arxiv.org/pdf/2005.14480.pdf

        Args:
            x: The image to forward pass

        Returns: The logits of the image

        """
        logits=torch.zeros((x.shape[0], self.num_classes)).to(x.device)
        features = self.backbone.forward_features(x)
        for i in range(self.num_classes) :
            prob_map = torch.sigmoid(self.final_convolution(features))

            weight_map = prob_map / prob_map.sum(dim=2, keepdim=True) \
                .sum(dim=3, keepdim=True)
            feat = (features * weight_map).sum(dim=2, keepdim=True) \
                .sum(dim=3, keepdim=True)

            feat = feat.flatten(start_dim=1)

            classifier = self.fc[i]
            feat = self.dropout(feat)
            logit = classifier(feat)

            logits[:,i] = logit[:,0]

        return logits

    def forward(self,images):
        outputs = torch.zeros((images.shape[0], self.num_classes)).to(images.device)



        for i in range(0,2) :
            #iterate through the two images for one patient
            image = images[:,i*self.channels:(i+1)*self.channels,:,:]

            #if not torch.round(torch.min(image),decimals=2)==torch.round(torch.mean(image),decimals=2) : #if the image is not empty
            outputs += self.weighted_forward(image) if self.prob_pool else self.backbone(image)




        if not self.training :
            #we apply the sigmoid function if in the inference phase
            outputs = torch.sigmoid(outputs)

            if self.hierarchical :
                # using conditional probability
                for parent_class, children in  self.hierarchy.items() :
                    prob_parent = outputs[:,parent_class]
                    outputs[:,children] = outputs[:,children] * prob_parent[:,None]

                prob_sick = outputs[:, -1]
                outputs[:,0:-1] = outputs[:,0:-1] * prob_sick[:, None]


        return outputs




if __name__ == "__main__":  # for debugging purpose

    for name in ["densenet121", "resnet18","convnext_small"]:
        for channels in [1, 3]:
            x = torch.zeros((4,2*channels, 320, 320))
            for hierarchical in [True, False]:
                for prob_pool in [True, False]:
                    print(name, channels, hierarchical, prob_pool)
                    model = CNN(name, 14, channels, hierarchical, False, 0, "weighted" if prob_pool else "avg")
                    y = model(x)
                    print(y.shape)
