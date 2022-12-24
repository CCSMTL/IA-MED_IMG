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


        if "yolo" in backbone_name:
            raise NotImplementedError("The Yolo model is not functionnal for now.")
            backbone = torch.hub.load('ultralytics/yolov5', "_create",
                                      f'{backbone_name}-cls.pt',device="cpu")  # ,classes=num_classes,channels=channels)

            classifier = list(backbone.named_modules())[-1]
            #
            if channels == 1 :
                channels321(backbone)

            setattr(backbone, classifier[0], torch.nn.Linear(classifier[1].in_features, num_classes, bias=True,device="cpu"))
        else :
            assert backbone_name in timm.list_models(),print("This model is not supported")
            backbone = timm.create_model(backbone_name, pretrained=pretrained, in_chans=channels,
                                         num_classes=num_classes,drop_rate=drop_rate,global_pool=global_pool)





        self.channels       = channels
        self.names          = names
        self.num_classes    = num_classes
        self.backbone       = backbone

        self.hierarchical   = hierarchical

        if self.hierarchical :
            self.hierarchy = {}
            for parent,children in hierarchy.items() :
                self.hierarchy[names.index(parent)] = [names.index(child) for child in children]


    def forward(self,images):
        outputs = torch.zeros((images.shape[0], self.num_classes)).to(images.device)



        for i in range(0,2) :
            #iterate through the two images for one patient
            image = images[:,i*self.channels:(i+1)*self.channels,:,:]

            #if not torch.round(torch.min(image),decimals=2)==torch.round(torch.mean(image),decimals=2) : #if the image is not empty
            outputs += self.backbone(image)




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
    x = torch.zeros((2, 2, 320, 320))
    for name in ["densenet121", "resnet18","convnext_small"]:
        cnn = CNN(name, len(names),channels=1)
        y = cnn(x)  # test forward loop
