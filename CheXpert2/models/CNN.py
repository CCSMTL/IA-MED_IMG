import functools

import torch
from torchvision import transforms
from CheXpert2.custom_utils import channels321
import timm

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

        self.channels = channels
        # if "yolo" in backbone_name:
        #     backbone = torch.hub.load('ultralytics/yolov5', "_create",
        #                               f'{backbone_name}-cls.pt')  # ,classes=num_classes,channels=channels)
        #     classifier = list(backbone.named_modules())[-1]
        #     # setattr(backbone,classifier[0],torch.nn.Linear(classifier[1].in_features,num_classes,bias=True))
        #     channels321(backbone)
        #     self.classifier = torch.nn.Linear(classifier[1].out_features, num_classes, bias=True)
        # else :
        assert backbone_name in timm.list_models()
        backbone = timm.create_model(backbone_name, pretrained=pretrained, in_chans=channels,
                                     num_classes=num_classes,drop_rate=drop_rate,global_pool=global_pool)



        self.num_classes = num_classes

        self.backbone=backbone

        self.hierarchical = hierarchical


    def forward(self,images):
        outputs = torch.zeros((images.shape[0], self.num_classes)).to(images.device)



        for i in range(0,2) :
            image = images[:,i*self.channels:(i+1)*self.channels,:,:]
            if not torch.round(torch.min(image),decimals=2)==torch.round(torch.mean(image),decimals=2) : #if the image is not empty

                outputs += self.backbone(image)

        if not self.training :
            outputs = torch.sigmoid(outputs)

            if self.hierarchical :
                # using conditional probability

                prob_sick = outputs[:, -1]
                prob_opacity = outputs[:, 0]
                prob_air = outputs[:, 1]
                prob_liquid = outputs[:, 2]

                # probability of opacity children given opacity
                outputs[:, [4, 7, 8, 12, 14]] = outputs[:, [4, 7, 8, 12, 14]] * prob_opacity[:, None]

                # prob of air children given air
                outputs[:, [5, 9, 16]] = outputs[:, [5, 9, 16]] * prob_air[:, None]

                # prob of liquid children given liquid
                outputs[:, [6, 10]] = outputs[:, [6, 10]] * prob_liquid[:, None]

                outputs[:,0:-1] = outputs[:,0:-1] * prob_sick[:, None]


        return outputs




if __name__ == "__main__":  # for debugging purpose
    x = torch.zeros((2, 2, 320, 320))
    for name in ["densenet121", "resnet18"]:
        cnn = CNN(name, 14,channels=1)
        y = cnn(x)  # test forward loop
