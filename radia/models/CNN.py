import functools

import torch
from torchvision import transforms
from radia.custom_utils import channels321
import timm
from radia import names, hierarchy
import logging

for key in hierarchy.keys():
    if key not in names:
        names.insert(0, key)


class CNN(torch.nn.Module):
    def __init__(
            self,
            backbone_name: str,
            num_classes: int,
            channels: int = 3,

            pretrained: bool = True,
            drop_rate: float = 0,
            global_pool: str = "avg",
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

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.drop_rate = drop_rate

        self.global_pool = "avg"
        self.channels = channels
        self.names = names
        self.num_classes = num_classes
        self.backbone = self.get_backbone()

        self.backbone = self.get_backbone()

    def get_backbone(self):
        if "yolo" in self.backbone_name:
            raise NotImplementedError("The Yolo model is not functionnal for now.")
            backbone = torch.hub.load(
                "ultralytics/yolov5", "_create", f"{backbone_name}-cls.pt", device="cpu"
            )  # ,classes=num_classes,channels=channels)

            classifier = list(backbone.named_modules())[-1]
            #
            if channels == 1:
                channels321(backbone)

            setattr(
                backbone,
                classifier[0],
                torch.nn.Linear(
                    classifier[1].in_features, num_classes, bias=True, device="cpu"
                ),
            )
        else:
            assert self.backbone_name in timm.list_models(), print(
                "This model is not supported"
            )
            backbone = timm.create_model(
                self.backbone_name,
                pretrained=self.pretrained,
                in_chans=self.channels,
                num_classes=self.num_classes,
                drop_rate=self.drop_rate,
                global_pool=self.global_pool,
            )

        return backbone



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

        return outputs

    def reset_classifier(self):

        self.backbone.reset_classifier(
            self.num_classes, self.drop_rate, self.global_pool
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
