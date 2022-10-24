import functools

import torch
from torchvision import transforms
from CheXpert2.custom_utils import channels321


class CNN(torch.nn.Module):
    def __init__(self, backbone_name, num_classes, channels=3, img_size=320, freeze_backbone=False, pretrained=True,
                 pretraining=True,drop_rate=0,global_pool="avg"):
        super().__init__()

        self.channels = channels
        if "yolo" in backbone_name:
            backbone = torch.hub.load('ultralytics/yolov5', "_create",
                                      f'{backbone_name}-cls.pt')  # ,classes=num_classes,channels=channels)
            classifier = list(backbone.named_modules())[-1]
            # setattr(backbone,classifier[0],torch.nn.Linear(classifier[1].in_features,num_classes,bias=True))
            channels321(backbone)
            self.classifier = torch.nn.Linear(classifier[1].out_features, num_classes, bias=True)
        else:
            try:
                import timm
                backbone = timm.create_model(backbone_name, pretrained=pretrained, in_chans=channels,
                                             num_classes=num_classes,drop_rate=drop_rate,global_pool=global_pool)

            except :
                raise NotImplementedError("This model has not been found within the available repos.")

        self.num_classes = num_classes

        self.backbone=backbone

        self.pretrain = pretraining
        self.preprocess = self.get_preprocess(channels)
    @staticmethod
    def get_preprocess(channels):
        """
        Pre-processing for the model . This WILL be applied before inference
        """
        if channels == 1:
            normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        return transforms.Compose(
            [

                transforms.ConvertImageDtype(torch.float32),
                normalize,
            ]
        )

    def forward(self,images):
        outputs = torch.zeros((images.shape[0], self.num_classes)).to(images.device)
        for channel in range(images.shape[1]):
            image = images[:, channel:channel + 1, :, :]
            if self.channels == 3:
                image = image.expand(-1, 3, -1, -1)

            image = self.preprocess(image)
            outputs+=self.backbone(image)

        return outputs




if __name__ == "__main__":  # for debugging purpose
    x = torch.zeros((2, 1, 320, 320))
    for name in ["densenet121", "resnet18"]:
        cnn = CNN(name, 14)
        y = cnn(x)  # test forward loop
