import functools

import torch
from torch.autograd import Variable
from CheXpert2.custom_utils import channels321,Identity
import copy

class CNN(torch.nn.Module):
    def __init__(self, backbone_name, num_classes, channels=3, img_size=320, freeze_backbone=False, pretrained=True,
                 pretraining=True,drop_rate=0,global_pool="avg"):
        super().__init__()
        # if backbone_name in torch.hub.list("pytorch/vision:v0.10.0"):
        #     repo = "pytorch/vision:v0.10.0"
        #     weights = "DEFAULT" if pretrained else None
        #     backbone = torch.hub.load(repo, backbone_name, weights=weights)
        #     backbone = backbone.features
        # else:

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
                self.frontal_feature=copy.copy(backbone.forward_features)
                self.lateral_feature=copy.copy(backbone.forward_features)

                #backbone.forward_head = Identity()
                self.classifier = backbone.forward_head
            except:
                raise NotImplementedError("This model has not been found within the available repos.")

        self.num_classes = num_classes

        self.backbone = backbone
        self.pretrain = pretraining

    def forward(self,frontal=None,lateral=None):
        x,y=0,0
        assert frontal is not None or lateral is not None
        if frontal is not None :
            x = self.frontal_feature(frontal)
        if lateral is not None :
            y = self.lateral_feature(lateral)

        return self.classifier(x+y)



if __name__ == "__main__":  # for debugging purpose
    x = torch.zeros((2, 1, 320, 320))
    for name in ["densenet121", "resnet18"]:
        cnn = CNN(name, 14)
        y = cnn(x)  # test forward loop
