import torch
from custom_utils import set_parameter_requires_grad

class CNN(torch.nn.Module):
    def __init__(self, backbone, num_classes,freeze_backbone=False):
        super().__init__()
        # TODO : VERIFY IMAGE SIZE WITH PRETRAINED MODELS!!
        # self.backbone=torch.hub.load('pytorch/vision:v0.10.0',backbone, pretrained=True)
        # print(torch.hub.list("facebookresearch/deit:main"))

        if backbone in torch.hub.list("pytorch/vision:v0.10.0"):
            repo = "pytorch/vision:v0.10.0"
        elif backbone in torch.hub.list("facebookresearch/deit:main"):
            repo = "facebookresearch/deit:main"
        else:
            pass

        self.backbone = torch.hub.load(
            repo, backbone, pretrained=True
        )


        # -------------------------------------------------------------
        # finds the size of the last layer of the model
        layers = []
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                layers.append(name)
        name = layers[::-1][0].split(".")
        x = getattr(self.backbone, name[0])
        if len(name) > 2:
            size = x[int(name[1])].out_features
            # x[int(name[1])]=torch.nn.Linear(size,num_classes,bias=True)
        else:
            size = x.out_features
            # x = torch.nn.Linear(size, num_classes, bias=True)
        # -------------------------------------------------------------
        # setattr(self.backbone, name[0],x)
        if freeze_backbone:
            set_parameter_requires_grad(self.backbone)
        self.classifier = torch.nn.Linear(size, num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
