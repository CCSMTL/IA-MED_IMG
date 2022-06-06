import torch
from custom_utils import set_parameter_requires_grad
from torch.autograd import Variable

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

        backbone = torch.hub.load(
            repo, backbone, pretrained=True
        )

        for name, weight1 in backbone.named_parameters():
            break

        name=name[:-7] #removed the .weight of first conv
        first_layer= getattr(backbone,name)
        try :
            first_layer=first_layer[0]
        except :
            pass


        new_first_layer = torch.nn.Conv2d(1, first_layer.out_channels, kernel_size=first_layer.kernel_size, stride=first_layer.stride,
                                          padding=first_layer.padding, bias=first_layer.bias).requires_grad_()

        new_first_layer.weight[:, :, :, :].data[...] = Variable(weight1[:, 1:2, :, :], requires_grad=True)
        setattr(backbone,name,new_first_layer)
        self.backbone = backbone

        # -------------------------------------------------------------

        # finds the size of the last layer of the model, and name of the first
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


        if freeze_backbone:
            set_parameter_requires_grad(self.backbone)

        #--------------------------------------------------------------
        self.classifier = torch.nn.Sequential(torch.nn.Linear(size, num_classes, bias=True))

    def forward(self, x):

        x = self.backbone(x)
        x = self.classifier(x)
        return x
