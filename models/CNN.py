import torch
import torchvision
from custom_utils import set_parameter_requires_grad

class CNN(torch.nn.Module) :
    def __init__(self,backbone,num_classes):
        super().__init__()
        #TODO : VERIFY IMAGE SIZE WITH PRETRAINED MODELS!!
        self.backbone=torch.hub.load('pytorch/vision:v0.10.0',backbone, pretrained=True)

        layers=[]
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                layers.append(name)
        name=layers[::-1][0].split(".")
        x = getattr(self.backbone, name[0])
        if len(name)>2 :
            size=x[int(name[1])].out_features
            #x[int(name[1])]=torch.nn.Linear(size,num_classes,bias=True)
        else :
            size = x.out_features
            #x = torch.nn.Linear(size, num_classes, bias=True)


        #setattr(self.backbone, name[0],x)
        self.classifier=torch.nn.Linear(size,14,bias=True)

    def forward(self,x):
        x=self.backbone(x)
        x=self.classifier(x)
        return x

