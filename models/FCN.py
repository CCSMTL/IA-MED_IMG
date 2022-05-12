import torch
import torchvision


class FCN(torch.nn.Module) :
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
            size=x[int(name[1])].in_features
            x[int(name[1])]=torch.nn.Linear(size,num_classes,bias=True)
        else :
            size = x.in_features
            x = torch.nn.Linear(size, num_classes, bias=True)
        setattr(self.backbone, name[0],x)

    def forward(self,x):

        return self.backbone(x)