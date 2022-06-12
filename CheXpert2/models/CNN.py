import torch
from CheXpert2.custom_utils import set_parameter_requires_grad
from torch.autograd import Variable
from functools import reduce


@torch.no_grad()
def get_output(model, x):
    y = model(x)
    return y.shape[1]


def channels321(backbone):
    for name, weight1 in backbone.named_parameters():
        break

    name = name[:-7]  # removed the .weight of first conv

    first_layer = reduce(getattr, [backbone] + name.split("."))

    try:
        first_layer = first_layer[0]
    except:
        pass

    new_first_layer = torch.nn.Conv2d(
        1,
        first_layer.out_channels,
        kernel_size=first_layer.kernel_size,
        stride=first_layer.stride,
        padding=first_layer.padding,
        bias=first_layer.bias,
    ).requires_grad_()

    new_first_layer.weight[:, :, :, :].data[...] = Variable(
        weight1[:, 1:2, :, :], requires_grad=True
    )
    # change first layer attribute
    name = name.split(".")
    last_item = name.pop()
    item = reduce(getattr, [backbone] + name)  # item is a pointer!
    setattr(item, last_item, new_first_layer)


class CNN(torch.nn.Module):
    def __init__(self, backbone_name, num_classes, channels=3, freeze_backbone=False):
        super().__init__()
        # TODO : VERIFY IMAGE SIZE WITH PRETRAINED MODELS!!
        # self.backbone=torch.hub.load('pytorch/vision:v0.10.0',backbone, pretrained=True)
        # print(torch.hub.list("facebookresearch/deit:main"))

        if backbone_name in torch.hub.list("pytorch/vision:v0.10.0"):
            repo = "pytorch/vision:v0.10.0"
        elif backbone_name in torch.hub.list("facebookresearch/deit:main"):
            repo = "facebookresearch/deit:main"
        else:
            pass

        backbone = torch.hub.load(repo, backbone_name, pretrained=True)
        if backbone_name.startswith("inception"):  # rip hardcode forced...
            backbone.transform_input = False

        if channels == 1:
            channels321(backbone)

        self.backbone = backbone

        # -------------------------------------------------------------

        # finds the size of the last layer of the model, and name of the first
        x = torch.zeros((2, channels, 320, 320))
        size = get_output(self.backbone, x)  # dirty way

        # -------------------------------------------------------------

        if freeze_backbone:
            set_parameter_requires_grad(self.backbone)

        # --------------------------------------------------------------
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(size, num_classes, bias=True)
        )

    def forward(self, x):

        x = self.backbone(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":  # for debugging purpose
    x = torch.zeros((2, 1, 320, 320))
    for name in ["densenet121", "resnet18", "inception_v3"]:
        cnn = CNN(name, 14)
        y = cnn(x)  # test forward loop
