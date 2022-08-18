from functools import reduce

import torch
from torch.autograd import Variable


@torch.no_grad()
def get_output(model, x):
    y = model(x)
    if "inception" in model._get_name().lower():
        y = y.logits
    return y.shape[1]


def channels321(backbone):
    try:
        for name, weight1 in backbone.named_parameters():
            break

        name = name[:-7]  # removed the .weight of first conv

        first_layer = reduce(getattr, [backbone] + name.split("."))

        try:
            first_layer = first_layer[0]
        except:
            pass
        bias = True if first_layer.bias is not None else False
        new_first_layer = torch.nn.Conv2d(
            1,
            first_layer.out_channels,
            kernel_size=first_layer.kernel_size,
            stride=first_layer.stride,
            padding=first_layer.padding,
            bias=bias,
        ).requires_grad_()

        new_first_layer.weight[:, :, :, :].data[...].fill_(0)
        new_first_layer.weight[:, :, :, :].data[...] += Variable(
            weight1[:, 1:2, :, :], requires_grad=True
        )
        # change first layer attribute
        name = name.split(".")
        last_item = name.pop()
        item = reduce(getattr, [backbone] + name)  # item is a pointer!
        setattr(item, last_item, new_first_layer)

    except:  # transformers
        name = "patch_embed.proj"
        weight1 = backbone.patch_embed.proj.weight
        first_layer = backbone.patch_embed.proj
        bias = True if first_layer.bias is not None else False
        new_first_layer = torch.nn.Conv2d(
            1,
            first_layer.out_channels,
            kernel_size=first_layer.kernel_size,
            stride=first_layer.stride,
            padding=first_layer.padding,
            bias=bias,
        ).requires_grad_()
        new_first_layer.weight[:, :, :, :].data[...].fill_(0)
        new_first_layer.weight[:, :, :, :].data[...] += Variable(
            weight1[:, 1:2, :, :], requires_grad=True
        )
        name = name.split(".")
        last_item = name.pop()
        item = reduce(getattr, [backbone] + name)  # item is a pointer!
        setattr(item, last_item, new_first_layer)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CNN(torch.nn.Module):
    def __init__(self, backbone_name, num_classes, channels=3, img_size=320, freeze_backbone=False, pretrained=True,
                 pretraining=True):
        super().__init__()
        # if backbone_name in torch.hub.list("pytorch/vision:v0.10.0"):
        #     repo = "pytorch/vision:v0.10.0"
        #     weights = "DEFAULT" if pretrained else None
        #     backbone = torch.hub.load(repo, backbone_name, weights=weights)
        #     backbone = backbone.features
        # else:
        try:
            import timm
            backbone = timm.create_model(backbone_name, pretrained=pretrained, in_chans=channels,
                                         num_classes=num_classes)
            backbone.forward_head = Identity()
        except:
            raise NotImplementedError("This model has not been found within the available repos.")


        self.num_classes=num_classes
        # -------------------------------------------------------------

        # finds the size of the last layer of the model, and name of the first
        # x = torch.zeros((2, channels, img_size, img_size))
        # size = get_output(self.backbone, x)  # dirty way
        #
        # # -------------------------------------------------------------
        #
        # if freeze_backbone:
        #     set_parameter_requires_grad(self.backbone)
        #
        # # --------------------------------------------------------------
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(size, num_classes, bias=True)
        # )

        self.backbone = backbone
        self.pretrain = pretraining

    def forward(self, x):

        x = self.backbone(x)
        return x




if __name__ == "__main__":  # for debugging purpose
    x = torch.zeros((2, 1, 320, 320))
    for name in ["densenet121", "resnet18"]:
        cnn = CNN(name, 14)
        y = cnn(x)  # test forward loop
