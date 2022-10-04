import contextlib
import copy
import functools
import numpy as np
import torch
from torch.autograd import Variable

# -----------------------------------------------------------------------------------

def convert(array1):
    array = copy.copy(array1)
    answers = []
    array = array.numpy().round(0)
    for item in array:
        if np.max(item) == 0:
            answers.append(13)
        else:
            answers.append(np.argmax(item))
    return answers


# -----------------------------------------------------------------------------------
def set_parameter_requires_grad(model, range=None):
    if range:
        range = min(len(list(model.children())), range)
        for child in list(model.children())[::-range]:
            try:  # TODO : remove try-except
                for param in child:
                    param.requires_grad = True
            except:
                pass
    else:

        for child in list(model.children())[::-1]:
            try:
                for param in child:
                    param.requires_grad = False
            except:
                pass


# -----------------------------------------------------------------------------------


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# -----------------------------------------------------------------------------------
@contextlib.contextmanager
def dummy_context_mgr():
    yield None


#----------------------------------------------------------------------------------
def channels321(backbone):

    for name, weight1 in backbone.named_parameters():
        break

    name = name[:-7]  # removed the .weight of first conv

    first_layer = functools.reduce(getattr, [backbone] + name.split("."))

    # try:
    #     first_layer = first_layer[0]
    # except:
    #     pass
    bias = True if first_layer.bias is not None else False
    new_first_layer = torch.nn.Conv2d(
        1,
        first_layer.out_channels,
        kernel_size=first_layer.kernel_size,
        stride=first_layer.stride,
        padding=first_layer.padding,
        bias=bias,
        device=backbone.device
    ).requires_grad_()

    new_first_layer.weight[:, :, :, :].data[...].fill_(0)
    new_first_layer.weight[:, :, :, :].data[...] += Variable(
        weight1[:, 1:2, :, :], requires_grad=True
    )
    # change first layer attribute
    name = name.split(".")
    last_item = name.pop()
    item = functools.reduce(getattr, [backbone] + name)  # item is a pointer!
    setattr(item, last_item, new_first_layer)





#-------------------------------------------------------------------------------------------

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x