import contextlib
import copy

import numpy as np
import torch


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


def Myloss(x,y) :
    x = torch.sigmoid(x)
    x2 = torch.zeros_like(x,device=x.device).requires_grad_(True)
    x[:, 1] = torch.mul(x2[:, 1], x[:, 0])
    x[:, [3, 4, 5, 7]] = torch.mul(x[:, [3, 4, 5, 7]], x2[:, 2][:, None])
    x[:, 6] = torch.mul(x2[:, 2], torch.mul(x2[:, 5], x[:, 6]))
    loss = torch.mean((x - y)**2)
    return loss