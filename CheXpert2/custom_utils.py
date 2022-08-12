import contextlib
import copy

import numpy as np


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
def set_parameter_requires_grad(model):
    for ex, param in enumerate(model.parameters()):
        param.requires_grad = False


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

