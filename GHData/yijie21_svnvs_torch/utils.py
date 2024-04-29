import torch
from datetime import datetime
import random
import numpy as np
import os


def tensor_to_device(x, device):
    """Cast a hierarchical object to pytorch device"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        for k in list(x.keys()):
            x[k] = tensor_to_device(x[k], device)
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        return type(x)(tensor_to_device(t, device) for t in x)
    else:
        raise ValueError('Wrong type !')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHED'] = str(seed)


class ClassProperty(property):
    """For dynamically obtaining system time"""
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class Notify(object):
    """Colorful printing prefix.
    A quick example:
    print(Notify.INFO, YOUR TEXT, Notify.ENDC)
    """

    def __init__(self):
        pass

    @ClassProperty
    def HEADER(cls):
        return str(datetime.now()) + ': \033[95m'

    @ClassProperty
    def INFO(cls):
        return str(datetime.now()) + ': \033[92mI'

    @ClassProperty
    def OKBLUE(cls):
        return str(datetime.now()) + ': \033[94m'

    @ClassProperty
    def WARNING(cls):
        return str(datetime.now()) + ': \033[93mW'

    @ClassProperty
    def FAIL(cls):
        return str(datetime.now()) + ': \033[91mF'

    @ClassProperty
    def BOLD(cls):
        return str(datetime.now()) + ': \033[1mB'

    @ClassProperty
    def UNDERLINE(cls):
        return str(datetime.now()) + ': \033[4mU'
    ENDC = '\033[0m'