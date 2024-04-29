import torch.nn as nn
from . import probe


def instrumented_sequential(cls):
    """
    Enable the probe modules injected in an `nn.Sequential` module.

    :param cls: the class to decorate, should be a subclass of `nn.Sequential`
    """
    class InstrumentedSequential(nn.Sequential):
        def __init__(self, *args, **kwargs):
            nn.Sequential.__init__(self)
            wrapped = cls(*args, **kwargs)
            counter = 0
            for _, m in wrapped.named_children():
                if not isinstance(m, probe.ProbeModule):
                    self.add_module(str(counter), m)
                    counter += 1
                else:
                    name = '__probe_{}'.format(m.key)
                    if name in [x for x, _ in self.named_modules()]:
                        raise ValueError('probe key name "{}" already exists'
                                .format(m.key))
                    self.add_module(name, m)
        def forward(self, *args, **kwargs):
            return nn.Sequential.forward(self, *args, **kwargs)
    return InstrumentedSequential


def uninstrumented_sequential(cls):
    """
    Disable the probe modules injected in an `nn.Sequential` module.

    :param cls: the class to decorate, should be a subclass of `nn.Sequential`
    """
    class UninstrumentedSequential(nn.Sequential):
        def __init__(self, *args, **kwargs):
            nn.Sequential.__init__(self)
            wrapped = cls(*args, **kwargs)
            counter = 0
            for _, m in wrapped.named_children():
                if not isinstance(m, probe.ProbeModule):
                    self.add_module(str(counter), m)
                    counter += 1
        def forward(self, *args, **kwargs):
            return nn.Sequential.forward(self, *args, **kwargs)
    return UninstrumentedSequential
