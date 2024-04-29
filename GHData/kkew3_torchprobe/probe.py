from __future__ import print_function
import torch.nn as nn


class ProbeModule(nn.Module):
    """
    The super class of the probe modules
    """
    def __init__(self, key):
        """
        :param key: the unique name of the probe
        """
        nn.Module.__init__(self)
        if '.' in key:
            raise ValueError('`.` must not appear in key ("{}")'.format(key))
        self.key = key

    def forward(self, x):
        self.do_probe(x.detach())
        return x

    def do_probe(self, x):
        raise NotImplementedError()


class UnauthorizedProbeAccessError(BaseException): pass

class SentinelProbeModule(ProbeModule):
    """
    The super class of the probe module with sentinel. For this type of probe,
    the method `authorize_pass` must be called before every pass, i.e. the
    implicit or explicit call of `forward` method; otherwise
    `UnauthorizedProbeAccessError` will be raised. This is used to avoid
    unintentional recording action of the underlying probe module.
    """
    def __init__(self, key):
        ProbeModule.__init__(self, key)
        self.sentinel = False

    def forward(self, x):
        if not self.sentinel:
            raise UnauthorizedProbeAccessError()
        y = ProbeModule.forward(self, x)
        self.sentinel = False
        return y

    def authorize_pass(self):
        self.sentinel = True


def _is_sentinel_leaf(module):
    return isinstance(module, SentinelProbeModule) and \
            not len(list(module.children()))

def authorize_all(module):
    """
    Call `authorize_pass` on each SentinelProbeModule instances in
    `torch.nn.Module` module.
    """
    for sprobe in filter(_is_sentinel_leaf, module.modules()):
        sprobe.authorize_pass()


# An example subclass
class SizeProbe(ProbeModule):
    """
    Inspect the size of upstream data.
    """
    def __init__(self, key, echo=True, out=None):
        """
        :param key: the unique name of the probe
        :param echo: False or None to suppress printing to stdout; True or
               a callable object (the print function to invoke instead of
               the native `print`) to print the size message
        :param out: the container to hold the size info, expecting a dict
               or similar type, where the key would be `key` and value
               the size info, i.e. a list of tuples
        """
        ProbeModule.__init__(self, key)
        self.echo = print if echo is True else echo
        self.out = out

    def do_probe(self, x):
        size_info = x.shape if hasattr(x, 'shape') else None
        if self.echo:
            self.echo('{}: {}'.format(self.key, size_info))
        if self.out is not None:
            self.out[self.key] = size_info
