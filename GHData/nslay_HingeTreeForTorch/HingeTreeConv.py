# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# May 2021
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import torch
import torch.nn as nn
import HingeTree
from functools import reduce
import operator

def _as_tuple(x, N: int):
    if isinstance(x, int):
        return tuple([x]*N)

    if len(x) != N:
        raise RuntimeError(f"Expected size {N} list/tuple.")

    return tuple(x)

class HingeTreeConv1d(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "kernel_size", "stride", "padding", "dilation", "init_type" ]

    def __init__(self, in_channels: int, out_channels: int, depth: int, kernel_size, stride = 1, padding = 0, dilation = 1, extra_outputs: int = 1, init_type: str = "random"):
        super(HingeTreeConv1d, self).__init__()

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.init_type = init_type

        self.kernel_size = _as_tuple(kernel_size, 1)
        self.stride = _as_tuple(stride, 1)
        self.padding = _as_tuple(padding, 1)
        self.dilation = _as_tuple(dilation, 1)

        thresholds = 6.0*torch.rand([out_channels, in_channels, 2**depth - 1]) - 3.0

        kernelCount = reduce(operator.mul, self.kernel_size)

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=kernelCount, dtype=torch.long)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=torch.long)
            ordinals -= kernelCount * (ordinals // kernelCount)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs > 1:
            weights = torch.randn([out_channels, in_channels, 2**depth, extra_outputs])
        else:
            weights = torch.randn([out_channels, in_channels, 2**depth])

        HingeTree.HingeTree.fix_thresholds(thresholds.view([out_channels*in_channels, -1]), ordinals.view([out_channels*in_channels, -1]), weights.view([out_channels*in_channels, 2**depth, -1]))

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return HingeTree.HingeTreeConv1d.apply(x, self.thresholds, self.ordinals, self.weights, self.kernel_size, self.stride, self.padding, self.dilation)

class HingeTreeConv2d(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "kernel_size", "stride", "padding", "dilation", "init_type" ]

    def __init__(self, in_channels: int, out_channels: int, depth: int, kernel_size, stride = 1, padding = 0, dilation = 1, extra_outputs: int = 1, init_type: str = "random"):
        super(HingeTreeConv2d, self).__init__()

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.init_type = init_type

        self.kernel_size = _as_tuple(kernel_size, 2)
        self.stride = _as_tuple(stride, 2)
        self.padding = _as_tuple(padding, 2)
        self.dilation = _as_tuple(dilation, 2)

        thresholds = 6.0*torch.rand([out_channels, in_channels, 2**depth - 1]) - 3.0

        kernelCount = reduce(operator.mul, self.kernel_size)

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=kernelCount, dtype=torch.long)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=torch.long)
            ordinals -= kernelCount * (ordinals // kernelCount)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs > 1:
            weights = torch.randn([out_channels, in_channels, 2**depth, extra_outputs])
        else:
            weights = torch.randn([out_channels, in_channels, 2**depth])

        HingeTree.HingeTree.fix_thresholds(thresholds.view([out_channels*in_channels, -1]), ordinals.view([out_channels*in_channels, -1]), weights.view([out_channels*in_channels, 2**depth, -1]))

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return HingeTree.HingeTreeConv2d.apply(x, self.thresholds, self.ordinals, self.weights, self.kernel_size, self.stride, self.padding, self.dilation)

class HingeTreeConv3d(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "kernel_size", "stride", "padding", "dilation", "init_type" ]

    def __init__(self, in_channels: int, out_channels: int, depth: int, kernel_size, stride = 1, padding = 0, dilation = 1, extra_outputs: int = 1, init_type: str = "random"):
        super(HingeTreeConv3d, self).__init__()

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.init_type = init_type

        self.kernel_size = _as_tuple(kernel_size, 3)
        self.stride = _as_tuple(stride, 3)
        self.padding = _as_tuple(padding, 3)
        self.dilation = _as_tuple(dilation, 3)

        thresholds = 6.0*torch.rand([out_channels, in_channels, 2**depth - 1]) - 3.0

        kernelCount = reduce(operator.mul, self.kernel_size)

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=kernelCount, dtype=torch.long)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=torch.long)
            ordinals -= kernelCount * (ordinals // kernelCount)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs > 1:
            weights = torch.randn([out_channels, in_channels, 2**depth, extra_outputs])
        else:
            weights = torch.randn([out_channels, in_channels, 2**depth])

        HingeTree.HingeTree.fix_thresholds(thresholds.view([out_channels*in_channels, -1]), ordinals.view([out_channels*in_channels, -1]), weights.view([out_channels*in_channels, 2**depth, -1]))

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return HingeTree.HingeTreeConv3d.apply(x, self.thresholds, self.ordinals, self.weights, self.kernel_size, self.stride, self.padding, self.dilation)

class HingeFernConv1d(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "kernel_size", "stride", "padding", "dilation", "init_type" ]

    def __init__(self, in_channels: int, out_channels: int, depth: int, kernel_size, stride = 1, padding = 0, dilation = 1, extra_outputs: int = 1, init_type: str = "random"):
        super(HingeFernConv1d, self).__init__()

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.init_type = init_type

        self.kernel_size = _as_tuple(kernel_size, 1)
        self.stride = _as_tuple(stride, 1)
        self.padding = _as_tuple(padding, 1)
        self.dilation = _as_tuple(dilation, 1)

        thresholds = 6.0*torch.rand([out_channels, in_channels, depth]) - 3.0

        kernelCount = reduce(operator.mul, self.kernel_size)

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=kernelCount, dtype=torch.long)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=torch.long)
            ordinals -= kernelCount * (ordinals // kernelCount)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs > 1:
            weights = torch.randn([out_channels, in_channels, 2**depth, extra_outputs])
        else:
            weights = torch.randn([out_channels, in_channels, 2**depth])

        HingeTree.HingeTree.fix_thresholds(thresholds.view([out_channels*in_channels, -1]), ordinals.view([out_channels*in_channels, -1]), weights.view([out_channels*in_channels, 2**depth, -1]))

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return HingeTree.HingeFernConv1d.apply(x, self.thresholds, self.ordinals, self.weights, self.kernel_size, self.stride, self.padding, self.dilation)

class HingeFernConv2d(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "kernel_size", "stride", "padding", "dilation", "init_type" ]

    def __init__(self, in_channels: int, out_channels: int, depth: int, kernel_size, stride = 1, padding = 0, dilation = 1, extra_outputs: int = 1, init_type: str = "random"):
        super(HingeFernConv2d, self).__init__()

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.init_type = init_type

        self.kernel_size = _as_tuple(kernel_size, 2)
        self.stride = _as_tuple(stride, 2)
        self.padding = _as_tuple(padding, 2)
        self.dilation = _as_tuple(dilation, 2)

        thresholds = 6.0*torch.rand([out_channels, in_channels, depth]) - 3.0

        kernelCount = reduce(operator.mul, self.kernel_size)

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=kernelCount, dtype=torch.long)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=torch.long)
            ordinals -= kernelCount * (ordinals // kernelCount)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs > 1:
            weights = torch.randn([out_channels, in_channels, 2**depth, extra_outputs])
        else:
            weights = torch.randn([out_channels, in_channels, 2**depth])

        HingeTree.HingeTree.fix_thresholds(thresholds.view([out_channels*in_channels, -1]), ordinals.view([out_channels*in_channels, -1]), weights.view([out_channels*in_channels, 2**depth, -1]))

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return HingeTree.HingeFernConv2d.apply(x, self.thresholds, self.ordinals, self.weights, self.kernel_size, self.stride, self.padding, self.dilation)

class HingeFernConv3d(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "kernel_size", "stride", "padding", "dilation", "init_type" ]

    def __init__(self, in_channels: int, out_channels: int, depth: int, kernel_size, stride = 1, padding = 0, dilation = 1, extra_outputs: int = 1, init_type: str = "random"):
        super(HingeFernConv3d, self).__init__()

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.init_type = init_type

        self.kernel_size = _as_tuple(kernel_size, 3)
        self.stride = _as_tuple(stride, 3)
        self.padding = _as_tuple(padding, 3)
        self.dilation = _as_tuple(dilation, 3)

        thresholds = 6.0*torch.rand([out_channels, in_channels, depth]) - 3.0

        kernelCount = reduce(operator.mul, self.kernel_size)

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=kernelCount, dtype=torch.long)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=torch.long)
            ordinals -= kernelCount * (ordinals // kernelCount)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs > 1:
            weights = torch.randn([out_channels, in_channels, 2**depth, extra_outputs])
        else:
            weights = torch.randn([out_channels, in_channels, 2**depth])

        HingeTree.HingeTree.fix_thresholds(thresholds.view([out_channels*in_channels, -1]), ordinals.view([out_channels*in_channels, -1]), weights.view([out_channels*in_channels, 2**depth, -1]))

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return HingeTree.HingeFernConv3d.apply(x, self.thresholds, self.ordinals, self.weights, self.kernel_size, self.stride, self.padding, self.dilation)

