# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# November 2020
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

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from HingeTree import HingeTree, HingeFern, HingeTrie, HingeTreeFusedLinear


class RandomHingeForest(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "init_type" ]

    in_channels: int
    out_channels: int
    depth: int
    extra_outputs: int
    init_type: str

    def __init__(self, in_channels: int, out_channels: int, depth: int, extra_outputs = None, init_type: str = "random"):
        super(RandomHingeForest, self).__init__()

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.init_type = init_type

        thresholds = 6.0*torch.rand([out_channels, 2**depth - 1]) - 3.0

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=in_channels, dtype=torch.long)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=torch.long)
            ordinals -= in_channels * (ordinals // in_channels)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs is None:
            weights = torch.randn([out_channels, 2**depth])
        elif hasattr(extra_outputs, "__iter__"):
            weights = torch.randn([out_channels, 2**depth] + list(extra_outputs))
        else:
            weights = torch.randn([out_channels, 2**depth, extra_outputs])

        HingeTree.fix_thresholds(thresholds, ordinals, weights)

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return HingeTree.apply(x, self.thresholds, self.ordinals, self.weights)

    def reachability(self, x):
        return HingeTree.reachability(x, self.thresholds, self.ordinals, self.weights)

    def leafmap(self, x):
        return HingeTree.leafmap(x, self.thresholds, self.ordinals, self.weights)

    def marginmap(self, x):
        return HingeTree.marginmap(x, self.thresholds, self.ordinals, self.weights)

    def pathmap(self, x):
        # This is nuts! How anyone thinks this trickery is easier than writing for loops in a native language is absolutely bonkers in the head!
        leafMap = HingeTree.leafmap(x, self.thresholds, self.ordinals, self.weights)
        origShape = leafMap.shape
        leafMap = leafMap.view([leafMap.shape[0], leafMap.shape[1], -1])

        batchSize = leafMap.shape[0]
        numTrees = leafMap.shape[1]

        # batchsize x trees x mappings --> batchsize x mappings x trees
        leafMap = leafMap.transpose(1,2)

        outShape = [batchSize, self.depth] + list(leafMap.shape[1:])
        
        pathMap = torch.zeros(outShape, dtype=leafMap.dtype, device=leafMap.device)

        # Add quantity of interior vertices
        leafMap += 2**self.depth - 1

        treeIndices = torch.arange(numTrees).to(leafMap.device)

        for d in range(self.depth):
            # Compute parent nodes
            leafMap -= 1
            leafMap = (leafMap / 2).type(torch.long)

            pathMap[:, self.depth-1-d, :, :] = self.ordinals[treeIndices, leafMap]

        # batchsize x depth x mappings x trees --> batchsize x trees x mappings x depth
        pathMap = pathMap.transpose(1,3)

        outShape = list(origShape) + [self.depth]

        pathMap = pathMap.view(outShape)

        return pathMap
        

    def check_thresholds(self):
        return HingeTree.check_thresholds(self.thresholds.data, self.ordinals.data, self.weights.data)

    def fix_thresholds(self):
        return HingeTree.fix_thresholds(self.thresholds.data, self.ordinals.data, self.weights.data)

    def init_medians(self, x):
        return HingeTree.init_medians(x, self.thresholds.data, self.ordinals.data, self.weights.data)

    def init_greedy(self, x, y):
        return HingeTree.init_greedy(x, y, self.thresholds.data, self.ordinals.data, self.weights.data)

    def load_state_dict(self, *args, **kwargs):
        super(RandomHingeForest, self).load_state_dict(*args, **kwargs)

        # For compatibility
        self.ordinals = self.ordinals.type(torch.long)

class RandomHingeFern(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "init_type" ]

    in_channels: int
    out_channels: int
    depth: int
    extra_outputs: int
    init_type: str

    def __init__(self, in_channels: int, out_channels: int, depth: int, extra_outputs = None, init_type: str = "random"):
        super(RandomHingeFern, self).__init__()

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.init_type = init_type

        thresholds = 6.0*torch.rand([out_channels, depth]) - 3.0

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=in_channels, dtype=torch.long)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=torch.long)
            ordinals -= in_channels * (ordinals // in_channels)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs is None:
            weights = torch.randn([out_channels, 2**depth])
        elif hasattr(extra_outputs, "__iter__"):
            weights = torch.randn([out_channels, 2**depth] + list(extra_outputs))
        else:
            weights = torch.randn([out_channels, 2**depth, extra_outputs])

        HingeFern.fix_thresholds(thresholds, ordinals, weights) # Doesn't do anything for ferns

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return HingeFern.apply(x, self.thresholds, self.ordinals, self.weights)

    def reachability(self, x):
        return HingeFern.reachability(x, self.thresholds, self.ordinals, self.weights)

    def leafmap(self, x):
        return HingeFern.leafmap(x, self.thresholds, self.ordinals, self.weights)

    def marginmap(self, x):
        return HingeFern.marginmap(x, self.thresholds, self.ordinals, self.weights)

    def pathmap(self, x):
        if x.dim() == 2:
            outShape = [x.shape[0], 1, 1]

            pathMap = self.ordinals.repeat(outShape)
        elif x.dim() > 2:
            # How anyone thinks this trickery is easier than writing for loops in a native language is absolutely bonkers in the head!
            outShape = [x.shape[0], x.shape[-1]] + list(x.shape[2:-1]) + [1,1]
            
            # ordinals is trees x depth
            pathMap = self.ordinals.repeat(outShape)

            # batchsize x lastDim x ... x trees x depth --> batchsize x trees x ... x lastDim x depth
            pathMap = pathMap.transpose(1,-2)
        else:
            raise RuntimeError("Input batch should be at least 2 dimensions.")

        return pathMap

    def check_thresholds(self):
        return HingeFern.check_thresholds(self.thresholds.data, self.ordinals.data, self.weights.data)

    def fix_thresholds(self):
        return HingeFern.fix_thresholds(self.thresholds.data, self.ordinals.data, self.weights.data)

    def init_medians(self, x):
        return HingeFern.init_medians(x, self.thresholds.data, self.ordinals.data, self.weights.data)

    def load_state_dict(self, *args, **kwargs):
        super(RandomHingeFern, self).load_state_dict(*args, **kwargs)

        # For compatibility
        self.ordinals = self.ordinals.type(torch.long)

class RandomHingeTrie(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "deterministic", "init_type" ]

    in_channels: int
    out_channels: int
    depth: int
    extra_outputs: int
    deterministic: bool
    init_type: str

    def __init__(self, in_channels: int, out_channels: int, depth: int, extra_outputs = None, init_type: str = "random"):
        super(RandomHingeTrie, self).__init__()

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.deterministic = deterministic
        self.init_type = init_type

        thresholds = 6.0*torch.rand([out_channels, 2**depth - 1]) - 3.0

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=in_channels, dtype=torch.long)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=torch.long)
            ordinals -= in_channels * (ordinals // in_channels)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs is None:
            weights = torch.randn([out_channels, 2**depth - 1])
        elif hasattr(extra_outputs, "__iter__"):
            weights = torch.randn([out_channels, 2**depth - 1] + list(extra_outputs))
        else:
            weights = torch.randn([out_channels, 2**depth - 1, extra_outputs])

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return HingeTrie.apply(x, self.thresholds, self.ordinals, self.weights)

    def init_medians(self, x):
        return HingeTrie.init_medians(x, self.thresholds.data, self.ordinals.data, self.weights.data)

    def load_state_dict(self, *args, **kwargs):
        super(RandomHingeForest, self).load_state_dict(*args, **kwargs)

        # For compatibility
        self.ordinals = self.ordinals.type(torch.long)

class RandomHingeForestFusedLinear(RandomHingeForest):
    def __init__(self, in_channels: int, number_of_trees: int, out_channels: int, depth: int, extra_outputs = None, init_type: str = "random", bias: bool = True):
        super(RandomHingeForestFusedLinear, self).__init__(in_channels, number_of_trees, depth, extra_outputs, init_type)

        self.linear_weights = nn.Parameter(torch.empty([out_channels, number_of_trees]))
        self.linear_bias = nn.Parameter(torch.zeros([out_channels]), requires_grad=bias)

        # Copied from nn.Linear
        init.kaiming_uniform_(self.linear_weights, a=math.sqrt(5))

        if bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.linear_bias, -bound, bound)

    def forward(self, x):
        return HingeTreeFusedLinear.apply(x, self.thresholds, self.ordinals, self.weights, self.linear_weights, self.linear_bias)

