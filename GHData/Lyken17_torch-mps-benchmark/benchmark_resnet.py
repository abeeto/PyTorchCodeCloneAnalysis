import time
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

net = models.resnet50()
data = torch.randn(1, 3, 224, 224)


def benchmark_cpu(net, data, device="cpu", repeats=200, backward=False):
    net = net.to(device)
    data = data.to(device)

    for i in range(5):
        net(data)

    start = time.time()
    for i in range(repeats):
        out = net(data)
        if backward:
            out.sum().backward()

    end = time.time() 
    print(f"({device}, {data.shape}, {backward})  Latency {(end - start) / repeats * 1000:.1f}ms")

for backward in (False, True):
    for bs in (1, 2, 4, 8):
        data = torch.randn(bs, 3, 224, 224)
        benchmark_cpu(net, data, device="cpu", backward=backward)
        benchmark_cpu(net, data, device="mps", backward=backward)