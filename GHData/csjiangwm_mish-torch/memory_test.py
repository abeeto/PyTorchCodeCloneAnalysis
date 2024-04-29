from memory_profiler import profile
import torch
from mish import MemoryEfficientMish, Mish

shape = (100,3,224,224)

@profile
def mish_test_cpu():
    x = torch.randn(*shape, requires_grad=True)

    m = Mish()
    y = m(x)

@profile
def memory_mish_test_cpu():
    x = torch.randn(*shape, requires_grad=True)

    m = MemoryEfficientMish()
    y = m(x)


if __name__ == '__main__':
    mish_test_cpu()
    memory_mish_test_cpu()
    
    