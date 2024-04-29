import numpy as np
import torch
from datetime import datetime


# IMPORTANT NOTE! Both grad variables and model needs to be on .cuda to make GPU work

# CPU
tensor_cpu = torch.rand(2, 2)

# CPU to GPU
if torch.cuda.is_available():
    tensor_cpu = tensor_cpu.cuda()
    tensor_gpu = torch.rand(2,2).cuda()  # How to create a tensor on gpu without conversion
    print(tensor_gpu)


print(tensor_cpu)


# GPU to CPU
tensor_cpu = tensor_cpu.cpu()


# Computing time comparison (gpu is faster in larger scale operations, cpu is faster on smaller scale)
# Also gpu is effortless at running most computes, while cpu can easily cap to 100%
startTime = datetime.now()
b = torch.ones(4000, 4000).cuda()  # remove/add .cuda()
for _ in range(50000):
    b += b
print("Time elapsed:\n", datetime.now() - startTime)
