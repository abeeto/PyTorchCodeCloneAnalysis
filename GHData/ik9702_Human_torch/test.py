import torch
from torch import nn
import numpy as np
import time 

device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


conv1 = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input_data = torch.randn(1000, 16, 50, 100)
t1_s = time.time()
m1 = conv1(input_data)
t1_e = time.time()
print(m1.size())

input_data = input_data.to(device)

conv2 = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)).to(device)
t2_s = time.time()
m2 = conv2(input_data)
t2_e = time.time()
print(m2.size())


print(f"cpu 속도 : {t1_e-t2_s}초, Cuda 속도 : {t2_e-t2_s}초")