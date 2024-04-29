import torch
from BigMM.mm import BIGmm
import time

print("test begin")

time1 = time.time()
a = torch.ones([40000, 40000]).float()
time2 = time.time()
print(time2 - time1)

b = BIGmm(a, a, [10000, 20000], gpu_ids=[0, 1])
time3 = time.time()
print("time is:", time3 - time2)

# c = torch.mm(a.cuda(0), a.cuda(0)).cpu()  # 子模块 3.3 s
# time4 = time.time()
#
# print(time4 - time3)

# d = torch.sum(torch.abs(c - b) > 0.01)
# print(d)
