# -*- coding:utf-8 -*-
import torch

print(torch.__version__)  # 1.12.0+cu116

x = torch.rand(5, 3)
print(x)

# tensor([[0.7288, 0.1262, 0.0512],
#         [0.0895, 0.5919, 0.2938],
#         [0.1630, 0.0331, 0.8036],
#         [0.1803, 0.4920, 0.8152],
#         [0.1020, 0.1207, 0.9895]])

print(torch.cuda.is_available())  # True
