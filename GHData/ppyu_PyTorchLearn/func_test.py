# -*- coding: utf-8 -*-
"""
@File   : func_test.py
@Author : Pengy
@Date   : 2020/10/13
@Description : Input your description here ... 
"""
import torch
from transformers import AutoConfig

a = torch.randn(3, 4)

a.split([1, 2], dim=0)
# 把维度0按照长度[1,2]拆分，形成2个tensor，
# shape（1，4）和shape（2，4）

a.split([2, 2], dim=1)
# 把维度1按照长度[2,2]拆分，形成2个tensor，
# shape（3，2）和shape（3，2）

b = a.split(1, dim=-1)
print(b[0].size(), b[1].size(),b[2].size(), b[3].size())

c= b[0].squeeze(-1)
print(c.size())


# 检查GPU环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(n_gpu)

config = AutoConfig.from_pretrained("bert-base-uncased")
print(config.num_labels)