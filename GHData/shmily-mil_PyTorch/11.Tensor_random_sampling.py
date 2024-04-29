'''

随机抽样:
1.定义随机种子:torch.manmul_seed(seed),保证随机抽样的数值不变
2.定义随机数满足的分布:torch.normal()

'''

import torch

torch.manual_seed(1)
mean = torch.rand(1,2)
std = torch.rand(1,2)
print(torch.normal(mean, std))