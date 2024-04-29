# coding: utf-8
import collections
import os
import shutil
import tqdm

import numpy as np
import PIL.Image
import torch
import torchvision


# 建议有参数的层和汇合（pooling）层使用torch.nn模块定义，激活函数直接使用torch.nn.functional。
def forward(self, x):
    x = torch.nn.functional.dropout(x, p=0.5, training=self.training)

# model(x)前用model.train()和model.eval()切换网络状态。
# 不需要计算梯度的代码块用with torch.no_grad()包含起来。model.eval()和torch.no_grad()的区别在于，model.eval()是将网络切换为测试状态，例如BN和随机失活（dropout）在训练和测试阶段使用不同的计算方法。
# torch.no_grad()是关闭PyTorch张量的自动求导机制，以减少存储使用和加速计算，得到的结果无法进行loss.backward()。
# torch.nn.CrossEntropyLoss的输入不需要经过Softmax。torch.nn.CrossEntropyLoss等价于torch.nn.functional.log_softmax + torch.nn.NLLLoss。
# loss.backward()前用optimizer.zero_grad()清除累积梯度。optimizer.zero_grad()和model.zero_grad()效果一样。

#PyTorch性能与调试

# torch.utils.data.DataLoader中尽量设置pin_memory=True，对特别小的数据集如MNIST设置pin_memory=False反而更快一些。num_workers的设置需要在实验中找到最快的取值。
# 用del及时删除不用的中间变量，节约GPU存储。
# 使用inplace操作可节约GPU存储，如
x = torch.nn.functional.relu(x, inplace=True)
# 减少CPU和GPU之间的数据传输。例如如果你想知道一个epoch中每个mini-batch的loss和准确率，先将它们累积在GPU中等一个epoch结束之后一起传输回CPU会比每个mini-batch都进行一次GPU到CPU的传输更快。
# 使用半精度浮点数half()会有一定的速度提升，具体效率依赖于GPU型号。需要小心数值精度过低带来的稳定性问题。
# 时常使用assert tensor.size() == (N, D, H, W)作为调试手段，确保张量维度和你设想中一致。
# 除了标记y外，尽量少使用一维张量，使用n*1的二维张量代替，可以避免一些意想不到的一维张量计算结果。
# 统计代码各部分耗时
with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as profile:
    ...
    print(profile)
# 或者在命令行运行

python -m torch.utils.bottleneck main.py
