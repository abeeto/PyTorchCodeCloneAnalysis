#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:35:07 2018

@author: jangqh
"""
import numpy as np
import torch
from torch import nn
from torch.nn import init

print "定义一个Sequential模型..."
net1 = nn.Sequential(
        nn.Linear(30, 40),
        nn.ReLU(),
        nn.Linear(40, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
        )


print "访问第一层的参数..."
w1 = net1[0].weight.data
print w1
b1 = net1[0].bias
print b1

print """
注意，这是一个 Parameter，也就是一个特殊的 Variable，我们可以访问其 .data属性得到其中的数据，然后直接定义一个新的 Tensor 对其进行替换，我们可以使用 PyTorch 中的一些随机数据生成的方式，比如 torch.randn，如果要使用更多 PyTorch 中没有的随机化方式，可以使用 numpy
"""
print "定义一个tensor直接对其进行替换...."
net1[0].weight.data = torch.from_numpy(np.random.uniform(3,5, size = (40,30)))
print net1[0].weight.data
          
print"""
可以看到这个参数的值已经被改变了，也就是说已经被定义成了我们需要的初始化方式，如果模型中某一层需要我们手动去修改，那么我们可以直接用这种方式去访问，但是更多的时候是模型中相同类型的层都需要初始化成相同的方式，这个时候一种更高效的方式是使用循环去访问，比如
"""


for layer in net1:
    if isinstance(layer, nn.Linear):###判断是否是线性层
        param_shape = layer.weight.size()       #(40L, 30L)
                                                #(50L, 40L)
                                                #(10L, 50L)
        layer.weight.data = torch.from_numpy(np.random.normal(0,0.5, size = param_shape))
                ###定义均值为0，方差为0.5的正态分布
print "net1[0].weight:\n", net1[0].weight
           

           
print """
对于 Module 的参数初始化，其实也非常简单，如果想对其中的某层进行初始化，可以直接像 Sequential 一样对其 Tensor 进行重新定义，其唯一不同的地方在于，如果要用循环的方式访问，需要介绍两个属性，children 和 modules，下面我们举例来说明
"""

class sim_net(nn.Module):
    def __init__(self):
        super(sim_net, self).__init__()
        self.l1 = nn.Sequential(
                nn.Linear(30, 40),
                nn.ReLU()
                )
        
        self.l1[0].weight.data = torch.randn(30, 40)  #直接对某一层初始化
        
        self.l2 = nn.Sequential(
                nn.Linear(40, 50),
                nn.ReLU()
                )
        
        self.l3 = nn.Sequential(
                nn.Linear(50, 10),
                nn.ReLU()
                )
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
net2  = sim_net()

print "访问children"
for i in net2.children():
    print i
    
print "访问modules"
for i in net2.modules():
    print i


print """children 只会访问到模型定义中的第一层，因为上面的模型中定义了三个 Sequential，所以只会访问到三个 Sequential，而 modules 会访问到最后的结构，比如上面的例子，modules 不仅访问到了 Sequential，也访问到了 Sequential 里面，这就对我们做初始化非常方便，比如
"""

"""
######torch.nn.init

"""

print(net1[0].weight)

print """

一种非常流行的初始化方式叫 Xavier，方法来源于 2010 年的一篇论文 Understanding the difficulty of training deep feedforward neural networks，其通过数学的推到，证明了这种初始化方式可以使得每一层的输出方差是尽可能相等的，有兴趣的同学可以去看看论文

我们给出这种初始化的公式

"""
###Xavier  初始化方向
init.xavier_uniform(net1[0].weight)
print net1[0].weight







