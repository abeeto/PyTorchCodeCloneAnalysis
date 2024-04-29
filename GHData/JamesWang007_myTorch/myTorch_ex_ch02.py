# -*- coding: utf-8 -*-
'''
    关于Gradient
    link = http://www.pytorchtutorial.com/pytorch-note2-gradient/

'''

# 在BP的时候，pytorch是将Variable的梯度放在Variable对象中的，我们随时都可以使用Variable.grad得到对应Variable的grad。刚创建Variable的时候，它的grad属性是初始化为0.0的。

import torch
from torch.autograd import Variable


# 需要求导的话，requires_grad=True属性是必须的。
w1 = Variable(torch.Tensor([1.0, 2.0, 3.0]), requires_grad=True) 
w2 = Variable(torch.Tensor([1.0, 2.0, 3.0]), requires_grad=True)

print(w1.grad)
print(w2.grad)


# 从下面这两段代码可以看出，使用d.backward()求Variable的梯度的时候，Variable.grad是累加的即: Variable.grad=Variable.grad+new_grad

d = torch.mean(w1)
d.backward()
w1.grad

d.backward()
w1.grad


# 既然累加的话，那我们如何置零呢？
w1.grad.data.zero_()
w1.grad

'''
通过上面的方法，就可以将grad置零。通过打印出来的信息可以看出，w1.grad其实是Variable。现在可以更清楚的理解一下Variable与Tensor之间的关系，上篇博客已经说过，Variable是Tensor的一个wrapper，那么到底是什么样的wrapper呢？从目前的掌握的知识来看，一个是保存weights的Tensor，一个是保存grad的Variable。Variable的一些运算，实际上就是里面的Tensor的运算。

pytorch中的所有运算都是基于Tensor的，Variable只是一个Wrapper，Variable的计算的实质就是里面的Tensor在计算。Variable默认代表的是里面存储的Tensor（weights）。理解到这，我们就可以对grad进行随意操作了。
'''
learning_rate = 0.1
w1.data.sub_(learning_rate * w1.grad.data) # w1.data是获取保存weights的Tensor
'''
Variable更多是用在feedforward中的，因为feedforward是需要记住各个Tensor之间联系的，这样，才能正确的bp。Tensor不会记录路径。而且，如果使用Variable操作的话，就会造成循环图了（猜测）。
'''

# torch.optim
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# in your training loop:
for i in range(steps):
    optimizer.zero_grad() # zero the gradient buffers, 必须要置零
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step() # Does the update
    
# 注意：torch.optim只用于更新参数，不care梯度的计算


import torch
from torch.autograd import Variable
w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)#需要求导的话，requires_grad=True属性是必须的。
w2 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)

z = w1*w2 + w1 # 第二次BP出现问题就在这，不知道第一次BP之后销毁了啥。
res = torch.mean(z)
res.backward() # 第一次求导没问题
res.backward() # 第二次BP会报错,但使用 retain_variables=True，就好了。
# Trying to backward through the graph second time, but the buffers have already been freed. Please specify retain_variables=True when calling backward for the first time


import torch
import torch.cuda as cuda
from torch.autograd import Variable
w1 = Variable(cuda.FloatTensor(2,3), requires_grad=True)
res = torch.mean(w1[1]) # 只用了variable的第二行参数
res.backward()
print(w1.grad)































































