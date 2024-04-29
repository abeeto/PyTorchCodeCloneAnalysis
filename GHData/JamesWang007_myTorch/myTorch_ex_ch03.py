# -*- coding: utf-8 -*-
'''
    自动求导
    link : http://www.pytorchtutorial.com/pytorch-note3-autograd/
'''


#  Backward过程中排除子图
'''      
    pytorch的BP过程是由一个函数决定的，loss.backward()， 可以看到backward()函数里并没有传要求谁的梯度。那么我们可以大胆猜测，在BP的过程中，pytorch是将所有影响loss的Variable都求了一次梯度。但是有时候，我们并不想求所有Variable的梯度。那就要考虑如何在Backward过程中排除子图（ie.排除没必要的梯度计算）。
    
    如何BP过程中排除子图？ Variable的两个参数（requires_grad和volatile）
'''
import torch
from torch.autograd import Variable
x = Variable(torch.randn(5,5))
y = Variable(torch.randn(5,5))
z = Variable(torch.randn(5,5), requires_grad=True)
a = x + y # x, y的 requires_grad的标记都为false, 所以输出的变量 requires_grad也为false

a.requires_grad

b = a + z #a ,z 中，有一个 requires_grad 的标记为True，那么输出的变量的 requires_grad为True
b.requires_grad


'''
变量的requires_grad标记的运算就相当于or。

如果你想部分冻结你的网络（ie.不做梯度计算），那么通过设置requires_grad标签是非常容易实现的。 下面给出了利用requires_grad使用pretrained网络的一个例子，只fine tune了最后一层。
'''
import torchvision
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = nn.Linear(512, 100)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)




# Locally disabling gradient computation
'''
The context managers torch.no_grad(), torch.enable_grad(), and torch.set_grad_enabled() are helpful for locally disabling and enabling gradient computation. See Locally disabling gradient computation for more details on their usage.

link = https://pytorch.org/docs/master/autograd.html#locally-disable-grad

'''
# e.g.1
x = torch.zeros(1, requires_grad = True)
with torch.no_grad():
    y = x * 2

y.requires_grad

is_train = False
with torch.set_grad_enabled(is_train):
    y = x * 2
y.requires_grad

torch.set_grad_enabled(True)
y = x * 2
y.requires_grad

torch.set_grad_enabled(False)
y = x * 2
y.requires_grad

# e.g.2
x = torch.tensor([1], requires_grad=True)
with torch.no_grad():
    y = x * 2
y.requires_grad

# torch.autograd.enable_grad
x = torch.tensor([1], requires_grad=True)
with torch.no_grad():
    with torch.enable_grad():
        y = x * 2
y.requires_grad
y.backward()
x.grad





















