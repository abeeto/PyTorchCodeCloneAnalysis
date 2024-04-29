# -*- coding: utf-8 -*-
'''
    link = http://www.pytorchtutorial.com/pytorch-note1-what-is-pytorch/#NN
    
    Variable：是Tensor的一个wrapper，不仅保存了值，而且保存了这个值的creator，需要BP的网络都是Variable参与运算
'''

import torch
x = torch.Tensor(2,3,4)
x.size()

torch.Size([2,3,4])

a = torch.rand(2,3,4)
b = torch.rand(2,3,4)

_=torch.add(a,b, out=x)     # 使用Tensor()方法创建出来的Tensor用来接收计算结果，当然torch.add(..)也会返回计算结果的

torch.cuda.is_available()



# 自动求导
# pytorch的自动求导工具包在torch.autograd中
from torch.autograd import Variable
x = torch.rand(5)
x =  Variable(x, requires_grad = True)
y = x * 2
grads = torch.FloatTensor([1,2,3,4,5])

# 如果y是scalar的话，那么直接y.backward()，然后通过x.grad方式，就可以得到var的梯度
# 如果y不是scalar，那么只能通过传参的方式给x指定梯度
y.backward(grads) 
x.grad


# neural networks
'''
    使用torch.nn包中的工具来构建神经网络 需要以下几步：
        定义神经网络的权重,搭建网络结构
        遍历整个数据集进行训练
        将数据输入神经网络
        计算loss
        计算网络权重的梯度
        更新网络权重
            weight = weight + learning_rate * gradient
'''

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):#需要继承这个类
    def __init__(self):
        super(Net, self).__init__()
        #建立了两个卷积层，self.conv1, self.conv2，注意，这些层都是不包含激活函数的
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        #三个全连接层
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): #注意，2D卷积层的输入data维数是 batchsize*channel*height*width
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
net

len(list(net.parameters()))
#为什么是10呢？ 因为不仅有weights，还有bias， 10=5*2。
#list(net.parameters())返回的learnable variables 是按照创建的顺序来的
#list(net.parameters())返回 a list of torch.FloatTensor objects
                                
           
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
#这个地方就神奇了，明明没有定义__call__()函数啊，所以只能猜测是父类实现了，并且里面还调用了forward函数
#查看源码之后，果真如此。那么，forward()是必须要声明的了，不然会报错

out
out.backward(torch.randn(1,10))





#使用loss criterion 和 optimizer训练网络
'''
torch.nn包下有很多loss标准。同时torch.optimizer帮助完成更新权重的工作。这样就不需要手动更新参数了
'''
target = torch.arange(1, 11) # a dummy target, for example
target = target.view(1, -1)  # make it the same as output
criterion = nn.MSELoss()

learning_rate =0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate) # 有了optimizer就不用写这些了

import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# in your training loop:
for i in range(num_iterations):
    optimizer.zero_grad()       # zero the gradient buffers，如果不写这个函数，也是可以正常工作的，不知这个函数的必要性在哪？
    
    # Loss Function 
    output = net(input)    # 这里就体现出来动态建图了，你还可以传入其他的参数来改变网络的结构
    
    loss = criterion(output, target)
    print(loss)   
    loss.backward()
    optimizer.step() # Does the update i.e. Variable.data -= learning_rate*Variable.grad


# 其它
'''
    关于求梯度，只有我们定义的Variable才会被求梯度，由creator创造的不会去求梯度

    自己定义Variable的时候，记得Variable(Tensor, requires_grad = True),这样才会被求梯度，不然的话，是不会求梯度的
'''
# numpy to Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy (a)

np.add(a, 1, out=a)
print(a) # 如果a 变的话， b也会跟着变，说明b只是保存了一个地址而已，并没有深拷贝
print(b) # Variable只是保存Tensor的地址，如果Tensor变的话，Variable也会跟着变

a = np.ones(5)
b = torch.from_numpy(a)
a_ = b.numpy() # Tensor --> ndarray
np.add(a, 1, out=a) # 这个和 a = np.add(a,1)有什么区别呢？
# a = np.add(a,1) 只是将a中保存的指针指向新计算好的数据上去
# np.add(a, 1, out=a) 改变了a指向的数据
 
# 将Tensor放到Cuda上
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y
    
# torch.Tensor(1,2,3) 与 torch.Tensor([1,2,3]) 的区别
torch.Tensor(1,2,3)   # 生成一个 shape 为 [1,2,3] 的 tensor
torch.Tensor([1,2,3]) # 生成一个值为 [1,2,3] 的 tensor











                  