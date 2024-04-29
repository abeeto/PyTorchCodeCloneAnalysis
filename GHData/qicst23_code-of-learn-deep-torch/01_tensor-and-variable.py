#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:18:00 2018

@author: jangqh
"""

import torch
import numpy as np
numpy_tensor = np.random.randn(3,4)
print type(numpy_tensor)
print numpy_tensor

"""
1、把pytorch当成numpy使用
"""

#numpy----pytorch tensor
#float tensor
print  "---"
pytorch_tensor1 = torch.Tensor(numpy_tensor)
print type(pytorch_tensor1)
print pytorch_tensor1

#double tensor
print "--"
pytorch_tensor2 = torch.from_numpy(numpy_tensor)
print type(pytorch_tensor2)
print pytorch_tensor2

##pytorch tensor----numpy
#CPU
print "----"
numpy_array = pytorch_tensor1.numpy()
print type(numpy_array)
print numpy_array

#GPU上的Tensor不能直接转换为numpy ndarray 类型，先用.cpu() 将GPU上的tensor转到CPU
numpy_array1 = pytorch_tensor1.cpu().numpy()
print type(numpy_array1)
print numpy_array1


##GPU加速
#1 cuda类型数据
#dtype = torch.cuda.FloatTensor #默认GPU类型
#gpu_tensor = torch.randn(4,5).type(dtype)

#2 放在GPU上运行，推荐
#gpu_tensor = torch.randn(4,5).cuda(0) #放在第一个GPU上
#gpu_tensor = torch.randn(4,5).cuda(1) #放在第二个GPU上

# 将tensor 放回CPU
#cpu_tensor = gpu_tensor.cpu()


if torch.cuda.is_available():
    a_cuda = a.cuda()
    print(a_cuda)
else:
    print("Don't support GPU")

#size
print("pytorch_tensor size:",pytorch_tensor1.size())

#type
print("pytorch_tensor type:",pytorch_tensor1.type())

#dimention
print("pytorch_tensor dim:",pytorch_tensor1.dim())

#tensor 元素个数
print("pytorch_tensor num:",pytorch_tensor1.numel())


##
#
x = torch.randn(3, 2)
print("x type:",type(x))
x = x.type(torch.DoubleTensor)
print("x type:", type(x))
x_array = x.numpy()
print ("x array:",x_array.dtype)

"""
2、tensor的操作
"""
print('-----------------')
#
x = torch.ones(2,2)
print(type(x),x)

x = x.long()
print(type(x),x)

x = x.float()
print(type(x),x)

print "========"
x = torch.randn(4,3)
print(type(x),x)

#沿着行取最大值和下标
max_value, max_idx = torch.max(x, dim=1)
print(max_value,max_idx)

#沿着行求和
sum_x = torch.sum(x, dim = 1)
print("sum:",sum_x)

#增加维度或减少维度
print("-=-==============")
print('x.size::',x.size())
x = x.unsqueeze(0)
print("x.size::",x.size())
print x
x = x.unsqueeze(2)
print(x.size())
print x

x = x.squeeze(2)
print(x.size())
x = x.squeeze()
print(x.size())
print x




#维度交换
x = torch.randn(3,4,5)
print x
print(x.size())

y= x.permute(1,0,2)
print(y.size())

z = x.transpose(0, 2)
print(z.size())


x = x.view(-1,10)  #-1 表示 任意大小，5表示第二维变成5
print(x.size())
x = x.view(3,20)
print(x.size())

###求和
print("-====================")
x = torch.randn(3, 4)
y = torch.randn(3, 4)

z = x+ y
print z


##pyttorch 支持 inplace 操作 直接对tensor 进行操作不需要另外开辟内存空间，一般都是操作符后面 _ 
x = torch.rand(3,3)
print(x.size())
x.unsqueeze_(0)
print(x.size())
x.transpose_(1,0)
print(x.size())

x = torch.ones(3,3)
y = torch.ones(3,3)
print(x)
x.add_(y)
print(x)

##
x = torch.ones(4,4).float()
x[1:3, 1:3] = 2
print(x)

"""
3、variable的操作
"""
from torch.autograd import Variable

x_tensor = torch.randn(2, 5)
y_tensor = torch.randn(2, 5)

#requires_grad = True 求梯度
x = Variable(x_tensor, requires_grad = True)
y = Variable(y_tensor, requires_grad = True)
print x_tensor[1][:]
print x

z = torch.sum(x*x + 0.5*y*y)
print(z.size())
print z.data

z.backward()
print(x.grad)
print(y.grad)


#####例子
print "-================"
x = torch.Tensor([2])
print x

xx = Variable(x, requires_grad = True)
print xx
y = xx*xx

y.backward()
print(xx.grad)

print "==========================\
        ============================="



"""
4、自动求导
"""
x = Variable(torch.Tensor([2]), requires_grad = True)
print x
y = x + 2
z = y ** 2 + 3
print z

z.backward()
print(x.grad)

#
x = Variable(torch.randn(10,20), requires_grad = True)
y = Variable(torch.randn(10,5),requires_grad = True)
w = Variable(torch.randn(20,5),requires_grad = True)


##pytorch 中矩阵乘法用torch.mm()
out = torch.mean(y-torch.mm(x, w))
out.backward()
print(x.grad)
print(y.grad)
print(w.grad)



"""
5、复杂情况下的自动求导
"""
print "---=--=-=-=---=------===---==-"
m = Variable(torch.FloatTensor([[2, 3]]),requires_grad = True)
print m
n = Variable(torch.zeros(1,2))
print n

n[0,0] = m[0, 0] ** 2
n[0,1] = m[0, 1] ** 3
print n

#将（w0,w1）取成（1，1）
print n.size()
#pytorch 中tensor 类型大小用x.size()
n.backward(torch.ones(n.size()))
print(m.grad)


"""
6、多次求导
"""
#pytorch 进行一次求导后，计算图就会丢弃
print "-========================"
x = Variable(torch.FloatTensor([3]), requires_grad = True)
y = x * 2 +x ** 2 + 3
print y

#####
####多次求导时，传入参数 retain_graph = True
#y.backward(retain_graph=True)
#print(x.grad)


###例子
print "============="
x = Variable(torch.FloatTensor([[2, 3]]), requires_grad = True)
k = Variable(torch.zeros(1,2))

k[0,0] = x[0,0] ** 2 + 3 * x[0, 1]
k[0,1] = x[0,1] ** 2 + 2 * x[0, 0]

print k

j = torch.zeros(2,2)

#k.backward(torch.FloatTensor([[1,0]]))
#j[:,0] = x.grad.data

k.backward(torch.FloatTensor([[0,1]]))
j[:,1] = x.grad.data
 
m.grad.data.zero_()
print m
print j
















