#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:39:19 2018

@author: jangqh
"""

"""
1、一维线性回归的代码实现
"""

import torch
import numpy as np
from torch.autograd import Variable
#import matplotlib 
#import matplotlib.pyplot as plt
#
#
#torch.manual_seed(2017)
#
#x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                    [9.779],[6.181],[7.59], [2.167], [7.042],
#                    [10.791], [5.313], [7.997], [3.1]], dtype = np.float32)
#y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                    [3.366], [2.596], [2.53], [1.221], [2.827],
#                    [3.465], [1.65], [2.904], [1.3]], dtype = np.float32)
#
##plt.plot(x_train, y_train, 'bo')
##plt.show()
#
####转化成tensor
#x_train = torch.from_numpy(x_train)
#y_train = torch.from_numpy(y_train)
#
#w = Variable(torch.randn(15), requires_grad  =True)
#b = Variable(torch.zeros(15), requires_grad = True)
#
###构建线性模型
#x_train = Variable(x_train)
#y_train = Variable(y_train)
#
#def linear_model(x):
#    return x * w + b
#
#y_ = linear_model(x_train)
#
#
#
##模型的输出结果
#plt.plot(x_train.data.numpy(), y_train.data.numpy(),  'bo',  label = 'real')
#plt.plot(x_train.data.numpy(), y_.data.numpy(),  'ro',  label = 'estimated')
#plt.legend()
##plt.show()
#
#
#####计算误差
#def get_loss(y_, y):
#    return torch.mean((y_ - y_train) ** 2)
#
#loss = get_loss(y_, y_train)
#
#print loss
#
#loss.backward()
#
#print(w.grad)
#print(b.grad)
#
####更新一次参数
#w.data = w.data - 1e-2 * w.grad.data
#b.data = b.data - 1e-2 * b.grad.data
#
#y_ = linear_model(x_train)
#plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label = 'real')
#plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label = 'estimated')
#plt.legend()
##plt.show()
#
#
#for e in range(10):
#    y_ = linear_model(x_train)
#    loss = get_loss(y_, y_train)
#    
##    w.grad.zero_()
##    b.grad.zero_()  #归零梯度 
#    
#    loss.backward()
#    
#    w.data = w.data - 1e-2 * w.grad.data
#    b.data = b.data - 1e-2 * b.grad.data
#    print('epoch:{}, loss:'.format(e, loss.data))

"""
2、多项式回归模型
"""

w_target = np.array([0.5, 3, 2.4])
b_target = np.array([0.9])

f_des = 'y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(b_target[0],
         w_target[0], w_target[1], w_target[2])
print (f_des)

x_sample = np.arange(-3,3.1,0.1)
#print x_sample
#print x_sample.shape
y_sample = b_target[0]+w_target[0]*x_sample+w_target[1]*x_sample**2+\
                   w_target[2]*x_sample**3
print "y_sample:",y_sample
print "y_sample.shape:",y_sample.shape
#plt.plot(x_sample, y_sample, label = 'real curve')
#print y_sample
#plt.show()

###构造数据集
x_train = np.stack([x_sample ** i for i in range(1,4)], axis = 1)
print "x_train:",x_train
print "x_train:",x_train.shape
x_trian = torch.from_numpy(x_train).float()
print "x_train:",x_train
print "x_train type:",type(x_train)
x_train = torch.Tensor(x_train)
print "x_train type:",type(x_train)

y_train = torch.from_numpy(y_sample).float().unsqueeze(1)
print y_train
print y_train.size()
y_train = torch.Tensor(y_train)
print "y_train type:", type(y_train)


w = Variable(torch.randn(3, 1), requires_grad  = True)
print "w-type:",type(w)
b = Variable(torch.randn(1), requires_grad = True)
print "b-type:",type(b)
#
x_train = Variable(x_train)
y_train = Variable(y_train)
print "x_train type:", type(x_train)
print "y_train type:", type(y_train)



print "x_train size:", x_train.size()
print "y_train size:", y_train.size()
print "x size:", w.size()
print "b szie:", b.size()

def multi_linear(x):
    return torch.mm(x, w) + b
#
#
y_pred = multi_linear(x_train)
#

#plt.plot(x_train.data.numpy()[:,0], y_pred.data.numpy(), label = 'fitting curve',color  ='r')
#plt.plot(x_train.data.numpy()[:,0], y_sample, label = 'real curve', color = 'b')
#plt.legend()
#plt.show()


loss = get_loss(y_pred, y_train)
print loss

loss.backward()

print(w.grad)
print(b.grad)

w.data = w.data - 0.01 * w.grad.data
b.data = b.data = 0.01 * b.grad.data



for e in range(100):
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred, y_train)
    
    w.grad.data.zero_()
    b.grad.data.zero_()
    
    w.data = w.data = 0.01 * w.grad.data
    b.data = b.data - 0.01 * b.grad.data
    
    if (e+1) %20 == 0:
        print("epoch {} , Loss:{:.5f}".format(e+1, loss.data[0]))


"""
3、多项式回归 P38
"""
#import torch.nn as nn
#import torch.optim as optim
#
#def make_feature(x):
#    x = x.unsqueeze(1)
#    return torch.cat([x**i for i in range(1,4)], 1)
#
#W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
#b_target = torch.FloatTensor([0.9])
#print b_target.size()
#print b_target[0]
#
#
#def f(x):
#    return x.mm(W_target) + b_target[0]
#
#def get_batch(batch_size = 32):
#    random = torch.randn(batch_size)
#    x = make_feature(random)
#    y = f(x)
#    if torch.cuda.is_available():
#        return Variable(x).cuda(), Variable(y).cuda()
#    else:
#        return Variable(x), Variable(y)
#
#class poly_model(nn.Module):
#    def __init__(self):
#        super(poly_model,self).__init__()
#        self.poly = nn.Linear(3, 1)
#        
#    def forward(self, x):
#        out = self.poly(x)
#        return out
#    
#if torch.cuda.is_available():
#    model = poly_model().cuda()
#else:
#    model = poly_model()
#    
#
#criterion = nn.MSELoss()
#optimizer = optim.SGD(model.parameters(), lr = 1e-3)
#epoch = 0
#
#
############################begin
#
#while True:
#    #读入原始数据x,y  tensor类型
#    batch_x, batch_y = get_batch()
##    print batch_x.size()
#    
#    #模型输出的 out
#    output = model(batch_x)
#    
#    #损失 y-out
#    loss = criterion(output, batch_y)
#    print_loss = loss.data[0]
##    print print_loss
#    
#    #优化器 梯度清零
#    optimizer.zero_grad()
#    #计算梯度
#    loss.backward()
#    #更新参数
#    optimizer.step()
#    
#    epoch += 1
#    if epoch %10 == 0:
#        print "Epoch:{}".format(epoch), print_loss 
#        pass
#
#    if print_loss < 0.5:
#        break
#
##print W_target
#print "batch_x:", batch_x.size()
#print "W_target:", W_target.size()
#print "b_target:", b_target.size()
#print "b_target:", b_target[:]
#print "batch_y:", batch_y.size()
#print "output:", output.size()

    












































