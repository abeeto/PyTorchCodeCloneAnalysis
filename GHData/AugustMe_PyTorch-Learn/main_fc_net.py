# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:36:35 2019

@author: ZQQ
"""

import torch
from torch import nn, optim
from torch.autograd import Variable  # 导出Variable
from torch.utils.data import DataLoader #导入DataLoader，在torch中需要dataloader进行迭代
from torchvision import datasets, transforms # 导入pytorch内置的数据集

from models import *  # 从models文件夹导入所有网络
from models import net

# 定义一些超参数
batch_size = 64
learning_rate = 0.02 # 学习率
num_epoches = 500

# 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]) # 黑白图片这么处理，彩色图片是三通道的

#''' 另一种写函数进行数据预处理'''
#import numpy as np
#def data_tf(x):
#    x = np.array(x, dtype='float32') / 255
#    x = (x - 0.5) / 0.5 # 标准化，
#    x = x.reshape((-1,)) # 拉平
#    x = torch.from_numpy(x)
#    return x

# 数据集的下载器
train_dataset = datasets.MNIST(root='./MNIST_data', 
                               train=True,
                               transform=data_tf,
                               download=False) # 为True需要先下载，然后处理；为False使用下载完成的

test_dataset = datasets.MNIST(root='./MNIST_data', 
                              train=False, 
                              transform=data_tf
                              )

### 测试加载的数据
#data, data_label = test_dataset[0]#读取测试集中第一个数据
#print(data.shape)#数据的大小: 1*28*28
#print(data_label)#数据的真实标签:7

''' 使用 pytorch 自带的 DataLoader 定义一个数据迭代器 '''
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #训练集，shuffle为True时打乱数据，让数据更有选择随机性
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #对测试集进行迭代器编号

#''' 一个迭代器的内容 '''
#data, data_label = next(iter(train_loader))#一个迭代器中的内容
#print(data.shape)# 打印出一个批次的数据大小: batch_size张图片
#print(data_label.shape) # 一个批次数据的标签：batch_size个
#print(data_label) # 一个批次的数据真实标签，batch_size个真实数据标签


'''选择模型，总共3个模型，调用其中一个 '''
model = net.simpleNet(28 * 28, 300, 100, 10)
#model = net.Activation_Net(28 * 28, 300, 100, 10)
#model = net.Batch_Net(28 * 28, 300, 100, 10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # SGD，随机梯度下降优化器

'''训练之前定义一些空列表来添加训练得到的一些值 '''
losses = []
acces = []
eval_losses = []
eval_acces = []

# 开始训练模型
for epoch in range(num_epoches): # 进行num_epoches次数
    #for data in train_loader:
    train_loss = 0
    train_acc = 0
    model.train() # 网络开始训练
    for img,label in train_loader: 
        #print(data)
        #img, label = data
        img = img.view(img.size(0), -1) # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        #print(img)                                                    #接上：这个地方如果不加这语句，则将其添加到写好的net中
        img = Variable(img) #将数据打包成Variable
        label = Variable(label) #得到标签
        #''' 前向传播 '''
        out = model(img)
        print(out)
#        loss = criterion(out, label)
#        #print_loss = loss.data.item() 取出loss的数值(loss.item() 也可以？)
#        #''' 后向传播 '''
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        #''' 记录误差 '''
#        train_loss += loss.item()
#        #''' 计算分类准确率 '''
#        _,pred = out.max(1) # pred为一个一次数据的预测标签，挑选出输出时值最大的位置
#        num_correct = (pred == label).sum().item()#记录正确的个数
#        acc = num_correct / img.shape[0] #计算精确率，img.shape[0]
#        train_acc += acc
#        
#    # 一个epoch后，得到的数据添加到列表中    
#    losses.append(train_loss / len(train_loader))
#    acces.append(train_acc / len(train_loader))
##        if epoch%50 == 0:
##            print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
#
##''' 模型评估,测试集上检验效果 '''
#    model.eval() #将模型改为预测模式
#    eval_loss = 0
#    eval_acc = 0
#    for img,label in test_loader:
#        img = img.view(img.size(0), -1)
#        img = Variable(img)
#        label = Variable(label)
#        # 前向传播,测试就不需要反向传播了
#        out = model(img)
#        loss = criterion(out,label)
#        # 记录误差
#        eval_loss += loss.item()
#        # 记录准确率
#        _,pred = out.max(1)
#        num_correct = (pred == label).sum().item()
#        acc = num_correct / img.shape[0]
#        eval_acc += acc
#        
#    eval_losses.append(eval_loss / len(test_loader))
#    eval_acces.append(eval_acc / len(test_loader))
#    
#    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
#          .format(epoch, train_loss / len(train_loader),train_acc / len(train_loader),
#                  eval_loss / len(test_loader),eval_acc / len(test_loader)))

#import numpy as np
#print('平均准确率：{:.4f}'.format( np.array(eval_acces).sum() / len(eval_acces) ))
#    
#  
''' 画出 loss 曲线和 准确率曲线 '''     
import matplotlib.pyplot as plt
import numpy as np

# 训练过程，train loss
plt.figure(1)
plt.title('train loss')
plt.plot(np.arange(len(losses)), losses) # plot(x,y)
plt.savefig('./MNIST_fc_result/train_loss_500_epoches.png',dpi=600) # 保存图片

# 训练过程，train_acc
plt.figure(2)
plt.plot(np.arange(len(acces)), acces)
plt.title('train acc') 
plt.savefig('./MNIST_fc_result/train_acc_500_epoches.png',dpi=600) # 保存图片

# 测试过程，test_loss
plt.figure(3)
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.savefig('./MNIST_fc_result/test_loss_500_epoches.png',dpi=600) # 保存图片

# 测试过程，test_acc
plt.figure(4)
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc') 
plt.savefig('./MNIST_fc_result/test_acc_500_epoches.png',dpi=600) # 保存图片

# 保存训练和测试过程中产生的变量 
losses_ = np.array(losses).reshape(len(losses),1) # 先将list类型保存为numpy
np.save('./variable_fc_result/losses',losses_) # 保存为npy类型

acces_ = np.array(acces).reshape(len(acces),1) # 先将list类型保存为numpy
np.save('./variable_fc_result/acces',acces_) # 保存为npy类型

eval_losses_ = np.array(eval_losses).reshape(len(eval_losses),1) # 先将list类型保存为numpy
np.save('./variable_fc_result/eval_lossess',eval_losses_) # 保存为npy类型

eval_acces_ = np.array(eval_acces).reshape(len(eval_acces),1) # 先将list类型保存为numpy
np.save('./variable_fc_result/eval_acces',eval_acces_) # 保存为npy类型

# 加载保存的变量,以losses为例
import numpy as np
import matplotlib.pyplot as plt
losses = np.load('./variable_result/losses.npy')
plt.figure()
plt.title('train loss')
plt.plot(np.arange(len(losses)),losses)












