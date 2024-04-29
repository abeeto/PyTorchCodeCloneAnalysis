
import os
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import writer
import torchvision
#from torch.utils.data import DataLoader
import torch.nn as nn
from model import *

this_dir=os.getcwd()
os.chdir('../practice_cifar10')
train_data=torchvision.datasets.CIFAR10(root='./data',train=True,transform=torchvision.transforms.ToTensor(),download=False)
test_data=torchvision.datasets.CIFAR10(root='./data',train=False,transform=torchvision.transforms.ToTensor(),download=False)

train_data_size=len(train_data)
test_data_size=len(test_data)
print('train len:',train_data_size,'test len:',test_data_size)

##使用dataloader加载

train_dataloader=DataLoader(train_data,batch_size=16)
test_dataloader=DataLoader(test_data,batch_size=16)


##创建模型
model=Cifar()
if torch.cuda.is_available():
    model=model.cuda()
#损失函数
loss_fn=nn.CrossEntropyLoss()
##优化器
lr=0.0001
opt=torch.optim.SGD(model.parameters(),lr=lr)

##设置参数
total_train_step=0
total_test_step=0
epoch=10

##添加tensorboard


for i in range(epoch):
    print('----------第{}轮训练-----------'.format(i+1))
    ##开始训练
    for data in train_dataloader:
        imgs,targets=data
        output=model(imgs)
        loss=loss_fn(output,targets)

        ##开始优化
        opt.zero_grad()###梯度清零
        loss.backward()
        opt.step()
        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print('训练次数：{}, loss {}'.format(total_train_step,loss.item()))
            #writer.add_scalar("train_loss",loss.item(),total_train_step)
    ##测试步骤
    total_test_loss=0
    total_acc=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            output=model(imgs)
            loss=loss_fn(output,targets)
            total_test_loss=total_test_loss+loss
            acc=(output.argmax(1)==targets).sum()
            total_acc=total_acc+acc

    print('整体测试集第loss {}'.format(total_test_loss))
    print('整体测试集 accuracy{}'.format(total_acc/test_data_size))

    total_test_loss=total_test_loss+1
    torch.save(model,'cifar10_model{}.pth'.format(i))
    print('model has been saved')



