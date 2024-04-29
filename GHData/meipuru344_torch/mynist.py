# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import sepconv

def read_mnist_label(name):#指定されたファイル名のバイナリをfloat32型で読み込む出力の型はtorch.fload32
    name_data_file = open(name, 'rb')#引数のファイルを r:Readモード, b:バイナリモードで読み込む
    name_data = np.fromfile(name_data_file, np.uint8, -1)#バイナリを8bitのunsignedintでnumpyに格納
    size = name_data.size - 8#先頭の8byteを無視
    output_name_data = torch.zeros(size,10)#60000x10　数字データをOneHot形式に変換する
    for i in range(size):
        output_name_data[i][name_data[i+8]] = 1.0;
    output_name_data = np.array(output_name_data, dtype=np.float32)#np.uint8をnp.float32に変換
    return torch.from_numpy(np.reshape(output_name_data, (int(size/100),100,10)))#ミニバッチサイズは100

def read_mnist_image(name):#指定されたファイル名のバイナリをuint8型で読み込む
    name_image_file = open(name,'rb')
    name_image = np.fromfile(name_image_file, np.uint8, -1)
    size = name_image.size - 16
    name_image = np.array(name_image, dtype=np.float32)
    return torch.from_numpy(np.reshape(name_image[16:], (int(size/78400),100,1,28,28)))#ミニバッチサイズは100

train_label = read_mnist_label('train-labels')  #600,100,10
test_label = read_mnist_label('test-labels')    #100,100,10
train_image = read_mnist_image('train-images')  #600,100,1,28,28
test_image = read_mnist_image('test-images')    #100,100,1,28,28


class Net(nn.Module):
    def Subnet(self):
            return torch.nn.Sequential(
                nn.Linear(84,10),
                nn.ReLU()
            )

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)#1x28x28->6x24x24
        self.conv2 = nn.Conv2d(6,16,5)#6x12x12->16x8x8


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))#6x24x24->6x12x12
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))#16x8x8->16x4x4

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

## one image
one = torch.ones(1,1,28,28, dtype=torch.float32)
allzero = torch.ones((1,16,4,4), dtype=torch.float32)#
allzero *=1

start = time.time()

#optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer = optim.Adamax(net.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
criterion = nn.MSELoss()
for j in range(100):
    print("epoch:"+str(j))
    for i in range(1000):
        output = net(one)
        loss = criterion(output,allzero)
        loss.backward()
        optimizer.step()
        #'{:.2f}'.format(123.456)
    print(output.sum())
for i in range(0):#テスト
    print(net(test_image[i])[0])
    print(test_label[i][0])

elapsed_time = time.time() - start
print(elapsed_time,' seconds with CPU')

#output = net()






























#
