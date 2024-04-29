# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

#計算に使用するデバイスの選択
device = 'cuda'

def read_mnist_label(name, device):#指定されたファイル名のバイナリをfloat32型で読み込む出力の型はtorch.fload32
    name_data_file = open(name, 'rb')#引数のファイルを r:Readモード, b:バイナリモードで読み込む
    name_data = np.fromfile(name_data_file, np.uint8, -1)#バイナリを8bitのunsignedintでnumpyに格納
    size = name_data.size - 8#先頭の8byteを無視
    output_name_data = torch.zeros(size,10)#60000x10　数字データをOneHot形式に変換する
    for i in range(size):
        output_name_data[i][name_data[i+8]] = 1.0;
    output_name_data = np.array(output_name_data, dtype=np.float32)#np.uint8をnp.float32に変換
    return torch.from_numpy(np.reshape(output_name_data, (size/100,100,10))).to(device)#ミニバッチサイズは100

def read_mnist_image(name, device):#指定されたファイル名のバイナリをuint8型で読み込む
    name_image_file = open(name,'rb')
    name_image = np.fromfile(name_image_file, np.uint8, -1)
    size = name_image.size - 16
    name_image = np.array(name_image, dtype=np.float32)
    return torch.from_numpy(np.reshape(name_image[16:], (size/78400,100,1,28,28))).to(device)#ミニバッチサイズは100

train_label = read_mnist_label('train-labels', device)  #600,100,10
test_label = read_mnist_label('test-labels', device)    #100,100,10
train_image = read_mnist_image('train-images', device)  #600,100,1,28,28
test_image = read_mnist_image('test-images', device)    #100,100,1,28,28



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)#1x28x28->6x24x24
        self.conv2 = nn.Conv2d(6,16,5)#6x12x12->16x8x8
        self.mlp1 = nn.Linear(256,120)#256->120
        self.mlp2 = nn.Linear(120,84)#120->84
        self.mlp3 = nn.Linear(84,10)#84->10

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))#6x24x24->6x12x12
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))#16x8x8->16x4x4
        x = x.view(-1, self.num_flat_features(x))#16x4x4->196
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
net.to(device)

start = time.time()#時間計測開始

optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()
for j in range(10):
    print('epoch:',j)
    for i in range(100):
        output = net(train_image[i])
        loss = criterion(output,train_label[i])
        loss.backward()
        optimizer.step()
for i in range(0):#テスト
    print(net(test_image[i])[0])
    print(test_label[i][0])

elapsed_time = time.time() - start
print(elapsed_time,' seconds with ',device)






























#
