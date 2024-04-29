import cv2
from torch.autograd import Variable
from torchvision import models
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
channel = 4

class MyDataset(Dataset):
    def __init__(self, root_dir, img, transform=None): #__init__是初始化该类的一些基础参数
        self.root_dir = root_dir   #文件目录
        self.transform = transform #变换
        #self.images = os.listdir(self.root_dir)#目录里的所有文件
        self.images = img
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        image_index = self.images[index]#根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)#获取索引为index的图片的路径名
        img = io.imread(img_path)# 读取该图片
        #label = img_path.split('\\')[-1].split('_')[0]# 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。我这里是"E:\\Python Project\\Pytorch\\dogs-vs-cats\\train\\cat.0.jpg"，所以先用"\\"分割，选取最后一个为['cat.0.jpg']，然后使用"."分割，选取[cat]作为该图片的标签
        l = img_path.split('/')[-1].split('_')[0]
        if l=='ellipse':
            label = 0
        # if l=='square':
        #     label = 1
        # if l=='triangle':
        #     label = 2
        # if l=='pentagon':
        #     label = 3
        # if l=='hexagon':
        #     label = 4  
        else:
            label = 1          
        #label =l
        sample = {'image':img,'label':label}#根据图片和标签创建字典
        
        if self.transform:
            sample = self.transform(sample)#对样本进行变换
        return sample #返回该样本

    def __len__(self): #return count of dataset
        return len(self.images)

path = './dataset_6/dataset/'
full_dataset = os.listdir(path)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_data = MyDataset(path, train_dataset,transform=None)#初始化类，设置数据集所在路径以及变换
train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)#使用DataLoader加载数据
train_data = MyDataset(path, train_dataset,transform=None)#初始化类，设置数据集所在路径以及变换
train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)#使用DataLoader加载数据


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 64, 64)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=channel,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 64, 64)
            #nn.ReLU(),                      # activation
            #nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 32, 32)
        )
        self.conv2 = nn.Sequential(         # input shape (32, 32, 32)
            nn.Conv2d(channel, channel*2, 3, 1, 1),     # output shape (64, 32, 32)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (64, 16, 16)
        )
        self.conv3 = nn.Sequential(         # input shape (64, 16, 16)
            nn.Conv2d(channel*2, channel*4, 3, 1, 1),     # output shape (128, 16, 16)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (128, 8, 8)
        )
        self.conv4 = nn.Sequential(         # input shape (128, 8, 8)
            nn.Conv2d(channel*4, channel*8, 3, 1, 1),     # output shape (256, 8, 8)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (256, 4, 4)
        )
        self.fc1 = nn.Linear(channel*4 * 8 * 8, 2)   # fully connected layer, output 2 classes
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.float()
        x = x.view(-1, 1, 64, 64)
        #x = x.reshape(-1, 1, 64, 64)
        x = self.conv1(x)
        x2 = self.relu(x)
        x1 = x2.reshape(-1, 1, 64, 64)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.fc1(x)
        #output = self.softmax(x)
        return  output   # return x for visualization

cnn = CNN()
cnn.load_state_dict(torch.load('cnn_5l_4ch_3cov_20e.pth'))

loss_func = nn.CrossEntropyLoss()  

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist 
i = 0
sum_loss = 0.0          #lossの合計
sum_correct = 0         #正解率の合計
sum_total = 0           #dataの数の合計

for i_batch,batch_data in enumerate(train_loader):
    b_x = batch_data['image']
    b_y = batch_data['label']
    #output = cnn(b_x)[0]               # cnn output
    output = cnn(b_x)
    loss = loss_func(output, b_y)   # cross entropy loss
    #optimizer.zero_grad()           # clear gradients for this training step
    sum_loss += loss.item()                            #lossを足していく
    _, predicted = output.max(1)                      #出力の最大値の添字(予想位置)を取得
    sum_total += b_y.size(0)                        #labelの数を足していくことでデータの総和を取る
    sum_correct += (predicted == b_y).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す
        #print(predicted != b_y)
    print(i_batch)
    print(predicted)
    print(b_y)    
    print("train mean loss={}, accuracy={}, miss={}".format((sum_loss*BATCH_SIZE/len(train_loader.dataset)), float(sum_correct/sum_total), (sum_total-sum_correct))) #lossとaccuracy出力
train_loss_value.append(sum_loss*BATCH_SIZE/len(train_loader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
train_acc_value.append(float(sum_correct/sum_total))  