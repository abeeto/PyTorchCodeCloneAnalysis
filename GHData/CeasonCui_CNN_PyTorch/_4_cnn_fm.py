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
import seaborn as sns
import pandas as pd


EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
channel = 4

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
        self.fc1 = nn.Linear(channel*1 * 32 * 32, 2)   # fully connected layer, output 2 classes
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.float()
        x = x.view(-1, 1, 64, 64)
        #x = x.reshape(-1, 1, 64, 64)
        x = self.conv1(x)
        x = self.relu(x)
        x1 = x.reshape(-1, 1, 64, 64)
        x = self.pool(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.fc1(x)
        #output = self.softmax(x)
        return  x1,x   # return x for visualization

cnn2 = CNN()
cnn2.load_state_dict(torch.load('cnn_52l_4ch_1cov_100e.pth'))

shape='hexagon'
path = './dataset_6/dataset/'+shape+'_d4_p361.jpg'
img = io.imread(path)
train_loader = DataLoader(img,batch_size=64,shuffle=False)#使用DataLoader加载数据
for i_batch,batch_data in enumerate(train_loader):
    print(batch_data.size())
    feature = cnn2(batch_data)[0]
    fs=cnn2(batch_data)[1]
    #fs=fs.numpy()
    # print(fs.max.item())
    # print(fs.min.item())
    print(feature.size())
    print(feature[1].size())
    for i in range(4):
        
        image = feature[i].cpu().clone().detach().numpy()
        # image = feature[i].cpu().clone()
        # print(image.max())
        # print(image.min())
        image = np.reshape((image), (64, 64, 1))
        image1 = np.concatenate((image, image, image), axis=2)
        cv2.imwrite('feature_'+shape+'_'+str(i+1)+'.jpg', image1.astype(np.uint8))
        #image = image.squeeze(0)
        #image1 = transforms.ToPILImage()(image)
        # image1.save('feature_'+str(i+1)+'.jpg', quality=95)
        #feature = feature[i].detach().numpy()
        #cv2.imwrite('./feature.jpg',feature)
# feature = feature.numpy()
# cv2.imwrite('./feature.jpg',feature)
print(cnn2.state_dict()['conv1.0.weight'])
#print(cnn2.state_dict().keys())
#df = cnn2.state_dict()['conv1.0.weight'].detach().numpy()
df = cnn2.state_dict()['conv1.0.weight']
print(df)

for i in range(4):
    print(df[i])
    plt.imshow(df[i][0].numpy(), cmap = 'seismic')
    plt.axis('off')
    plt.show()


# a = np.array([[1,0.107562,0.270034,0.266753],
#               [0.107562,1,0.543716,0.540923],
#               [0.270034,0.543716,1,0.950266],
#               [0.266753,0.540923,0.950266,1]])

# fig, ax = plt.subplots(figsize = (9,9))
# #二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
# for i in range(4):
#     sns.heatmap(pd.DataFrame(np.round(df[i],2)),
#                 annot=True, square=True, cmap="Blues")
# #sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
# #            square=True, cmap="YlGnBu")
# # ax.set_title('二维数组热力图', fontsize = 18)
# # ax.set_ylabel('image', fontsize = 18)
# # ax.set_xlabel('iamge', fontsize = 18) #横变成y轴，跟矩阵原始的布局情况是一样的
#     plt.savefig('./out_'+str(i)+'.png')
#     plt.show()
