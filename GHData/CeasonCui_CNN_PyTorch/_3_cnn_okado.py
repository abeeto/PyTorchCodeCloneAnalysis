import os

# third-party library
import cv2
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
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 100              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
channel = 8



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
        if l=='square':
            label = 1
        if l=='triangle':
            label = 2
        if l=='pentagon':
            label = 3
        if l=='hexagon':
            label = 4   
        # else:
        #     label = 1         
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
    # for i_batch,batch_data in enumerate(train_loader):
    #     if i_batch == 3:
    #         break
    #     print(i_batch)#打印batch编号
    #     print(batch_data['image'].size())#打印该batch里面图片的大小
    #     #print(batch_data['image'])
    #     print(batch_data['label'])#打印该batch里面图片的标签

# filelist = os.listdir(path)
# random.shuffle(filelist)
# train = filelist[: int(0.8 * len(filelist))]
# torch.utils.data.random_split

# plot one example
# print(data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data[0]['image'], cmap='gray')
plt.title('%i' % train_data[0]['label'])
plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
#train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels[:2000]
test_data = MyDataset(path, test_dataset,transform=None)#初始化类，设置数据集所在路径以及变换
test_loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)

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
        self.fc1 = nn.Linear(channel*1 * 32 * 32, 5)   # fully connected layer, output 2 classes
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.float()
        x = x.view(-1, 1, 64, 64)
        #x = x.reshape(-1, 1, 64, 64)
        x = self.conv1(x)
        x = self.relu(x)
        x1 = x.reshape(-1, 1, 64, 64)
        x = self. pool(x)
        #x1 = x.reshape(-1, 1, 64, 64)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.fc1(x)
        #output = self.softmax(x)
        return output, x1    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
# training and testing

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist 

for epoch in range(EPOCH):
    i = 0
    sum_loss = 0.0          #lossの合計
    sum_correct = 0         #正解率の合計
    sum_total = 0           #dataの数の合計
    for step, batch_data in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = batch_data['image']
        b_y = batch_data['label']
        #output = cnn(b_x)[0]               # cnn output
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

    for step, batch_data in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = batch_data['image']
        b_y = batch_data['label']
        #output = cnn(b_x)[0]               # cnn output
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        sum_loss += loss.item()                            #lossを足していく
        _, predicted = output.max(1)                      #出力の最大値の添字(予想位置)を取得
        sum_total += b_y.size(0)                        #labelの数を足していくことでデータの総和を取る
        sum_correct += (predicted == b_y).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す
        #print(predicted != b_y)
        if step==1:
                print(epoch)
                print(predicted)
                print(b_y)
    
    print("train mean loss={}, accuracy={}, miss={}".format((sum_loss*BATCH_SIZE/len(train_loader.dataset)), float(sum_correct/sum_total), (sum_total-sum_correct))) #lossとaccuracy出力
    train_loss_value.append(sum_loss*BATCH_SIZE/len(train_loader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
    train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持
    
    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    #test dataを使ってテストをする
    for step, batch_data in enumerate(test_loader):   # gives batch data, normalize x when iterate train_loader
        t_x = batch_data['image']
        t_y = batch_data['label']
        optimizer.zero_grad()
        output = cnn(t_x)[0]
        feature = cnn(t_x)[1]
        # print('fs: ')
        # print(feature.size())
        loss = loss_func(output, t_y)   # cross entropy loss
        sum_loss += loss.item()
        _, predicted = output.max(1)
        sum_total += t_y.size(0)
        sum_correct += (predicted == t_y).sum().item()
        # if step==1:
        #     for i in range(4):       
        #         image = feature[i].cpu().clone().detach().numpy()
        #         # image = feature[i].cpu().clone()
        #         # print(image.max())
        #         # print(image.min())
        #         #image = np.reshape((image * 255), (64, 64, 1))
        #         image1 = np.concatenate((image, image, image), axis=2)
        #         cv2.imwrite('feature_'+str(epoch)+'_'+str(i+1)+'.jpg', image1.astype(np.uint8))
    print("test  mean loss={}, accuracy={}, miss={}"
            .format(sum_loss*BATCH_SIZE/len(test_loader.dataset), float(sum_correct/sum_total),(sum_total-sum_correct)))
    test_loss_value.append(sum_loss*BATCH_SIZE/len(test_loader.dataset))
    test_acc_value.append(float(sum_correct/sum_total))
        
        # if step % 50 == 0:
        #     for step, batch_tdata in enumerate(test_loader):
        #         t_x = batch_tdata['image']
        #         t_y = batch_tdata['label']
        #         test_output, last_layer = cnn(t_x)
        #         pred_y = torch.max(test_output, 1)[1].data.numpy()
        #         accuracy = float((pred_y == t_y).astype(int).sum()) / float(t_y.size(0))
        #         print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
        #         if HAS_SK:
        #             # Visualization of trained flatten layer (T-SNE)
        #             tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        #             plot_only = 500
        #             low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
        #             labels = test_data['label'].numpy()[:plot_only]
        #             plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
# test_output, _ = cnn(test_data['label'][:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_data['label'][:10].numpy(), 'real number')

torch.save(cnn.state_dict(),'cnn_5l_2ch_3cov_100e.pth')

# net2=torch.load('cnn8_3.pkl')
# print (net2)

plt.plot(range(EPOCH), train_loss_value)
plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 2.5)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("accuracy_image.png")


