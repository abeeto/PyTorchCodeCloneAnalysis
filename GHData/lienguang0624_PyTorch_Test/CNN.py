import torch
import  torch.nn as nn
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

EPOCH =1
BATCH_SIZE = 50
LR =0.001
DOWNLOAD_MNIST =False
# 加载数据集
train_data = torchvision.datasets.MNIST(
    root = './mnist',
    train = True,
    transform=torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST
)
# # 展示实例图片
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(),cmap = 'gray')
# plt.title('%i'%train_data.train_labels[0])
# plt.show()
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_data = torchvision.datasets.MNIST(root='./mnist',train=False)
# 归一化
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:2000].cuda()
# 建立神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,# 输入层数
                out_channels=16,# 输出层数
                kernel_size=5,# 卷积核大小
                stride=1,# 步长
                padding=2# 填充
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 输入层数
                out_channels=32,  # 输出层数
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 填充
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32*7*7,10)# 全连接
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1) # 展开
        output = self.out(x)
        return output

cnn = CNN()
cnn.cuda()
# 定义优化方式（随机梯度下降）
optimizer = torch.optim.Adam(cnn.parameters(),lr = 0.01)

# 定义损失函数计算方式
loss_func = torch.nn.CrossEntropyLoss()# 这个自带softmax

# 训练
for epoch in range(EPOCH):
    for step,(batch_x,batch_y) in enumerate(train_loader):# enumerate枚举器，返回索引与内容
        b_x = Variable(batch_x).cuda()
        b_y = Variable(batch_y).cuda()
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()# 梯度设为零
        loss.backward()# 反向传递
        optimizer.step()# 优化梯度

        if step%50 == 0 :
            test_output = cnn(test_x)
            pred_y = torch.max(test_output,1)[1].cuda().data.squeeze()
            accuracy = (pred_y == test_y).sum().item()/test_y.size(0)
            print('\nEpoch:',epoch,'|train loss:%.4f'% loss.item(),'|test accuracy:',accuracy)

