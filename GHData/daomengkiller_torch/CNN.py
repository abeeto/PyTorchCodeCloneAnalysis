import torch
import torch.nn as nn
from torch.autograd import Variable  # 这个类可记录数据，并且记录creator，能计算梯度，对于BP神经网络，要用这种变量
import torch.utils.data as Data
import torchvision  # 拓展模块，数据库
import matplotlib.pyplot as plt  #

torch.manual_seed(1)  # 设置随机数种子

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False  # 标记是否下载了数据

train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 数据目录
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)  # 设置数据集
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)  # 设置测试数据集

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # 加载数据

# 设置只测试前2000
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 第一张图片的维度(1,28,28),1为图片的特征层，28*28为图片的长宽
            nn.Conv2d(
                in_channels=1,  # n_channels 为图片的特征层，例如灰度只有1层，RGB有3层，
                out_channels=16,  # out_channels 为输出的特征层，
                kernel_size=5,  # 扫描的区块
                stride=1,  # 两个扫描的区块间隔
                padding=2,  # 源图像的边沿补全
            ),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层，对图像进行，二次输出（16,14,14）
        )
        self.conv2 = nn.Sequential(  # 同上，（16,14,14）
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出（32,7,7）
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输入32*7*7，输出10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch,32,7,7)
        x = x.view(x.size(0), -1)  # 扁平化数据,(batch,32*7*7),将三维的数据变为二维数据结构
        output = self.out(x)
        return output, x


cnn = CNN()  # 生成网络
print(cnn)  # 打印网络
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 设置梯度优化函数
loss_func = nn.CrossEntropyLoss()  # 计算损失函数
# 以下是打印数据集的可视化
from matplotlib import cm

try:
    from sklearn.manifold import TSNE

    HAS_SK = True
except:
    HAS_SK = False
    print('please install sklearn for layer visualization')


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


plt.ion()
# 训练各种数据
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)  # 计算损失函数
        optimizer.zero_grad()  # 清空梯度缓冲区
        loss.backward()  # 方向传播计算
        optimizer.step()  # 更新网络的权重值

        if step % 1 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch:', epoch, '| train loss :%.4f' % loss.data[0], '| test accracy:%.2f' % accuracy)
            if HAS_SK:
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()  # 交互开关

test_output, _ = cnn(test_x[:10])  # 测试数据
# 将test_output的数据的各行为一个单位，返回每个单位的最大的数，形成队列
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()  # 输出结果

print(pred_y, 'predicton number')  # 预测数值
print(test_y[:10].numpy(), 'real number')  # 真实结果
