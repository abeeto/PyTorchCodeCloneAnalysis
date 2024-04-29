import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

#定义一个基于convnet的卷积神经网络

#1 加载数据
torch.manual_seed(1234)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = test_batch_size = 32
kwargs ={'num_workers':1,'pin_memory':True} if use_cuda else {}

# torch.utils.data.DataLoader在训练模型时使用到此函数，用来把训练数据分成多个batch，
# 此函数每次抛出一个batch数据，直至把所有的数据都抛出，也就是个数据迭代器。
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data',
                   train=True,  # 如果true，从training.pt创建数据集
                   download=True,  # 如果ture，从网上自动下载
                   # transform 接受一个图像返回变换后的图像的函数，相当于图像先预处理下
                   # 常用的操作如 ToTensor, RandomCrop，Normalize等.
                   # 他们可以通过transforms.Compose被组合在一起
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # .ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，
                       # 其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可。
                       transforms.Normalize((0.1307,), (0.3081,))  # 所有图片像素均值和方差
                       # .Normalize作用就是.ToTensor将输入归一化到(0,1)后，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
                   ])),  # 第一个参数dataset：数据集
    batch_size=batch_size,
    shuffle=True,  # 随机打乱数据
    **kwargs)  ##kwargs是上面gpu的设置

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data',
                   train=False, #如果False，从test.pt创建数据集
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size,
    shuffle=True,
    **kwargs)

#2 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)
        # in_channels：输入图像通道数，手写数字图像为1，彩色图像为3
        # out_channels：输出通道数，这个等于卷积核的数量
        # kernel_size：卷积核大小
        # stride：步长
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # 上个卷积网络的out_channels，就是下一个网络的in_channels，所以这里是20
        # out_channels：卷积核数量50
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # 全连接层torch.nn.Linear(in_features, out_features)
        # in_features:输入特征维度，4*4*50是自己算出来的，跟输入图像维度有关
        # out_features；输出特征维度
        self.fc2 = nn.Linear(500, 10)
        # 输出维度10，10分类

    def forward(self, x):
        # print(x.shape)  #手写数字的输入维度，(N,1,28,28), N为batch_size
        x = F.relu(self.conv1(x))  # x = (N,50,24,24)
        x = F.max_pool2d(x, 2, 2)  # x = (N,50,12,12)
        x = F.relu(self.conv2(x))  # x = (N,50,8,8)
        x = F.max_pool2d(x, 2, 2)  # x = (N,50,4,4)
        x = x.view(-1, 4 * 4 * 50)  # x = (N,4*4*50)
        x = F.relu(self.fc1(x))  # x = (N,4*4*50)*(4*4*50, 500)=(N,500)
        x = self.fc2(x)  # x = (N,500)*(500, 10)=(N,10)
        return F.log_softmax(x, dim=1)  # 带log的softmax分类，每张图片返回10个概率

#3 初始化模型 定义损失函数
lr = 0.01
momentun = 0.5
model = Net()
model = model.to(device)
optimizer = optim.SGD(model.parameters(),lr=lr,momentun=momentun)

#4 定义训练和测试函数
def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()  # 进入训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度归零
        output = model(data)  # 输出的维度[N,10] 这里的data是函数的forward参数x
        loss = F.nll_loss(output, target)  # 这里loss求的是平均数，除以了batch
        # F.nll_loss(F.log_softmax(input), target) ：
        # 单分类交叉熵损失函数，一张图片里只能有一个类别，输入input的需要softmax
        # 还有一种是多分类损失函数，一张图片有多个类别，输入的input需要sigmoid

        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),  # 100*32
                len(train_loader.dataset),  # 60000
                100. * batch_idx / len(train_loader),  # len(train_loader)=60000/32=1875
                loss.item()
            ))
            # print(len(train_loader))


def test(model, device, test_loader):
    model.eval()  # 进入测试模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # reduction='sum'代表batch的每个元素loss累加求和，默认是mean求平均
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # print(target.shape) #torch.Size([32])
            # print(pred.shape) #torch.Size([32, 1])
            correct += pred.eq(target.view_as(pred)).sum().item() #对了几个
            # pred和target的维度不一样
            # pred.eq()相等返回1，不相等返回0，返回的tensor维度(32，1)。
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#5 开始训练
epochs = 2
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

save_model = True
if (save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")
    #词典格式，model.state_dict()只保存模型参数