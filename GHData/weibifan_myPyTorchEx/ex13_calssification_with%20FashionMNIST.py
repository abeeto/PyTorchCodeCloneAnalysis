# -*- coding: utf-8 -*-
# weibifan 2022-10-12
# 一个完整的图像分类案例，两层ReLU + FashionMNIST
# quickstar

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets  #需要额外安装
from torchvision.transforms import ToTensor

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 第1步：现在训练数据和测试数据。
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 第2步：构建batch迭代指针。基于第1步。
# 第1种方法：先将数据加载到datasets中，然后基于datasets对象构建batch指针。

batch_size = 64
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 浏览数据情况。
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 第3步：定义模型，继承自nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  #将图像的矩阵变成向量

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  #线性变换 o1=x0 * w1+b1
            nn.ReLU(), # x1=foo(o1)
            nn.Linear(512, 512), #线性变换，o2=x1 * w2 +b2
            nn.ReLU(), # x2=foo(o2)
            nn.Linear(512, 10) # o3=x2 * w3 +b3
        )

    def forward(self, x):
        x = self.flatten(x) # 将图像的矩阵变成向量
        logits = self.linear_relu_stack(x) #顺序执行
        return logits

model = NeuralNetwork().to(device)
print(model)

# 第4步：定义损失函数
loss_fn = nn.CrossEntropyLoss()

#  第5步：定义梯度下降方法
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 用训练集更新模型参数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # 将数据搬到GPU mem

        # Compute prediction error
        pred = model(X) #获得前向结果
        loss = loss_fn(pred, y) # 计算损失

        # Backpropagation
        optimizer.zero_grad() #设置梯度为0，也就是参数为0
        loss.backward() # 计算梯度
        optimizer.step() # 更新权重

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 用测试集测试一下，看看性能如何。
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 第6步：训练模型，同时观察测试误差变化情况
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# 第7步：保存模型
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# 第8步：加载模型
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval() #进入评估模式，不反向传播
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')