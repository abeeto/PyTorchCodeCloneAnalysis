# -*- coding: utf-8 -*-
# weibifan 2022-10-1
# 构建图像模型：两层ReLU+随机数据

'''
第0步：数据和模型输入怎么匹配。

第1步：前向是怎么计算的。
①每个步骤做那些处理。参数是什么。
②步骤之间是怎么连接的。
③参数怎么初始化的。

第2步：损失函数怎么定义

第3步：反向传播是怎么计算的，也就是参数是怎么修改的。



'''
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 自定义一个神经网络的类，
# ①必须要继承自nn.Module
# ②必须要重载构造函数（__init__()）和前向传递函数（forward()）
# ③一般还重载参数初始化函数（init_weights()）

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 生成模型的对象，并将该对象加载到GPU中
model = NeuralNetwork().to(device)
print(model)

# 构建随机数据
X = torch.rand(1, 28, 28, device=device)

# 前向计算
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

#---------------------
input_image = torch.rand(3,28,28)  #图像的表示方法 RGB颜色，28*28像素
print(input_image.size())

#将28*28的矩阵，变成784的向量
flatten = nn.Flatten()  #用类生成一个对象。
flat_image = flatten(input_image)  #调用对象的构造函数__init__()
print(flat_image.size())

# 构造一个线性变换器，理论公式y=wx，其中x维度为784，y维度为20，w是20*784的矩阵
# flat_image 是 3* 784，实际公式 y=xw，此时 w为 784*20，最后hidden1是3*20
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# ReLU激活函数，y=ReLu(y)，按元素操作。输入是20维，输出也是20维
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# 构建一个神经网络模型的类
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10) #线性变换y=xw，w是20*10，从20维变成10维
)

input_image = torch.rand(3,28,28)
logits = seq_modules(input_image) #logits是矩阵，3 * 10


softmax = nn.Softmax(dim=1) # 1表示按列做归一化
pred_probab = softmax(logits)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")