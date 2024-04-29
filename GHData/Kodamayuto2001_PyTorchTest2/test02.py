import torch
import torchvision
from torchvision import transforms
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F 

#データ変形
transform = transforms.Compose([transforms.ToTensor()])

#学習用データ読み込み
train = torchvision.datasets.MNIST(root="Resources/",train=True,download=False,transform=transform)

#モデルの構築
conv = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1)

x = train[0][0]

#入力層
x = x.reshape(1,1,28,28)

#畳み込み層
x = conv(x)

#プーリング層
x = F.max_pool2d(x,kernel_size=2,stride=2)

#全結合層 10クラス分類
fc = nn.Linear(x.shape[1]*x.shape[2]*x.shape[3],10)
x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
x = fc(x)

