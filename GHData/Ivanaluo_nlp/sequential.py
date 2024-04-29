import torch
from torch import nn
import torch.nn.functional as F

######################################### 写法一：堆叠torch.nn常规写法 #################################
class Net(nn.Module):
	def __init__(self):
		super().__init__()

		self.fc1 = nn.Linear(784, 256)
		self.fc2 = nn.Linear(256, 10)

		self.sigmoid = nn.Sigmoid()             # 看这里
		self.softmax = nn.softmax(dim=1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.sigmoid(x)                     # 看这里
		x = self.fc2(x)
		x = self.softmax(x)

		return x

net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
net.to(device)    # 设置单gpu
logit = net(image)

###################################### 写法二：前向使用torch.nn.functional函数 ##############################
class Net(nn.Module):
	def __init__(self):
		super().__init__()

		self.fc1 = nn.Linear(784, 256)
		self.fc2 = nn.Linear(256, 10)

	def forward(self, x):
		x = F.sigmoid(self.fc1(x))               # 看这里
		x = F.softmax(self.fc2(x), dim = 1)

		return x

###################################### 写法三：前向使用torch.nn.functional函数 ##############################
net = nn.Sequential(
          nn.Linear(784, 256),
          nn.Sigmoid() ,
          nn.Linear(256, 10),
          nn.softmax(dim=1)
        )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
net.to(device)    
logit = net(image)




