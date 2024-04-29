import torch
from torch import nn
import torch.nn.functional as F

######################################### 写法一：堆叠torch.nn常规写法 #################################
class Net(nn.Module):
	def __init__(self):
		super().__init__()

		self.fc1 = nn.Linear(784, 256)
		self.fc2 = nn.Linear(256, 10)

		self.sigmoid = nn.Sigmoid()             
		self.dropout = nn.Dropout(p = 0.2)      # 看这里
		self.softmax = nn.softmax(dim=1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.sigmoid(x)          
		x = self.dropout(x)                     # 看这里
		x = self.fc2(x)
		x = self.softmax(x)

		return x

# 注意Dropout 在训练时采用，是为了减少神经元对部分上层神经元的依赖，类似将多个不同网络结构的模型集成起来，减少过拟合的风险。
# 而在测试时，应该用整个训练好的模型，因此不需要dropout。
# 训练/校验/测试具体实现可见train_v_t.py
net = Net()
# turn off gradients
with torch.no_grad():
	net.eval()  # set model to evaluation mode
	for images, label in testloader:  # validation pass here
		pass
# set model back to train mode
model.train()  

###################################### 写法二：前向使用torch.nn.functional函数 ##############################
class Net(nn.Module):
	def __init__(self):
		super().__init__()

		self.fc1 = nn.Linear(784, 256)
		self.fc2 = nn.Linear(256, 10)

	def forward(self, x):
		x = F.sigmoid(self.fc1(x))               
		x = F.dropout(x, p = 0.5, training = self.training)      # 看这里 设置training参数
		x = F.softmax(self.fc2(x), dim = 1)

		return x


###################################### 写法三：前向使用torch.nn.functional函数 ##############################
net = nn.Sequential(
          nn.Linear(784, 256),
          nn.Dropout(0.5),  # drop 50% of the neuron
          nn.Sigmoid() ,
          nn.Linear(256, 10),
          nn.softmax(dim=1)
        )










