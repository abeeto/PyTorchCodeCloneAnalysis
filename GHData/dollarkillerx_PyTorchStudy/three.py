import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self): #定义网络结构    输入数据  1*32*32  
        super(Net,self).__init__()
        # 定义第一层 (卷积层)
        self.conv1 = nn.Conv2d(1,6,3) # 定义2维卷积层  输入频道1  输出频道6 卷积3*3
        # 定义第二层 (卷积层)
        self.conv2 = nn.Conv2d(6,16,3) # 输入频道6 输出频道16  卷积3*3
        # 第三层 (全连接层)
        self.fc1 = nn.Linear(16*28*28,512) # 全连接层输入为1*n的向量     输入维度16*28*28=12544  输出维度 512
            # 16为上一层的输入  28是原始输入数据 32每经过一层数据就会减2    
        # 第四层 (全连接层)
        self.fc2 = nn.Linear(512,64) # 输入维度512，输出维度64
        # 第五层 (全连接层)
        self.fc3 = nn.Linear(64,2) # 输入维度64 ， 输出维度2
    def forward(self,x): # 定义数据流向
        x = self.conv1(x) 
        x = F.relu(x)  # 隐藏层  激活函数 激活
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = x.view(-1,16*28*28)  # 更改形状
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x

net = Net()
print(net)

input_data = torch.randn(1,1,32,32) # 生成随机数据
print(input_data)
print(input_data.size())


# 运行网络
out = net(input_data)
print(out)
print(out.size())

# 随机生成真实值 
target = torch.randn(2)
target = target.view(1,-1)
print(target)


# 计算误差
criterion = nn.L1Loss()  # 绝对误差的平均值
loss = criterion(out,target) 
print(loss)

# 反向传递 
net.zero_grad()  # 清零之前的梯度
loss.backward()  # 计算梯度，方向传递


# 更新权重
import torch.optim as optim
optimizer = optim.SGD(net.parameters(),lr=0.01) # 更新所有的paramteters,lr 学习速率
optimizer.step()


# 重写测试
out = net(input_data)
print(out)
print(out.size())

# 再次计算损失
loss = criterion(out,target) 
print(loss)   # 多次的计算损失 和多次的反向传递  会让权重越来越准确