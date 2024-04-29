import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5))
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5))

        self.linear1 = nn.Linear(in_features=16 * 5 * 5,out_features=120)
        self.linear2 = nn.Linear(in_features=120,out_features=60)
        self.linear3 = nn.Linear(in_features=60,out_features=10)


    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        x = x.view(-1,self.num_flat_features(x))

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)  #最后一个不激活

        return x


    def num_flat_features(self,x):
        size = x.size()[1:] #except batch dimension

        num_features = 1;
        for s in size:
            num_features *= s  #各个feature map对应点相乘，得到单层feature map

        return num_features


# net = Net()
# input = torch.rand(1,1,32,32)
# out = net(input)
# # net.zero_grad() #所有梯度缓存器设置为0
# # out.backward(torch.rand(1,10)) #随机梯度反向传播  这个是随机反向传播，下面是loss损失
#
#
#
# target = torch.rand(1,10)
# criterion = nn.MSELoss()
# loss = criterion(out,target)
#
#
#
#
# '''
# 典型更新规则 weight = weight - learning_rate * gradient
# '''
# # net.zero_grad()
# # print(net.conv1.bias.grad)
# # loss.backward() #反向传播损失
# # print(net.conv1.bias.grad)
# # learning_rate = 0.001
# # for f in net.parameters():
# #     f.data.sub_(learning_rate * f.grad) #f.grad,梯队，反向传播回来了！
#
#
# optimizer = optim.Adam(net.parameters(),lr=0.001)  #其他更新规则，Adam
# optimizer.zero_grad()
# loss.backward()
# optimizer.step() #update!!

'''
完整流程========================================================================================================================
'''

net = Net()
input = torch.rand(1,1,32,32)
out = net(input)

print(net.conv1.bias.grad)
criterion = nn.MSELoss()
target = torch.rand(1,10)
loss = criterion(target,out)

optimizer = optim.SGD(net.parameters(),lr=0.001)
optimizer.zero_grad()  #清空梯度缓存。梯度缓存在optimizer里，之前是net里，因为这里用了optimizer
loss.backward()
optimizer.step() #update

print(net.conv1.bias.grad)

'''
end=============================================================================================================================
'''