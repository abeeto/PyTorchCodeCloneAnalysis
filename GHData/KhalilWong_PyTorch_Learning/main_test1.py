import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

################################################################################
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

################################################################################
def main_test1():
    net = Net()
    input = torch.randn(1, 1, 32, 32)
    target = torch.randn(10)  # 本例子中使用模拟数据
    target = target.view(1, -1)  # 使目标值与数据值尺寸一致
    print(target)
    #
    # 创建优化器(optimizer）
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # 在训练的迭代中：
    for i in range(1000):
        optimizer.zero_grad()   # 清零梯度缓存
        output = net(input)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # 更新参数
    print(loss.item())
    print(output)
    params = list(net.parameters())
    print(params[9])

################################################################################
if __name__ == '__main__':
    main_test1()
