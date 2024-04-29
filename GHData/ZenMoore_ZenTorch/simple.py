import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sim_config as config


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 注意，2D卷积层的输入data维数是 batchsize*channel*height*width
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def criterion(x, y):
    return nn.MSELoss(x, y)


net = Net()
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# in your training loop:
for i in range(config.num_iteations):
    optimizer.zero_grad() # zero the gradient buffers，如果不归0的话，gradients会累加

    output = net(config.input_data) # 这里就体现出来动态建图了，你还可以传入其他的参数来改变网络的结构

    loss = criterion(output, config.target)
    loss.backward() # 得到grad，i.e.给Variable.grad赋值
    optimizer.step() # Does the update，i.e. Variable.data -= learning_rate*Variable.grad
    print(loss.eval())