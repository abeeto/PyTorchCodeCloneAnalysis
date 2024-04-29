import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Сверточный слой
        # i --> input channels
        # 6 --> output channels
        # 5 --> kernel size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Полностью связанный слой
        # 16 * 4 * 4 --> input vector dimensions
        # 120 --> output vector dimensions
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Convolution-> ReLu-> Pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        # reshape, «-1» означает адаптивный
        # x = (n * 16 * 4 * 4) --> n : input channels
        # x.size()[0] == n --> input channels
        x = x.view(x.size()[0], -1)

        # Полностью связанный слой
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x