import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TripletNet(nn.Module):
    def __init__(self, embedding_dim=128, num_class=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 128, 2)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(5)
        # self.dropout = nn.Dropout2d()
        # self.fc = nn.Linear(128*5*5, embedding_dim)
        # self.classifier = nn.Linear(embedding_dim, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        # x = self.dropout(x)
        # x = x.view(-1, 128*5*5)
        # x = F.relu(self.fc(x))
        # x = self.classifier(x)

        return x
