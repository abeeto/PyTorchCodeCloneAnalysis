import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TripletNet(nn.Module):
    def __init__(self, embedding_dim=128, num_class=9):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, 2)
        self.bn4 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(5)
        # self.dropout = nn.Dropout2d()
        # self.classifier = nn.Linear(embedding_dim, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.avgpool(x)
        # x = self.dropout(x)
        # x = x.view(-1, 128*5*5)
        # x = F.relu(self.fc(x))
        # x = self.classifier(x)

        return x


class TripletResNet(nn.Module):
    def __init__(self, metric_dim):
        super(TripletResNet, self).__init__()
        resnet = torchvision.models.__dict__['resnet18'](pretrained=True)
        for params in resnet.parameters():
            params.requires_grad = False

        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(resnet.fc.in_features, metric_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    # def forward(self, x):
    #     x = self.model(x)
    #     x = x.view(x.size(0), -1)
    #     # metric = self.fc(x)
    #     metric = F.normalize(self.fc(x))
    #     return metric
