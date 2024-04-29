import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST


class net(nn.Module):

    def __init__(self, num_fe_layers=2, num_class=2):
        super(net, self).__init__()

        # 1. Dummy
        # self.conv1 = nn.Conv2d(1, 32, 5)
        # self.conv2 = nn.Conv2d(32, 64, 5)
        # self.lc = nn.Linear(64, 10)

        # 2. Sequential
        # self.model = nn.Sequential(
        #    nn.Conv2d(1, 32, 5),
        #    nn.AdaptiveMaxPool2d(),
        #    nn.ReLU(),
        #    nn.Conv2d(1, 32, 5),
        #    nn.AdaptiveMaxPool2d(),
        #    nn.ReLU(),
        #    nn.Linear(64, 10)
        # )

        # 3. ModuleList
        self.feature_extractor = nn.ModuleList()
        self.classifier = nn.ModuleList()

        for _ in range(num_fe_layers):  # feature_extractor
            self.feature_extractor.append(
                nn.Conv2d(1, 32, 5),
                nn.AdaptiveMaxPool2d(),
                nn.ReLU()
            )

        for _ in range(1):  # classifier
            self.classifier.append(
                nn.Linear(32, 10)
            )

        # 4.
        # self.add_module()

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 3))
        # x = F.relu(F.max_pool2d(self.conv2(x), 3))
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.lc(x))

        # 2
        # x = self.model(x)

        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = net()
    # optmizer
    # loss
    print(model)

    for epoch in range(100):
        # read batch from mnist
        # y = model(x)
        # loss
        # optimizer.step()
        pass

    print("Done")
