from torch.autograd import Variable
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

class Img2Num(nn.Module):
    def __init__(self):
        self.num_classes = 10
        super(Img2Num, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Conv2d(16, 120, 5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = x.float()
        if len(x.size()) == 2:
            (H, W) = x.data.size()
            img = img.view(1, 1, H, W)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.fc1(x)
        (N,C,H,W) = x.size()
        x = x.view(N,-1)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x

    def train(self):
        self.loss_function = nn.MSELoss()
        # self.optimizer = optim.SGD(self.parameters(), lr=0.2)
        self.optimizer = optim.Adadelta(self.parameters())
        # Load MNIST
        root = 'torchvision/mnist/'
        download = True
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
        batch_size = 256
        train_loader = torch.utils.data.DataLoader(
                         dataset=train_set,
                         batch_size=batch_size,
                         shuffle=True)

        epoch = 1
        if epoch > 1:
            print("== Start training for {0:d} epochs".format(epoch))

        for i in range(epoch):
            # training
            batch_idx = 0
            for batch_idx, (x, target) in enumerate(train_loader):
                self.optimizer.zero_grad()

                x, target = Variable(x), Variable(Img2Num.oneHot(target, self.num_classes))
                x_pred = self.forward(x)
                loss = self.loss_function(x_pred, target)
                loss.backward()
                self.optimizer.step()
                if (batch_idx+1)% 100 == 0:
                    # print '==>>> batch index: {}, train loss: {:.6f}'.format(batch_idx, loss.data[0])
                    print '==>>> batch index: {}/{}'.format(batch_idx+1, len(train_loader))
            print '==>>> batch index: {}/{}'.format(batch_idx+1, len(train_loader))

            if epoch > 1:
                print("-- Finish epoch {0:d}".format(i+1))

    @staticmethod
    def oneHot(target, num_classes):
        # oneHot encoding
        label = []
        for l in target:
                label.append([1 if i==l else 0 for i in range(num_classes)])
        return torch.FloatTensor(label)
