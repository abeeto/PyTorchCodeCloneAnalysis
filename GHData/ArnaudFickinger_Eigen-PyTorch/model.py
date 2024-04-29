import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from options import Options

opt = Options().parse()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.coarse1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(2)
        self.coarse2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.coarse3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.coarse4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.coarse5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3)
        self.coarse6 = nn.Linear(in_features= 8 * 6 * 256, out_features=1 * 4096)  # change to size
        self.coarse7 = nn.Linear(in_features=1 * 4096, out_features=1 * 74 * 55)
        self.refined1 = nn.Conv2d(in_channels=3, out_channels=63, kernel_size=9, stride=2)
        self.refined2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.refined3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, x):
        out = F.relu(self.coarse1(x))
        out = F.relu(self.pool(out))
        out = F.relu(self.coarse2(out))
        out = F.relu(self.pool2(out))
        out = F.relu(self.coarse3(out))
        out = F.relu(self.coarse4(out))
        out = F.relu(self.coarse5(out))
        out = F.relu(self.pool3(out))
        out = out.view(-1,8*6*256)
        out = F.relu(self.coarse6(out))
        if opt.dropout:
            out = self.drop1(out)
        out = self.coarse7(out)
        out = out.reshape(-1,1,74,55)
        if opt.refined:
            out2 = F.relu(self.refined1(x))
            out2 = F.relu(self.pool4(out2))
            out = torch.cat((out, out2), 1)
            out = F.relu(self.refined2(out))
            out = self.refined3(out)
        return out