import torch.nn as nn
import torch.nn.functional as F

class G(nn.Module):

    def __init__(self, nz, ngf, nc):
        super(G, self).__init__()
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ngf * 2)

        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ngf * 1)

        self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.__initialize_weights()

    def forward(self, input):
        x = input.view(input.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        output = self.tanh(x)
        return output

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    
class D(nn.Module):

    def __init__(self, ndf, nc, num_classes=10):
        super(D, self).__init__()
        self.ndf = ndf
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0)
        self.gan_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()
    
    def forward(self, input):

        x = self.conv1(input)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)

        x = self.conv5(x)
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)

        s = self.gan_linear(x)
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
