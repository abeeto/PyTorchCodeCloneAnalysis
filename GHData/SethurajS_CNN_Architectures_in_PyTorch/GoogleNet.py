# Importing the requirements
import torch
import torch.nn as nn

# GoogleNet 
class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=1000):
        super(GoogleNet,self).__init__()

        self.conv1 = Conv_Block(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = Conv_Block(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception_3a = Inception_Block(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception_Block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception_4a = Inception_Block(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception_Block(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception_Block(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception_Block(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception_Block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception_5a = Inception_Block(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception_Block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024*10*10, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool_1(x)
        x = self.conv2(x)
        x = self.maxpool_2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool_3(x)

        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.maxpool_4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc(x)

        return x

# Inception Block
class Inception_Block(nn.Module):
    # Number of filters in each barnch -- {in_channel->out1x1}, {in_channel->reduce_3x3-->out_3x3}, {in_channel->reduce_5x5-->out_5x5}, {in_channel->Maxpool-->out_1x1}
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_out_1x1):
        super(Inception_Block, self).__init__()

        self.branch_1 = nn.Conv2d(in_channels, out_1x1, kernel_size=(1 ,1), stride=(1, 1), padding=(0, 0))

        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=(1, 1)),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=(3 ,3), padding=(1, 1))
            )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=(1, 1)),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2))
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels, pool_out_1x1, kernel_size=(1, 1))
        )
    
    def forward(self, x):
        # dim == [NumInputs, Channels, Height, Width] -- [0, 1, 2 ,3]
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], 1) # concat at dim=1 (Channels)

# Convolution Block
class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_Block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing GoogleNet
    model = GoogleNet(in_channels=3, out_classes=1000).to(device=device)
    x = torch.randn(3, 3, 512, 512).to(device=device)
    print(model(x).shape)
