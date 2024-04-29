import torch
import torch.nn as nn

#LeNet Architecture
#[1x32x32 input] -> [5x5 conv with s=1, p=0] -> [avg pool with s=2, p=0] -> [5x5 conv with s=1, p=0] ->
# [avg pool with s=2, p=0] -> [5x5 conv with output=120] -> [fully connected layer 84] -> [fully connected layer 10]

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.fc1 = nn.Linear(120,84)
        self.output = nn.Linear(84,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        # output of conv3 layer is num_examples x 120 x 1 x 1 but to proceed it to Linear layer we need to convert
        # the shape num_examples x 120
        x = x.reshape(x.shape[0], -1)  # the -1 will concatenate 120x1x1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output(x)
        return x
x = torch.randn(64, 1, 32, 32) # 64 images, channel=1, weight=32, height=32
model = LeNet()
print(model(x))
print(model(x).shape)