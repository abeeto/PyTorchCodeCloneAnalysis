# Import Dependencies
import math
import torch
import torch.nn as nn


# Fire Module - Squeeze & Expand Layers
class FireModule(nn.Module):
    def __init__(self, in_planes, squeeze_planes, expand_planes):
        """

        :param in_planes: number of input channels
        :param squeeze_planes: number of intermediate channels
        :param expand_planes: number of output channels
        """

        super(FireModule, self).__init__()

        # Squeeze Layer - 1 x 1 Convolution Layer (Pointwise Convolution)
        self.squeeze1x1 = nn.Conv2d(in_channels=in_planes, out_channels=squeeze_planes, kernel_size=(1, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(squeeze_planes)

        # Excite Layer - (1x1 Convolution, 3x3 Convolution)
        # 1x1 Convolution
        self.expand1x1 = nn.Conv2d(in_channels=squeeze_planes, out_channels=expand_planes, kernel_size=(1, 1),
                                   stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(expand_planes)

        # 3x3 Convolution
        self.expand3x3 = nn.Conv2d(in_channels=squeeze_planes, out_channels=expand_planes, kernel_size=(3, 3),
                                   stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # Initialize Convolutional Layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # Input - Squeeze Layer
        x = self.squeeze1x1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Expand Layer
        # 1x1 Convolution
        out_1 = self.expand1x1(x)
        out_1 = self.bn2(out_1)

        # 3x3 Convolution
        out_2 = self.expand3x3(x)
        out_2 = self.bn3(out_2)

        # Before sending the output, the outputs of expand layers is concatenated
        out = torch.cat([out_1, out_2], dim=1)
        out = self.relu(out)

        return out


# SqueezeNet Model Class
class SqueezeNet(nn.Module):
    def __init__(self, in_ch, num_classes):
        """

        :param in_ch: Number of channels in Input Image
        :param num_classes: Number of Output Classes
        """

        super(SqueezeNet, self).__init__()

        self.num_classes = num_classes

        # Input Conv1 Layer
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=96, kernel_size=(3, 3), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)

        # MaxPool 1
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        # Fire Module - 2
        self.fire2 = FireModule(in_planes=96, squeeze_planes=16, expand_planes=64)

        # Fire Module - 3
        self.fire3 = FireModule(in_planes=128, squeeze_planes=16, expand_planes=64)

        # Fire Module - 4
        self.fire4 = FireModule(in_planes=128, squeeze_planes=32, expand_planes=128)

        # MaxPool 4
        self.max_pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fire Module - 5
        self.fire5 = FireModule(in_planes=256, squeeze_planes=32, expand_planes=128)

        # Fire Module - 6
        self.fire6 = FireModule(in_planes=256, squeeze_planes=48, expand_planes=192)

        # Fire Module - 7
        self.fire7 = FireModule(in_planes=384, squeeze_planes=48, expand_planes=192)

        # Fire Module - 8
        self.fire8 = FireModule(in_planes=384, squeeze_planes=64, expand_planes=256)

        # MaxPool 8
        self.max_pool8 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fire Module - 9
        self.fire9 = FireModule(in_planes=512, squeeze_planes=64, expand_planes=256)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Conv10 Layer
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=(1, 1), stride=(1, 1))

        # Average Pool Layer
        self.avg_pool = nn.AvgPool2d(kernel_size=(13, 13), stride=(1, 1))

        # Use Adaptive Average Pooling for random input image size
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Output Activation
        self.softmax = nn.LogSoftmax(dim=1)

        # Xavier Initialization for Conv layer as per paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Input Layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Fire Module 2
        x = self.fire2(x)

        # Fire Module 3
        x = self.fire3(x)

        # Fire Module 4
        x = self.fire4(x)

        # MaxPool
        x = self.max_pool(x)

        # Fire Module 5
        x = self.fire5(x)

        # Fire Module 6
        x = self.fire6(x)

        # Fire Module 7
        x = self.fire7(x)

        # Fire Module 8
        x = self.fire8(x)

        # MaxPool
        x = self.max_pool(x)

        # Fire Module 9
        x = self.fire9(x)
        x = self.dropout(x)

        # Conv10
        x = self.conv10(x)
        x = self.relu(x)

        # Global Avg Pool
        x = self.avg_pool(x)

        # Softmax Activation
        x = self.softmax(x)

        return torch.flatten(x, start_dim=1)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Test Fire Module
    fire = FireModule(in_planes=256, squeeze_planes=48, expand_planes=192)

    # Test input
    x = torch.rand((256, 512, 512))
    fire_out = fire(torch.unsqueeze(x, dim=0))
    print("fire_out.shape: ", fire_out.shape)

    model = SqueezeNet(in_ch=3, num_classes=10)
    model.to(device)

    # Test Model
    a = torch.rand(1, 3, 224, 224)
    out = model(a.to(device))
    print("model_output.shape: ", out.shape)
