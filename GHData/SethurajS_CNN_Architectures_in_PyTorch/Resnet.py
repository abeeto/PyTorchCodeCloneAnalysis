# Importing the requirements
import torch
import torch.nn as nn
import sys

# Convolution Block
class Block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()

        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=(1, 1),
                               stride=(1, 1), padding=(0, 0))
        self.batchnorm1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels, kernel_size=(3, 3),
                               stride=stride, padding=(1, 1))
        self.batchnorm2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels * self.expansion, kernel_size=(1, 1),
                               stride=(1, 1), padding=(0, 0))
        self.batchnorm3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x

# Resnet Block     
class ResNet(nn.Module):
    def __init__(self, block, layers, img_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        
        self.layer_1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )

        self.layer_2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )

        self.layer_3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )

        self.layer_4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(2048*7*7, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x 

    def _make_layer(self, block, num_res_block, intermediate_channels, stride):
        identity_sample = None
        layers =[]

        if stride != 1 or self.in_channels != intermediate_channels*4:
            identity_sample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=intermediate_channels*4, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(intermediate_channels*4)
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_sample, stride)
        )

        self.in_channels = intermediate_channels * 4

        for _ in range(num_res_block - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ResNet50(img_channel=3, num_classes=1000):
        return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)


    def ResNet101(img_channel=3, num_classes=1000):
        return ResNet(Block, [3, 4, 23, 3], img_channel, num_classes)


    def ResNet152(img_channel=3, num_classes=1000):
        return ResNet(Block, [3, 8, 36, 3], img_channel, num_classes)


    def Resnet(model_name):
        if model_name == "Resnet50":
            print("\n>>> Model Using -- Resnet50\n")
            model = ResNet50(img_channel=3, num_classes=1000)
        elif model_name == "Resnet101":
            print("\n>>> Model Using -- Resnet101\n")
            model = ResNet101(img_channel=3, num_classes=1000)
        elif model_name == "Resnet152":
            print("\n>>> Model Using -- Resnet152\n")
            model = ResNet152(img_channel=3, num_classes=1000)
        else:
            print("\n>>> Invalid Model !!\n    Enter only one of these ( Resnet50 | Resnet101 | Resnet152 ).\n")
            sys.exit()

        model.to(device=device)
        x = torch.randn(4, 3, 224, 224).to(device=device)
        print(">>> "+str(model(x).shape))
    
    model = str(input("\n>>> Enter the model name ( Resnet50 | Resnet101 | Resnet152 ): "))
    Resnet(model_name=model)