import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#VGG MODEL Architecture
from torch.version import cuda

VGG16= [64,64,'Maxpooling', 128,128,'Maxpooling', 256,256,256,'Maxpooling', 512,512,512,'Maxpooling',512,512,512,'Maxpooling']
# Then flatten and 4096x4096x1000 Layers

#Other VGG architectures
VGG11 = [64,'Maxpooling', 128,'Maxpooling', 256,256,'Maxpooling', 512,512,'Maxpooling',512,512,'Maxpooling']
VGG13 = [64,64,'Maxpooling', 128,128,'Maxpooling', 256,256,'Maxpooling', 512,512,'Maxpooling',512,512,'Maxpooling']
VGG19 = [64,64,'Maxpooling', 128,128,'Maxpooling', 256,256,256,256,'Maxpooling', 512,512,512,512,'Maxpooling',512,512,512,512,'Maxpooling']




class VGGNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGGNet,self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layer(VGG16)

        self.fc = nn.Sequential(
            nn.Linear(512*7*7,4096),#224(image size) / (2**5(number of maxpooling) = 224/32 = 7
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes))

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:

                out_channels = x

                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU()]

                in_channels = x

            elif type(x) == str:
                # out_channels = x
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VGGNet(in_channels=3,num_classes=1000).to(device)
x = torch.randn(1, 3, 224,224).to(device)
print(model(x).shape)
