import torch 
import torch.nn as nn 
from torchvision.transforms import functional as tfunc

class convblock(nn.Module):
    def __init__(self, inchannels, outchannels, kernelsize, stride, padding):
        super().__init__()
        self.seq = nn.Sequential(
                    nn.Conv2d(inchannels, outchannels, kernelsize, stride, padding, bias=False),
                    nn.BatchNorm2d(outchannels),
                    nn.ReLU(inplace=True)
                )
    def forward(self, x):
        return self.seq(x)
#Stem
class STEM(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.seq1 = nn.Sequential(
                    convblock(inchannels, 32, 3, 2, 0), 
                    convblock(32, 32, 3, 1, 0), 
                    convblock(32, 64, 3, 1, 1), 
                )

        self.conv0 = convblock(64, 96, 3, 2, 0)
        self.mp = nn.MaxPool2d(3, 2, 0)

        self.seq2 = nn.Sequential(
                    convblock(160, 64, 1, 1, 1),
                    convblock(64, 64, (7, 1), 1, 1), 
                    convblock(64, 64, (1, 7), 1, 1),  
                    convblock(64, 96, 3, 1, 0), 
                )
        self.seq3 = nn.Sequential(
                    convblock(160, 64, 1, 1, 0), 
                    convblock(64, 96, 3, 1, 0), 
                )
        self.conv1 = convblock(192, 192, 3, 2, 0)
    def forward(self, x):
        x = self.seq1(x)
        x = torch.cat((self.conv0(x), self.mp(x)), 1) 
        x = torch.cat((self.seq2(x), self.seq3(x)), 1)
        x = torch.cat((self.conv1(x), self.mp(x)), 1)
        return x 

#inception module A
class IM_A(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.seq1 = nn.Sequential(
                    convblock(inchannels, 64, 1, 1, 0), 
                    convblock(64, 96, 3, 1, 1),
                    convblock(96, 96, 3, 1, 1)
                )
        self.seq2 = nn.Sequential(
                    convblock(inchannels, 64, 1, 1, 0),
                    convblock(64, 96, 3, 1, 1)
                )
        self.conv0 = convblock(inchannels, 96, 1, 1, 0)
        self.seq3 = nn.Sequential(
                    nn.AvgPool2d((1,1)),
                    convblock(inchannels, 96, 1, 1, 0)
                )
    def forward(self, x):
        x = torch.cat((self.seq1(x), self.seq2(x), self.conv0(x), self.seq3(x)), 1)
        return x

#inception module B
class IM_B(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.seq1 = nn.Sequential(
                    convblock(inchannels, 192, 1, 1, 2), 
                    convblock(192, 192, (1, 7), 1, 1), 
                    convblock(192, 224, (7, 1), 1, 1),
                    convblock(224, 224, (1, 7), 1, 1),
                    convblock(224, 256, (7, 1), 1, 1), 
                )
        self.seq2 = nn.Sequential(
                    nn.Conv2d(inchannels, 192, 1, 1, 1), 
                    convblock(192, 224, (1, 7), 1, 1),
                    convblock(224, 256, (7, 1), 1, 1),
                )
        self.conv0 = convblock(inchannels, 384, 1, 1, 0)
        self.seq3 = nn.Sequential(
                    nn.AvgPool2d((1,1)),
                    convblock(inchannels, 128, 1, 1, 0)
                )
    def forward(self, x):
        x = torch.cat((self.seq1(x), self.seq2(x), self.conv0(x), self.seq3(x)), 1) 
        return x

#inception module C
class IM_C(nn.Module):
    def __init__(self, inchannels):
        super().__init__()

        self.seq1 = nn.Sequential(
                    convblock(inchannels, 384, 1, 1, 0), 
                    convblock(384, 448, (1, 3), 1, 1),
                    convblock(448, 512, (3, 1), 1, 1), 
                )
        self.conv0 = convblock(512, 256, (3, 1), 1, 0)
        self.conv1 = convblock(512, 256, (1, 3), 1, 0)

        self.conv2 = convblock(inchannels, 384, 1, 1, 0)      
        self.conv3 =  convblock(384, 256, (3, 1), 1, 1)
        self.conv4 =  convblock(384, 256, (1, 3), 1, 1)


        self.conv5 = convblock(inchannels, 256, 1, 1, 0)
        
        self.seq2 = nn.Sequential(
                    nn.AvgPool2d((1,1)),
                    convblock(inchannels, 256, 1, 1, 0)
                )

    def forward(self, x):
        seqout = self.seq1(x)
        conv2out = self.conv2(x)
        
        out1 = torch.cat((self.conv0(seqout), self.conv3(conv2out)), 1)
        out2 = torch.cat((self.conv1(seqout), self.conv4(conv2out)), 1)
        out = torch.cat((tfunc.resize(out1, (8,8)), tfunc.resize(out2, (8,8)), self.conv5(x), self.seq2(x)), 1)

        return out

#Reduction A
# k = 192, l = 224, m = 256, n = 384 
class IR_A(nn.Module):
    def __init__(self, inchannels, k = 192, l = 224, m = 256, n = 384):
        super().__init__()
        self.seq1 = nn.Sequential(
                    convblock(inchannels, k, 1, 1, 0),
                    convblock(k, l, 3, 1, 1),
                    convblock(l, m, 3, 2, 0)
                )
        self.conv0 = convblock(inchannels, n, 3, 2, 0)
        self.mp = nn.MaxPool2d(3, 2, 0)
    def forward(self, x):
        x = torch.cat((self.seq1(x), self.conv0(x), self.mp(x)), 1) 
        return x

#Reduction B 
class IR_B(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.seq1 = nn.Sequential(
                    convblock(inchannels, 256, 1, 1, 1),
                    convblock(256, 256, (1,7), 1, 1),
                    convblock(256, 320, (7,1), 1, 1),
                    convblock(320, 320, 3, 2, 0)
                )
        self.seq2 = nn.Sequential(
                    convblock(inchannels, 192, 1, 1, 0),
                    convblock(192, 192, 3, 2, 0)
                )
        self.mp = nn.MaxPool2d(3, 2, 0) 

    def forward(self, x):
        x = torch.cat((self.seq1(x), self.seq2(x), self.mp(x)), 1) 
        return x

#Inception V4
class InceptionV4(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        
        self.inceptionv4 = nn.Sequential(
                   STEM(inchannels),
                   *[IM_A(384) for i in range(4)],
                   IR_A(384), 
                   *[IM_B(1024) for i in range(7)],
                   IR_B(1024),
                   *[IM_C(1536) for i in range(3)],
                )

        self.avgpool = nn.AvgPool2d((8,8))
        
        self.classfier = nn.Sequential(
                   nn.Flatten(),
                   nn.Linear(1536, outchannels),
                   nn.Softmax(dim=1)
                )
    
    def forward(self, x):
        return self.classfier(self.avgpool(self.inceptionv4(x)))
