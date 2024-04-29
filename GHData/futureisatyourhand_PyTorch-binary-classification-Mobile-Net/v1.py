import torch.nn as nn
class DepthPoint(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(DepthPoint,self).__init__()
        self.depth=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,stride,1,bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels,out_channels,1,1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    def forward(self,x):
        return self.depth(x)

class MobileNetV1(nn.Module):
    """
    standard conv->...->5*DepthPoint->DepthPoint(1024,1024,2)->DepthPoint(1024.1024,2)->adaptiveavg->fc
    """
    def __init__(self,num_classes,alpha):
        super(MobileNetV1,self).__init__()
        self.num_classes=num_classes
        self.conv1=nn.Sequential(
            nn.Conv2d(3,int(alpha*32),3,2,1,bias=False),
            nn.BatchNorm2d(int(alpha*32)),
            nn.ReLU6(inplace=True),
        )

        self.conv2=nn.Sequential(
            DepthPoint(int(alpha*32),int(alpha*64),1),
            DepthPoint(int(alpha*64),int(alpha*128),2),
            DepthPoint(int(alpha*128),int(alpha*128),1),
            DepthPoint(int(alpha*128),int(alpha*256),2),
            DepthPoint(int(alpha*256),int(alpha*256),1),
            DepthPoint(int(alpha*256),int(alpha*512),2),
            DepthPoint(int(alpha*512),int(alpha*512),1),
            DepthPoint(int(alpha*512),int(alpha*512),1),
            DepthPoint(int(alpha*512),int(alpha*512),1),
            DepthPoint(int(alpha*512),int(alpha*512),1),
            DepthPoint(int(alpha*512),int(alpha*512),1),
            DepthPoint(int(alpha*512),int(alpha*1024),2),
            DepthPoint(int(alpha*1024),10,2),
        )
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.linear=nn.Linear(10,num_classes)
        self.loss=nn.CrossEntropyLoss(reduce=True)
        self.softmax=nn.Softmax(-1)
    def forward(self,x,target=None):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.avg(x)
        x=x.view(x.shape[0],-1)
        x=self.linear(x)
        if target is None:
            return self.softmax(x)
        return self.loss(x,target)
