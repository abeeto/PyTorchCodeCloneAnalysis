"""
file - network.py
定义神经网络
"""

import torch
from torch import nn
from torchvision.models import vgg16, densenet121, convnext_tiny


# NIMA_Vgg16 have 138608434 paramerters in total
class NIMA_Vgg16(nn.Module):
    """Neural IMage Assessment model by Google"""

    def __init__(self, num_classes=10):
        super(NIMA_Vgg16, self).__init__()
        self.vgg = vgg16(pretrained=True)
        self.features = self.vgg.features
        self.avgpool = self.vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# NIMA_Dense121 have 7989106 paramerters in total
class NIMA_Dense121(nn.Module):
    def __init__(self, num_classes=10):
        super(NIMA_Dense121, self).__init__()
        self.densenet = densenet121(pretrained=True)
        self.features = self.densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=1024, out_features=num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# NIMA_ConvNeXt() have 28596818 paramerters in total
class NIMA_ConvNeXt(nn.Module):
    def __init__(self, num_classes=10):
        super(NIMA_ConvNeXt, self).__init__()
        self.convnext = convnext_tiny(pretrained=True)
        self.features = self.convnext.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            self.convnext.classifier[0],
            self.convnext.classifier[1],
            nn.Linear(in_features=768, out_features=num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.classifier(out)
        return out


NIMA_Dict = {
    'vgg16': NIMA_Vgg16(),
    'densenet121': NIMA_Dense121(),
    'convnext': NIMA_ConvNeXt()
}

# 调试用
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler,SGD
if __name__ == '__main__':
    a = torch.tensor([0.005, 0.017, 0.048, 0.155, 0.326, 0.261, 0.114, 0.049, 0.017, 0.009]).reshape(1,10)
    b = torch.arange(start=1, end=11, step=1, dtype=torch.float32).reshape(10, 1)
    c = a@b
    print(b, c)
    print(a.shape)

    d = a @ (b-c)**2
    dd = torch.sqrt(d)
    print(d, dd)

    # writer = SummaryWriter('./logs/nets/NIMA_ConvNeXt')
    #
    # image = torch.randn(7, 3, 224, 256)
    # net = NIMA_ConvNeXt()
    # writer.add_graph(net, image)
    # writer.close()

    # net = NIMA_Dict['convnext']
    # print(net)
    # image = torch.randn(7, 3, 224, 256)
    # out = net(image)
    # print(out.shape)
    # print("NIMA_ConvNeXt have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))

    # optimizer = SGD([
    #      {'params': net.features.parameters(), 'lr': 5e-3},
    #      {'params': net.classifier.parameters(), 'lr': 5e-4}
    # ],
    #     momentum=0.9)
    # print(optimizer.param_groups[0]['lr'])
    # print(optimizer.param_groups[1]['lr'])
    # writer = SummaryWriter('./losg/lr')
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8)
    # for epoch in range(100):
    #     optimizer.step()
    #     print('=-'*30)
    #     print(optimizer.param_groups[0]['lr'])
    #     print(optimizer.param_groups[1]['lr'])
    #     scheduler.step()
    #
    #     writer.add_scalar('lr1', optimizer.param_groups[0]['lr'], global_step=epoch)
    #     writer.add_scalar('lr2', optimizer.param_groups[1]['lr'], global_step=epoch)




