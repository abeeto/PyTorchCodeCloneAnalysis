import numpy as np
import torch.nn as nn
import math, torch
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from torch.nn import functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class tlayer1(nn.Module):
    def __init__(self, block, planes, blocks=3, stride=1):
        super(tlayer1, self).__init__()
        self.inplanes = 64
        self.downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion))
        self.block1 = block(self.inplanes, planes, stride, downsample=self.downsample)
        self.block2 = block(self.inplanes * 4, planes)
        self.block3 = block(self.inplanes * 4, planes)

    # ##########nhuk#################################### testing layers
    # def forward(self, x):
    #     intermediate_features = {}
    #     x = self.block1(x)
    #     intermediate_features['block1'] = x
    #     x = self.block2(x)
    #     intermediate_features['block2'] = x
    #     out = self.block3(x)
    #     return out, intermediate_features
    # ##########nhuk####################################

    ##########nhuk#################################### original one
    def forward(self, x):
        intermediate_features = {}
        x = self.block1(x)
        print("block1: ", x.shape)
        intermediate_features['block1'] = x
        x = self.block2(x)
        print("block2: ", x.shape)
        intermediate_features['block2'] = x
        out = self.block3(x)
        print("block3: ", out.shape)
        return out
    ##########nhuk####################################


class ResNet(nn.Module):
    # layers = [3, 4, 6, 3]
    def __init__(self, block, layers, num_classes=1000, train=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.istrain = train

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = tlayer1(block,64)  # todo:delete it
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        # self.avgpool = nn.AvgPool2d((16,8), stride=1)

        self.num_features = 512
        self.feat = nn.Linear(512 * block.expansion, self.num_features)
        init.kaiming_normal_(self.feat.weight, mode='fan_out')
        init.constant_(self.feat.bias, 0)

        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.classifier = nn.Linear(self.num_features, num_classes)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        intermediate_features = {}
        print("######################################## t_resnet.construct")
        print("before conv1:", x.shape)
        x = self.conv1(x)
        print("t_resnet_conv1:", x.shape)
        intermediate_features['conv1'] = x
        x = self.bn1(x)
        print("t_resnet_bn1:", x.shape)
        intermediate_features['bn1'] = x
        x = self.relu(x)
        print("t_resnet_relu1:", x.shape)
        intermediate_features['relu1'] = x
        x = self.maxpool(x)
        print("t_resnet_maxpool1:", x.shape)

        x = self.layer1(x)
        intermediate_features['layer1'] = x
        x = self.layer2(x)
        intermediate_features['layer2'] = x
        x = self.layer3(x)
        intermediate_features['layer3'] = x
        x = self.layer4(x)

        x = F.avg_pool2d(x, x.size()[2:])
        intermediate_features['avgpool'] = x
        x = x.view(x.size(0), -1)

        x = self.feat(x)
        fea = self.feat_bn(x)
        intermediate_features['feat'] = fea
        fea_norm = F.normalize(fea)
        intermediate_features['feat_norm'] = fea_norm

        x = F.relu(fea)
        x = self.classifier(x)

        return x, intermediate_features


def resnet50(pretrained=None, num_classes=1000, train=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, train)
    weight = torch.load(pretrained)
    static = model.state_dict()

    base_param = []
    for name, param in weight.items():
        if name not in static:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        static[name].copy_(param)
        base_param.append(name)

    params = []
    params_dict = dict(model.named_parameters())
    for key, v in params_dict.items():
        if key in base_param:
            params += [{'params': v, 'lr_mult': 1}]
        else:
            # new parameter have larger learning rate
            params += [{'params': v, 'lr_mult': 10}]

    return model, params


if __name__ == '__main__':
    input = torch.randn(1, 3, 224, 224)
    bottle_net = Bottleneck(64, 64, stride=1, downsample=None)
    torch_output = bottle_net.forward(input)
    # change to numpy
    torch_output = torch_output.numpy()
    print(torch_output.shape)

    baseline_intput = np.zeros((1, 3, 224, 224))
    # compare two numpy array
    print(np.allclose(torch_output, baseline_intput))

    # snapshot = r'../evalution/resnet50_duke2market_epoch00100.pth'
    # model, _ = resnet50(pretrained=snapshot, num_classes=702)
    # with open("../rubb/torch_model.txt", "w") as f:
    #     f.write(str(model))
    # with open("../rubb/torch_params.txt", "w") as f:
    #     f.write(str(model.state_dict().keys()))
    # print(model)
    # oooodict = model.state_dict().keys()
    # print(model.state_dict().keys())

    ################################################## generate mindspore model
    # ms_out_dir = "../pretrained/resnet50_duke2market"
    # model.eval()
    # # np.zeros((1, 3, 256, 128)).astype(np.int64)
    # mm=torch.randn(6, 3, 256, 128)
    #
    # y= model(mm)
    # mm = (mm,)
    # # x = torch.tensor(np.zeros((1, 3, 256, 128)).astype(np.int32))
    # # add x to tuple
    # # x = (x,)
    # # print(type(x))
    # pytorch2mindspore(model, mm, ms_out_dir)
    ##################################################
