import torch
from torch import nn
# from lib.region_layer import RegionLayer
# from lib.replace_region_layer import ReplaceRegionLayer
# from region_layer import RegionLayer
from region_layer_new import RegionLayer_31,RegionLayer_88
from torch.nn import functional as F
import numpy as np


class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class Network_new(nn.Module):
    def __init__(self, class_number=7):
        super(Network_new, self).__init__()

        self.class_number = class_number

        self.extractor1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=1),
            RegionLayer_88(in_channels=64, grid=(4, 4)),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64))

        self.extractor2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            RegionLayer_88(in_channels=128, grid=(4, 4)),

            nn.ReLU(inplace=True))
            # nn.MaxPool2d(kernel_size=2, stride=2))

        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128))

        self.extractor3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            RegionLayer_88(in_channels=256, grid=(4, 4)),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256))

        self.extractor4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            RegionLayer_88(in_channels=512, grid=(4, 4)),

            nn.ReLU(inplace=True))
            # nn.MaxPool2d(kernel_size=2, stride=2))

        self.bottleneck4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512))

        self.extractor5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            RegionLayer_31(in_channels=512, grid=(4, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv=nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool=nn.AvgPool2d(kernel_size=6)

        self.relu = nn.ReLU(inplace=True)

        self.classifier=nn.Sequential(
            nn.Linear(in_features=512, out_features=class_number)
        )
    def forward(self, x):
        """

        :param x:   (b, c, h, w)
        :return:    (b, class_number)
        """

        batch_size = x.size(0)

        x = self.extractor1(x)
        # print(x.shape)
        short_cut=x
        x=self.bottleneck1(x)
        x=self.relu(x+short_cut)
        short_cut = x
        x = self.bottleneck1(x)
        x = self.relu(x + short_cut)

        # print(x.shape)

        x = self.extractor2(x)
        # print(x.shape)
        short_cut = x
        x = self.bottleneck2(x)
        x = self.relu(x + short_cut)
        short_cut = x
        x = self.bottleneck2(x)
        x = self.relu(x + short_cut)
        # print(x.shape)
        x = self.extractor3(x)
        # print(x.shape)
        short_cut = x
        x = self.bottleneck3(x)
        x = self.relu(x + short_cut)
        short_cut = x
        x = self.bottleneck3(x)
        x = self.relu(x + short_cut)
        # print(x.shape)


        x = self.extractor4(x)
        # print(x.shape)
        short_cut = x
        x = self.bottleneck4(x)
        x = self.relu(x + short_cut)
        short_cut = x
        x = self.bottleneck4(x)
        x = self.relu(x + short_cut)
        # print(x.shape)

        x = self.extractor5(x)
        print(x.shape)
        x = self.Conv(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)
        output=self.classifier(x)

        return output

    @staticmethod
    def multi_label_sigmoid_cross_entropy_loss(pred, y, size_average=True):
        """

        :param pred: (b, class)
        :param y: (b, class)
        :return:
        """

        batch_size = pred.size(0)
        pred = nn.Sigmoid()(pred)

        # try:
        # pos_part = (y > 0).float() * torch.log(pred)
        pos_to_log = pred[y > 0]
        pos_to_log[pos_to_log.data == 0] = 1e-20
        pos_part = torch.log(pos_to_log).sum()

        # neg_part = (y < 0).float() * torch.log(1 - pred)
        neg_to_log = 1 - pred[y < 0]
        neg_to_log[neg_to_log.data == 0] = 1e-20
        neg_part = torch.log(neg_to_log).sum()
        # except Exception:
        #     # print(pred[y > 0].min())
        #     # print((1 - pred[y < 0]).min())
        #     pdb.set_trace()

        loss = -(pos_part + neg_part)

        if size_average:
            loss /= batch_size

        return loss

    #@staticmethod
    def statistics(self,pred, y, thresh):
        batch_size = pred.size(0)
        class_nb = pred.size(1)
        # print('y=',y.data)
        # print('pred',pred)
        # print('pred > thresh',pred > thresh)
        pred = pred > thresh
        pred = pred.long()
        pred[pred == 0] = -1
        # print('pred', pred)
        statistics_list = []
        for j in range(class_nb):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(batch_size):
                if pred[i][j] == 1:
                    if y[i][j] == 1:
                        TP += 1
                    elif y[i][j] == 0:
                        FP += 1
                    else:
                        assert False
                elif pred[i][j] == -1:
                    # print('y[i][j]=',y[i][j])
                    if y[i][j] == 1:
                        FN += 1
                    elif y[i][j] == 0:
                        TN += 1
                    else:
                        assert False
                else:
                    assert False
            statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
        return statistics_list

    @staticmethod
    def calc_f1_score(statistics_list):
        f1_score_list = []

        for i in range(len(statistics_list)):
            TP = statistics_list[i]['TP']
            FP = statistics_list[i]['FP']
            FN = statistics_list[i]['FN']

            precise = TP / (TP + FP + 1e-20)
            recall = TP / (TP + FN + 1e-20)
            f1_score = 2 * precise * recall / (precise + recall + 1e-20)
            f1_score_list.append(f1_score)
        mean_f1_score = sum(f1_score_list) / len(f1_score_list)

        return mean_f1_score, f1_score_list

    @staticmethod
    def update_statistics_list(old_list, new_list):
        if not old_list:
            return new_list

        assert len(old_list) == len(new_list)

        for i in range(len(old_list)):
            old_list[i]['TP'] += new_list[i]['TP']
            old_list[i]['FP'] += new_list[i]['FP']
            old_list[i]['TN'] += new_list[i]['TN']
            old_list[i]['FN'] += new_list[i]['FN']

        return old_list

if __name__ == '__main__':
    from torch import nn
    from torch.autograd import Variable
    import numpy as np
    import os
    import sys
    image = Variable(torch.randn(2, 3, 48, 48))
    label = Variable(torch.from_numpy(np.random.randint(3, size=[2, 7]) - 1))
    print('image.shape=',image.shape)
    print('label.shape=',label.shape)

    net = Network_new()
    # print(net)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    while True:
        pred = net(image)
    #
    #     loss = net.multi_label_sigmoid_cross_entropy_loss(pred, label)
    #     print(loss.item())
    #     print('\n')
    #     opt.zero_grad()
    #     loss.backward()
    #     opt.step()




