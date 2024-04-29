import  torch
from    torch import  nn
from    torch.nn import functional as F



class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        # we add stride support for resbok, which is distinct from tutorials.
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self, x):
        """

		:param x: [b, ch, h, w]
		:return:
		"""
        # x = x.transpose(1,3).transpose(2,3)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        print('original_extra(x):', self.extra(x).shape)
        print('out:', out.shape)
        out = self.extra(x) + out
        out = F.relu(out)
        
        return out




class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outlayer = nn.Linear(512*1*1, 100)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        """
conv1_hw:
torch.Size([64, 64, 10, 10])
blk1_hw:
torch.Size([64, 128, 5, 5])
blk2_hw:
torch.Size([64, 256, 3, 3])
blk3_hw:
torch.Size([64, 512, 2, 2])
blk4_hw:
torch.Size([64, 512, 2, 2])
avg_hw:
torch.Size([64, 512, 1, 1])
"""
        # print(x.shape)
        x=x.transpose(1,3).transpose(2,3)
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # (32, 64, 10, 10)
        print("conv1_hw:")
        print(x.shape)
        
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        print("blk1_hw:")
        print(x.shape)
        x = self.blk2(x)
        print("blk2_hw:")
        print(x.shape)
        x = self.blk3(x)
        print("blk3_hw:")
        print(x.shape)
        x = self.blk4(x)
        print("blk4_hw:")
        print(x.shape)
        
        
        # print('after conv:', x.shape) #[b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        print("avg_hw:")
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        
        
        return x


class ResNet34(nn.Module):

    def __init__(self):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1_1 = ResBlk(64, 64, stride=2)
        self.blk1_2 = ResBlk(64, 64, stride=2)
        self.blk1_3 = ResBlk(64, 128, stride=1)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2_1 = ResBlk(128, 128, stride=2)
        self.blk2_2 = ResBlk(128, 128, stride=1)
        self.blk2_3 = ResBlk(128, 256, stride=1)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3_1 = ResBlk(256, 256, stride=2)
        self.blk3_2 = ResBlk(256, 256, stride=1)
        self.blk3_3 = ResBlk(256, 512, stride=1)
        # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4_1 = ResBlk(512, 512, stride=2)
        self.blk4_2 = ResBlk(512, 512, stride=1)
        self.blk4_3 = ResBlk(512, 512, stride=1)

        self.outlayer = nn.Linear(512 * 1 * 1, 100)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # print(x.shape)
        x = x.transpose(1, 3).transpose(2, 3)
        # print(x.shape)
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1_1(x)
        x = self.blk1_2(x)
        x = self.blk1_3(x)

        x = self.blk2_1(x)
        x = self.blk2_2(x)
        x = self.blk2_3(x)

        x = self.blk3_1(x)
        x = self.blk3_2(x)
        x = self.blk3_3(x)

        x = self.blk4_1(x)
        x = self.blk4_2(x)
        x = self.blk4_3(x)

        # print('after conv:', x.shape) #[b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x

def main():

    blk = ResBlk(64, 128, stride=4)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('block:', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('resnet:', out.shape)




if __name__ == '__main__':
    main()