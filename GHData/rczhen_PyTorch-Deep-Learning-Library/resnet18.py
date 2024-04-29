import torch
import torch.nn as nn


class Identity(nn.Module):
    """
    定义一个layer，目标是什么也不做。
    好处是在forward里面不需要用if else来判断到底是不是要加一层。
    """

    def _init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    """
    Define a class for the reusable component, residual block.
    结构：两层卷积 + 一个skip connection.
    """

    def __init__(self, in_dim, out_dim, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_dim,
                               out_channels=out_dim,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim,
                               out_channels=out_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

        # Residual block 输入输出相加需要size一样。假如输入输出维度不一致，需要做一次卷积down sample让他们一致。
        # 之前在普通NN中通过一层fully connected layer改变维度再相加，是一回事。
        # 在这里写if else并写一个Identity()：如果输入输出维度不一致，降采样，如果一致，就不做。
        # 目的是，不论那种情况都return一个down_sample的函数，forward里的逻辑就简单些
        if stride == 2 or in_dim != out_dim:
            self.down_sample = nn.Sequential(*[
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_dim)])
        else:
            self.down_sample = Identity()

    def forward(self, x):
        h = x

        # print("Residual Input --", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("Residual #1 --", x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        # print("Residual #2 --", x.shape)

        identity = self.down_sample(h)
        # print("Residual identity --", identity.shape)
        x = x + identity
        x = self.relu(x)
        # print("Residual Output --", x.shape)

        return x


class ResNet18(nn.Module):
    def __init__(self, in_dim=64, num_classes=10):
        super().__init__()  # 继承时需要super init

        # parameters
        self.in_dim = in_dim  # 全程更新的参数。每一个layer/residual block的in_channels数量，即上一层out_channels的数量。

        # stem layers
        self.conv1 = nn.Conv2d(in_channels=3,  # stem层的输入维度取决于输入（图片通道数）
                               out_channels=in_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU()  # 没有参数、不需要学习的层，定义一遍就好。有参数的层每层都要定义

        # blocks
        # 复杂一点的逻辑，可定义函数实现，做到decouple
        self.layer1 = self._make_layer(dim=64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(dim=128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(dim=256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(dim=512, n_blocks=2, stride=2)

        # head layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 让feature map变成想要的尺寸，会根据给定的output size自动做pooling
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, dim, n_blocks, stride):
        layer_list = []  # 一个block由多个layers组成，用list存起来

        # 第一个layer可能需要down sample (applying stride)，所以单独写
        layer_list.append(ResidualBlock(self.in_dim, dim, stride=stride))  # Define a new class for residual block
        self.in_dim = dim  # 加了一个layer需要更新滑动参数in_dim。没写就报错！！

        # 剩下layers都一样，用for loop写
        for i in range(1, n_blocks):
            layer_list.append(ResidualBlock(self.in_dim, dim, stride=1))

        return nn.Sequential(*layer_list)

    def forward(self, x):
        """
        Define network architecture.
        Focus high level structure, while leaving the layer details to __init__().
        """
        # stem layers
        # print("Input --", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("#3 --", x.shape)

        # blocks
        x = self.layer1(x)
        # print("#4 --", x.shape)
        x = self.layer2(x)
        # print("#5 --", x.shape)
        x = self.layer3(x)
        # print("#6 --", x.shape)
        x = self.layer4(x)
        # print("#7 --", x.shape)

        # head layers / classifier
        x = self.avg_pool(x)
        # print("#8 --", x.shape)
        x = x.flatten(1)
        # print("#9 --", x.shape)
        x = self.classifier(x)
        # print("Output --", x.shape)

        return x


def main():
    t = torch.randn([4, 3, 32, 32])  # batch size 4, 3 channels, 32*32 image size.
    # print('Model input shape: ', t.shape)
    model = ResNet18()
    print(model)
    out = model(t)
    print('Model output shape: ', out.shape)


if __name__ == '__main__':
    main()
