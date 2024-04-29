# 导入torch框架
import torch
import torch.nn as nn
import math

# VGG11 Model

# 在写自己定义的网络时，要继承torch框架的Module类
class VGG(nn.Module):
    # init方法是构造方法，在这里面我们声明要使用哪些层，进行什么操作
    def __init__(self, num_classes='code1'):  # num_classes指的是我们模型要进行多少个类别的分类，cifar10是10分类任务，故num=10
        super(VGG, self).__init__()
        # 卷积层，in_channels:输入维度；out_channels:输出维度；kernel_size:卷积核大小；stride:步长；padding:是否补0
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 归一化层，num_features:输入feature map的通道数
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=512)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=512)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=512)
        # relu激活函数层，增加非线性，inplace=True表示对原值进行操作，即将得到的值又直接复制到该值中，节省参数空间，对模型训练没有影响；
        # 又因为relu没有参数，所以只需要声明一次就可，我们可以多次使用
        self.relu = nn.ReLU(inplace=True)
        # maxpool层，kernel size:池化核大小 stride：步长 padding：是否补0
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层分类器，in_features:高维抽象特征的数目 out_features:分类数，值得注意的是，我们这里为了节省参数以及训练时间，将vgg的三个全连接层缩减为1个，对于简单的分类任务，这基本上是不影响的
        self.classifier = nn.Linear(in_features=512, out_features=num_classes)

        # 对模型的有参数的层进行初始化操作，像relu和pool层是没有参数的
        # 目前根据实验情况来看，并没有很明确的指出哪种初始化方式效果好，哪种效果不好，需要根据实际情况尝试
        for m in self.modules():
            # 如果是卷积层，就进行高斯分布式的初始化
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            # 如果是归一化层，权重赋值为1，偏置赋值为0
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # 如果是分类层，则进行高斯分布初始化
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    # 我们在init构造方法里说明了我们要使用哪些层，接下来需要在forward函数里进行堆叠，开始搭积木了。
    # VGG的搭积木方法就是一条直线，从前往后，顺序进行即可
    def forward(self, x):
        # [32, 3, 32, 32]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # [32, 64, 32, 32]
        x = self.maxpool(x)
        # [32, 64, 16, 16]
        # 值得注意的是，特别是在VGG中，我们的特征图经过卷积层之后的大小是不变的，只有经过最大池化层才将H×W进行减半操作，这是2014年vgg的特点
        # 多次使用最大池化层很有可能造成信息的丢失。在以后提出的网络中基本上很少使用pool，或者使用修改后的池化操作。

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # [32, 128, 16, 16]
        x = self.maxpool(x)
        # [32, 128, 8, 8]

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # [32, 256, 8, 8]
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # [32, 256, 8, 8]
        x = self.maxpool(x)
        # [32, 256, 4, 4]

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        # [32, 256, 4, 4]
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        # [32, 256, 4, 4]
        x = self.maxpool(x)
        # [32, 512, 2, 2]

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        # [32, 512, 2, 2]
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        # [32, 512, 2, 2]
        x = self.maxpool(x)
        # [32, 512, 1, 1]
        # 在这里进行reshpe，将BCHW的特征图转化为高维语义信息，B×num_features
        x = x.view(x.size(0), -1)
        # [32, 512]
        x = self.classifier(x)
        return x


# ResNet18
# 在写自己定义的网络时，要继承torch框架的Module类
class ResNet(nn.Module):
    # init方法是构造方法，在这里面我们声明要使用哪些层，进行什么操作
    def __init__(self, num_classes='code2'):  # num_classes指的是我们模型要进行多少个类别的分类，cifar10是10分类任务，故num=10
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 还是卷积+bn+relu（relu没有参数，一次声明多次使用）的组合
        # 但这里我们对层名中加了stage，因为resnet是分模块的，无论是resnet18，34，50，101等等，都是分为4个stage，划分的标准是：stage内的层输出的特征图的shape都是一样的
        self.stage1_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.stage1_bn1 = nn.BatchNorm2d(64)
        self.stage1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.stage1_bn2 = nn.BatchNorm2d(64)
        # ResNet是残差网络，残差是发生在维度不匹配的情况，即F(x)与x的channel对不上，这时候就需要进行对齐操作
        # 在第一个stage中通道维度都是一样的，64，在下面的forward函数里可以看出
        self.stage1_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.stage1_bn3 = nn.BatchNorm2d(64)
        self.stage1_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.stage1_bn4 = nn.BatchNorm2d(64)

        self.stage2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.stage2_bn1 = nn.BatchNorm2d(128)
        self.stage2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.stage2_bn2 = nn.BatchNorm2d(128)
        # 对齐
        self.stage2_shortcut_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.stage2_shortcut_bn = nn.BatchNorm2d(128)

        self.stage2_conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.stage2_bn3 = nn.BatchNorm2d(128)
        self.stage2_conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.stage2_bn4 = nn.BatchNorm2d(128)

        self.stage3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.stage3_bn1 = nn.BatchNorm2d(256)
        self.stage3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.stage3_bn2 = nn.BatchNorm2d(256)
        # 对齐
        self.stage3_shortcut_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)
        self.stage3_shortcut_bn = nn.BatchNorm2d(256)

        self.stage3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.stage3_bn3 = nn.BatchNorm2d(256)
        self.stage3_conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.stage3_bn4 = nn.BatchNorm2d(256)

        self.stage4_conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.stage4_bn1 = nn.BatchNorm2d(512)
        self.stage4_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.stage4_bn2 = nn.BatchNorm2d(512)
        # 对齐
        self.stage4_shortcut_conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)
        self.stage4_shortcut_bn = nn.BatchNorm2d(512)

        self.stage4_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.stage4_bn3 = nn.BatchNorm2d(512)
        self.stage4_conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.stage4_bn4 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)
        # 初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # [32, 3, 32, 32]
        x_identity = self.relu(self.bn1(self.conv1(x)))
        # [32, 64, 32, 32]
        # Stage 1，上层传下来的x_identity是[32, 64, 32, 32]
        x = self.relu(self.stage1_bn1(self.stage1_conv1(x_identity)))
        # [32, 64, 32, 32]
        x = self.relu(self.stage1_bn2(self.stage1_conv2(x)))
        # [32, 64, 32, 32]
        # x是[32, 64, 32, 32]，x_identity是[32, 64, 32, 32]，x与x_identity维度匹配，可以直接相加，故不需要额外操作进行对齐
        x = x + x_identity
        x = self.relu(self.stage1_bn3(self.stage1_conv3(x)))
        # [32, 64, 32, 32]
        x_identity = self.relu(self.stage1_bn4(self.stage1_conv4(x)))
        # [32, 64, 32, 32]
        # Stage 2，上层传下来的x_identity是[32, 64, 32, 32]
        x = self.relu(self.stage2_bn1(self.stage2_conv1(x_identity)))
        # [32, 128, 16, 16]
        x = self.relu(self.stage2_bn2(self.stage2_conv2(x)))
        # [32, 128, 16, 16]
        # x是[32, 128, 16, 16]，x_identity是[32, 64, 32, 32]，x与x_identity维度不匹配，不可以直接相加，故需要额外操作进行对齐
        x_identity = self.relu(self.stage2_shortcut_bn(self.stage2_shortcut_conv(x_identity)))  # 对齐操作，生成新的identity
        # [32, 128, 16, 16]
        x = x + x_identity  # x是[32, 128, 16, 16]，x_identity是[32, 128, 16, 16]，进行加和操作
        x = self.relu(self.stage2_bn3(self.stage2_conv3(x)))
        # [32, 128, 16, 16]
        x_identity = self.relu(self.stage2_bn4(self.stage2_conv4(x)))
        # [32, 128, 16, 16]
        # Stage 3，上层传下来的x_identity是[32, 128, 16, 16]
        x = self.relu(self.stage3_bn1(self.stage3_conv1(x_identity)))
        # [32, 256, 8, 8]
        x = self.relu(self.stage3_bn2(self.stage3_conv2(x)))
        # [32, 256, 8, 8]
        # x是[32, 256, 8, 8]，x_identity是[32, 128, 16, 16]，x与x_identity维度不匹配，不可以直接相加，故需要额外操作进行对齐
        x_identity = self.relu(self.stage3_shortcut_bn(self.stage3_shortcut_conv(x_identity)))  # 对齐操作，生成新的identity
        # [32, 256, 8, 8]
        x = x + x_identity  # # x是[32, 256, 8, 8]，x_identity是[32, 256, 8, 8]，进行加和操作
        x = self.relu(self.stage3_bn3(self.stage3_conv3(x)))
        # [32, 256, 8, 8]
        x_identity = self.relu(self.stage3_bn4(self.stage3_conv4(x)))
        # [32, 256, 8, 8]
        # Stage 4， 上层传下来的x_identity是[32, 256, 8, 8]
        x = self.relu(self.stage4_bn1(self.stage4_conv1(x_identity)))
        # [32, 512, 4, 4]
        x = self.relu(self.stage4_bn2(self.stage4_conv2(x)))
        # [32, 512, 4, 4]
        # x是[32, 512, 4, 4]，x_identity是[32, 256, 8, 8]，x与x_identity维度不匹配，不可以直接相加，故需要额外操作进行对齐
        x_identity = self.relu(self.stage4_shortcut_bn(self.stage4_shortcut_conv(x_identity)))  # 对齐操作，生成新的identity
        # [32, 512, 4, 4]
        x = x + x_identity  # x是[32, 512, 4, 4]，x_identity是[32, 512, 4, 4]，进行加和操作
        x = self.relu(self.stage4_bn3(self.stage4_conv3(x)))
        # [32, 512, 4, 4]
        x = self.relu(self.stage4_bn4(self.stage4_conv4(x)))
        # [32, 512, 4, 4]

        x = self.avg(x)
        # [32, 512, 1, 1]
        x = x.view(x.size(0), -1)
        # [32, 512]
        x = self.linear(x)
        # [32, 10]
        return x

# 从resnte搭积木的方法来看，是没有用到最大池化层的，特征图的H×W缩减都是通过卷积操作的步长和kernel size进行协调的
# 而且从宏观的角度来看，resnet和vgg也是一样，网络层按顺序进行组合，将模型加深；接下来的googlenet在一定程度上对模型的宽度进行了扩展


# GoogLeNetv2 Model

# googlenet是由多个Inception组成的，我们这里v2是调用了9次Inception，每个Inception的内部由四个branch组成
# 分别为1x1 conv branch，3x3 conv branch，5x5 conv branch，3x3 pool，然后对其输出沿通道维度进行堆叠
class Inception(nn.Module):
    def __init__(self, in_planes, b1_out_planes, b2_med_planes, b2_out_planes, b3_med_planes, b3_out_planes, pool_planes):
        # in_planes:inception的输入维度；b1_out_planes：branch 1的输出维度；
        # b2_med_planes：branch 2的中间层输出维度，b2_out_planes：branch 2的输出维度
        # b3_med_planes：branch 3的中间层输出维度，b3_out_planes：branch 3的输出维度
        # pool_planes：branch 4的输出维度
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.branch1 = nn.Sequential(  # Sequential将多个层组合在一起，表达起来更方便
            nn.Conv2d(in_planes, b1_out_planes, kernel_size=1),
            nn.BatchNorm2d(b1_out_planes),
            nn.ReLU(True),
        )
        # 先通过1x1 conv将通道维度降下来，再使用3x3 conv，减少计算量，增加非线性
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, b2_med_planes, kernel_size=1),
            nn.BatchNorm2d(b2_med_planes),
            nn.ReLU(True),
            nn.Conv2d(b2_med_planes, b2_out_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(b2_out_planes),
            nn.ReLU(True),
        )
        # 先通过1x1 conv将通道维度降下来，再使用两个3x3 conv代替5x5 conv，减少计算量，增加非线性
        # 1x1 conv -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, b3_med_planes, kernel_size=1),
            nn.BatchNorm2d(b3_med_planes),
            nn.ReLU(True),
            nn.Conv2d(b3_med_planes, b3_out_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(b3_out_planes),
            nn.ReLU(True),
            nn.Conv2d(b3_out_planes, b3_out_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(b3_out_planes),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        # 我们把这个Inception的输入x分别放到分支1234中，并将其结果沿通道维度进行堆叠，得到最终的输出
        y = torch.cat([y1, y2, y3, y4], 1)
        return y


class GoogLeNet(nn.Module):
    def __init__(self, num_classes='code3'):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # 我们的googlenetv2中调用了9次inception结构
        self.layer1 = Inception(in_planes=192, b1_out_planes=64, b2_med_planes=96, b2_out_planes=128, b3_med_planes=16, b3_out_planes=32, pool_planes=32)
        self.layer2 = Inception(in_planes=256, b1_out_planes=128, b2_med_planes=128, b2_out_planes=192, b3_med_planes=32, b3_out_planes=96, pool_planes=64)

        self.layer3 = Inception(in_planes=480, b1_out_planes=192, b2_med_planes=96, b2_out_planes=208, b3_med_planes=16, b3_out_planes=48, pool_planes=64)
        self.layer4 = Inception(in_planes=512, b1_out_planes=160, b2_med_planes=112, b2_out_planes=224, b3_med_planes=24, b3_out_planes=64, pool_planes=64)
        self.layer5 = Inception(in_planes=512, b1_out_planes=128, b2_med_planes=128, b2_out_planes=256, b3_med_planes=24, b3_out_planes=64, pool_planes=64)
        self.layer6 = Inception(in_planes=512, b1_out_planes=112, b2_med_planes=144, b2_out_planes=288, b3_med_planes=32, b3_out_planes=64, pool_planes=64)
        self.layer7 = Inception(in_planes=528, b1_out_planes=256, b2_med_planes=60, b2_out_planes=320, b3_med_planes=32, b3_out_planes=128, pool_planes=128)

        self.layer8 = Inception(in_planes=832, b1_out_planes=256, b2_med_planes=160, b2_out_planes=320, b3_med_planes=32, b3_out_planes=128, pool_planes=128)
        self.layer9 = Inception(in_planes=832, b1_out_planes=384, b2_med_planes=192, b2_out_planes=384, b3_med_planes=48, b3_out_planes=128, pool_planes=128)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # 如果从宏观上看，googlenet也是一个通过Inception模块不断加深的结构，但Inception相比于其他结构有一定宽度。
        out = self.pre_layers(x)
        # [32, 192, 32, 32]
        out = self.layer1(out)
        # [32, 256, 32, 32]
        out = self.layer2(out)
        # [32, 480, 32, 32]
        out = self.maxpool(out)  # maxpool对H×W减半
        # [32, 480, 16, 16]
        out = self.layer3(out)
        # [32, 512, 16, 16]
        out = self.layer4(out)
        # [32, 512, 16, 16]
        out = self.layer5(out)
        # [32, 512, 16, 16]
        out = self.layer6(out)
        # [32, 528, 16, 16]
        out = self.layer7(out)
        # [32, 832, 16, 16]
        out = self.maxpool(out)  # maxpool对H×W减半
        # [32, 832, 8, 8]
        out = self.layer8(out)
        # [32, 832, 8, 8]
        out = self.layer9(out)
        # [32, 1024, 8, 8]
        out = self.avgpool(out)
        # [32, 1024, 1, 1]
        out = out.view(out.size(0), -1)
        # [32, 1024]
        out = self.linear(out)
        return out

# 14年的googlenet和vgg还是比较广泛的使用maxpool进行特征图缩减的，在以后的文章中，基本上都会避免pool的使用，信息丢失有些严重


def googlenet():
    net = GoogLeNet()
    return net

def resnet():
    net = ResNet()
    return net

def vgg():
    net = VGG()
    return net



