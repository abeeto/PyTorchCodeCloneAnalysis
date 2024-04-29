import torch
import torch.nn as nn 
import numpy as np 
from resnet import resnet50 
from fpn import FPN 


'''
引用 'Focal Loss for Dense Object Detection' 论文原文：
'For denser scale cover- age than in [20], at each level we add anchors of sizes 
{2^0, 2^(1/3), 2^(2/3)} of the original set of 3 aspect ratio anchors. This improve 
AP in our setting. In total there are A = 9 anchors per level and across levels they 
cover the scale range 32 - 813 pixels with respect to the network’s input image. '

分类与位置回归过程中，特征图的每个点预测9个anchor，中间特征图的通道数保持256
'''
   
class Classifier(nn.Module):
    '''
    分类器
    '''
    def __init__(self, fpn_channel=256, class_num=80, anchor_num=9):
        super(Classifier, self).__init__() 
        '''
        引用 'Focal Loss for Dense Object Detection' 论文原文：
        'Taking an input feature map with C channels from a given pyramid level, 
        the subnet applies four 3×3 conv layers, each with C filters and each 
        followed by ReLU activations, followed by a 3×3 conv layer with KA filters.'
        '''
        self.class_num = class_num
        self.anchor_num = anchor_num
        self.subnet = nn.Sequential(
            nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, padding=1), 
            nn.ReLU(), 
        )
        self.classifier = nn.Conv2d(fpn_channel, anchor_num * class_num, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.freeze_bn()
    
    def freeze_bn(self):
        '''
        Freeze BatchNorm layers.
        使用ImageNet的预训练网络的时候，要frozenBN层
        '''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, feature_map):
        x = feature_map 
        x = self.subnet(x)
        x = self.classifier(x)
        y = self.sigmoid(x)
        
        batch_size, channel, width, height = y.size()
        # 转换y格式，(batch, channel, width, height) -> (batch, width, height, channel)
        y1 = y.permute(0, 2, 3, 1)

        # (batch, width, height, channel) -> (batch, width, height, anchor, class)
        class_num = self.class_num
        anchor_num = self.anchor_num
        y2 = y1.view(batch_size, width, height, anchor_num, class_num)

        '''
        调用view之前最好先contiguous
        x.contiguous().view() 
        因为view需要tensor的内存是整块的 
        contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, 
        permute等，需要用contiguous()来返回一个contiguous copy。 
        一种可能的解释是： 
        有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于
        内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。 
        判断是否contiguous用torch.Tensor.is_contiguous()函数。
        ————————————————
        版权声明：本文为CSDN博主「ShellCollector」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
        原文链接：https://blog.csdn.net/jacke121/article/details/80824575
        '''
        # (batch, width, height, anchor, class) -> (batch, n, class) 
        # n = width * height * anchor
        output = y2.contiguous().view(batch_size, -1, class_num)
        return output

class Localizer(nn.Module):
    '''
    位置回归 
    '''
    def __init__(self, fpn_channel=256, anchor_num=9):
        super(Localizer, self).__init__()
        '''
        引用 'Focal Loss for Dense Object Detection' 论文原文：
        'The design of the box regression subnet is identical to the 
        classification subnet except that it terminates in 4A linear 
        outputs per spatial location'
        '''
        self.subnet = nn.Sequential(
            nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, padding=1), 
            nn.ReLU(), 
        )
        self.localizer = nn.Conv2d(fpn_channel, anchor_num * 4, kernel_size=3, padding=1)
   
    def forward(self, feature_map):
        x = feature_map 
        x = self.subnet(x)
        y = self.localizer(x)

        batch_size, channel, width, height = y.size()
        # 转换y格式，(batch, channel, width, height) -> (batch, width, height, channel)
        y1 = y.permute(0, 2, 3, 1)

        # (batch, width, height, channel) -> (batch, n, 4)
        # channel = anchor * 4, n = width * height * anchor
        output = y1.contiguous().view(batch_size, -1, 4)
        return output

class RetinaNet(nn.Module):
    '''
    Retina Net
    '''
    def __init__(self, training=True, fpn_channel=256, class_num=80, anchor_num=9):
        super(RetinaNet, self).__init__()
        self.resnet = resnet50(pretrained=training) 
        self.fpn = FPN(fpn_channel=256) 
        self.classifier = Classifier(fpn_channel, class_num, anchor_num) 
        self.localizer = Localizer(fpn_channel, anchor_num) 

        self.initialization()

    def initialization(self):
        '''
        引用 'Focal Loss for Dense Object Detection' 论文原文：
        'All new conv layers except the final one in the RetinaNet subnets are 
        initialized with bias b = 0 and a Gaussian weight fill with σ = 0.01. 
        For the final conv layer of the classification subnet, we set the bias 
        initialization to b = − log((1 − π)/π), where π specifies that at 
        the start of training every anchor should be labeled as foreground with 
        confidence of ∼π. We use π = .01 in all experiments, although results 
        are robust to the exact value. '
        '''
        # 初始化神经网络参数
        theta = 0.01
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, theta)
                # 我们也可以使用更加科学的参数初始化方法如下：
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # 根据论文，初始化 分类器 和 位置回归 的最后一层卷积层的参数
        PI = 0.01
        self.classifier.classifier.weight.data.fill_(0)
        self.classifier.classifier.bias.data.fill_(-np.log((1.0 - PI) / PI))
        self.localizer.localizer.weight.data.fill_(0)
        self.localizer.localizer.bias.data.fill_(0)

    def forward(self, images):
        C3, C4, C5 = self.resnet(images)

        # 获得多尺度特征图，feature_map = [P3, P4, P5, P6, P7]
        feature_maps = self.fpn(C3, C4, C5)

        # 保存每个尺度下的分类结果和位置回归结果
        classifications = []
        localizations = [] 
        for feature_map in feature_maps:
            classifications.append(self.classifier(feature_map)) 
            localizations.append(self.localizer(feature_map))

        # classifications和localizations中，每个结果格式均为(batch, n, 4)，
        # 其中，每个结果中 n 的数值都不同，因此可以在 n 这个维度上进行合并
        # (batch, n, 4) of 5 scales -> (batch, N, 4)
        # 当输入图片大小为640*640时，N = 76725
        classification = torch.cat(classifications, dim=1)
        localization = torch.cat(localizations, dim=1)

        return classification, localization
        

if __name__ == '__main__':
    import numpy as np 
    fake_images = np.random.uniform(size=[1, 3, 640, 640])
    fake_images = torch.Tensor(fake_images)
    print(fake_images.size())

    retinanet = RetinaNet()
    classification, localization = retinanet(fake_images)
    print('classification:', classification.size())
    print('localization:', localization.size())

'''
torch.Size([1, 3, 640, 640])
classification: torch.Size([1, 76725, 80])
localization: torch.Size([1, 76725, 4])
'''