from functools import partial

from torch import nn
import torchvision.models as M

### Some model examples:
# resnet18 = M.resnet18
# resnet34 = M.resnet34
# resnet50 = M.resnet50
# resnet101 = M.resnet101
# resnet152 = M.resnet152
# vgg16 = M.vgg16
# vgg16_bn = M.vgg16_bn
# densenet121 = M.densenet121
# densenet161 = M.densenet161
# densenet201 = M.densenet201

class ResNetFinetune(nn.Module):

    # finetune = True

    def __init__(self,
                 num_classes,
                 net_cls=M.resnet50,
                 activation=None,
                 dropout=False):

        super().__init__()
        self.net = net_cls(pretrained=True)
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        if activation:
            self.activation = activation
        else:
            self.activation = False

    def trainable_params(self):
        return self.net.fc.parameters()

    def forward(self, x):

        if self.activation:
            return self.activation(self.net(x))
        else:
            return self.net(x)

class DenseNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.densenet121):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)







































