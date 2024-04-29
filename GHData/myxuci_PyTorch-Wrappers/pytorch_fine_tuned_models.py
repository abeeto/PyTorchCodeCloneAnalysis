import pretrainedmodels
import torch
from torch import nn
import torchvision.models as M

class ResNetFinetune(nn.Module):

    def __init__(self, output_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.net.fc.in_features, output_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, output_classes)

    def trainable_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        # Old version:
        # return self.net(x)

        # New version, use log_softmax
        return torch.nn.functional.log_softmax(self.net(x))


class DenseNetFinetune(nn.Module):

    def __init__(self, output_classes, net_cls=M.densenet121):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, output_classes)

    def trainable_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)

# Added output_trans_func:
class InceptionV3Finetune(nn.Module):
    finetune = True

    def __init__(self, output_classes, output_trans_func=None):
        super().__init__()
        self.net = M.inception_v3(pretrained=True)
        self.output_trans_func = output_trans_func
        self.net.fc = nn.Linear(self.net.fc.in_features, output_classes)

    def trainable_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        if self.net.training:
            x, _aux_logits = self.net(x)
            if self.output_trans_func is not None:
                return self.output_trans_func(x)
            else:
                return x
        else:
            if self.output_trans_func is not None:
                return self.output_trans_func(self.net(x))
            else:
                return self.net(x)
            
class PNASNet5Larget(nn.Module):
    '''
    Progressive Neural Architecture Search.
    URL: 
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/pnasnet.py
        https://zhuanlan.zhihu.com/p/35050923
    Input image size:
        (331, 331, 3)
    '''
    finetune = True
    
    def __init__(self, output_classes, output_trans_func=None):
        super().__init__()
        self.net = pretrainedmodels.__dict__['pnasnet5large'](1000, pretrained='imagenet')
        self.output_trans_func = output_trans_func
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, output_classes)
        
    def trainable_params(self):
        return self.net.last_linear.parameters()
    
    def forward(self, x):
        if self.net.training:
            # x, _aux_logits = self.net(x)
            x = self.net(x)
            if self.output_trans_func is not None:
                return self.output_trans_func(x)
            else:
                return x
        else:
            if self.output_trans_func is not None:
                return self.output_trans_func(self.net(x))
            else:
                return self.net(x)
            
class SENets(nn.Module):
    '''
    Input image size:
        (331, 331, 3)
    '''
    finetune = True
    
    def __init__(self, output_classes, output_trans_func=None):
        super().__init__()
        self.net = pretrainedmodels.__dict__['senet154'](1000, pretrained='imagenet')
        self.output_trans_func = output_trans_func
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, output_classes)
        
    def trainable_params(self):
        return self.net.last_linear.parameters()
    
    def forward(self, x):
        if self.net.training:
            x, _aux_logits = self.net(x)
            if self.output_trans_func is not None:
                return self.output_trans_func(x)
            else:
                return x
        else:
            if self.output_trans_func is not None:
                return self.output_trans_func(self.net(x))
            else:
                return self.net(x)

class FinetunePretrainedmodels(nn.Module):
    finetune = True

    def __init__(self, output_classes, net_cls, net_kwards):
        super().__init__()
        self.net = net_cls(**net_kwards)
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, output_classes)

    def trainable_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)