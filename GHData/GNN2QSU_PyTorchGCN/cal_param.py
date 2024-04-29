import torch
import numpy as np
import torch.nn as nn
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16_bn, vgg16
from torchvision.models.alexnet import alexnet


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        bias = True

        layers = []
        layers.append(nn.Conv3d(3, 64, kernel_size=3, bias=bias))


        layers.append(nn.Conv3d(64, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(128, 128, groups=128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=1, bias=bias))


        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(256, 256, groups=256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=1, bias=bias))


        layers.append(nn.Conv3d(256, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))



        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))
        layers.append(nn.Conv3d(512, 512, groups=512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=1, bias=bias))


        layers.append(nn.Linear(512, 512, bias=bias))
        layers.append(nn.Linear(512, 256, bias=bias))
        layers.append(nn.Linear(256, 3, bias=bias))

        self.features = nn.Sequential(*layers)
        pass

    pass


class Net2(nn.Module):

    def __init__(self):
        super().__init__()
        bias = True

        layers = []
        layers.append(nn.Conv3d(3, 64, kernel_size=3, bias=bias))


        layers.append(nn.Conv3d(64, 128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(128, 128, kernel_size=3, bias=bias))


        layers.append(nn.Conv3d(128, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))


        layers.append(nn.Conv3d(256, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))



        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))


        layers.append(nn.Linear(512, 512, bias=bias))
        layers.append(nn.Linear(512, 256, bias=bias))
        layers.append(nn.Linear(256, 3, bias=bias))

        self.features = nn.Sequential(*layers)
        pass

    pass


class Net3(nn.Module):

    def __init__(self):
        super().__init__()
        bias = True

        layers = []
        layers.append(nn.Conv3d(3, 64, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(64, 128, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(128, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(256, 256, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(256, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))
        layers.append(nn.Conv3d(512, 512, kernel_size=3, bias=bias))

        layers.append(nn.Linear(512, 512, bias=bias))
        layers.append(nn.Linear(512, 256, bias=bias))
        layers.append(nn.Linear(256, 3, bias=bias))

        self.features = nn.Sequential(*layers)
        pass

    pass


class NetVgg16(nn.Module):

    def __init__(self):
        super().__init__()
        vgg = vgg16()
        self.features = vgg.features
        self.classifier = nn.Linear(512, 10, bias=False)
        pass

    pass


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param


if __name__ == '__main__':
    """
    resnet:281548163
    vGG：59903747
    light:22127747
    resnet50 25557032
    vgg16 138357544
    vgg16_bn 138365992
    vgg16 14719808
    vgg16_bn 14728256
    alexnet 61100840
    """
    model = alexnet().to(torch.device("cpu"))
    num = view_model_param(model)
    print(num)
    pass
