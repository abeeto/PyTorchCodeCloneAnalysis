import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as dset

from torch.autograd import Variable
from torch.utils.data import DataLoader
import vgg

def conv_block(name, in_c, out_c, kernel_size=4, stride=2, padding=1, transpose=False, bn=True, leaky_relu=False, bias=True):
    block = nn.Sequential()

    if not transpose:
        block.add_module(name + ' Conv2d', nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=bias))
    else:
        block.add_module(name + ' Conv2d_Transpose' , nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding, bias=bias))
    if bn:
        block.add_module(name + ' Batch_norm' , nn.BatchNorm2d(out_c))
    if leaky_relu:
        block.add_module(name + ' Leaky_ReLU' , nn.LeakyReLU(0.2, inplace=True))
    else:
        block.add_module(name + ' ReLU' , nn.ReLU(inplace=True))
    
    return block


class D(nn.Module):
    def __init__(self, name='D'):
        super(D, self).__init__()
        self.name = name

        self.build()

    def build(self):
        self.layer1_1 = conv_block(self.name + '1', 3, 64, leaky_relu=True, kernel_size=3, bn=False, bias=False)

        self.layer2_1 = conv_block(self.name + '2_1', 64, 128, leaky_relu=True)
        self.layer2_2 = conv_block(self.name + '2_2', 128, 128, leaky_relu=True, kernel_size=3, stride=1)

        self.layer3_1 = conv_block(self.name + '3_1', 128, 256, leaky_relu=True)
        self.layer3_2 = conv_block(self.name + '3_2', 256, 256, leaky_relu=True, kernel_size=3, stride=1)
        self.layer3_3 = conv_block(self.name + '3_3', 256, 256, leaky_relu=True, kernel_size=3, stride=1)

        self.layer4_1 = conv_block(self.name + '4_1', 256, 128, leaky_relu=True) 
        self.layer4_2 = conv_block(self.name + '4_2', 128, 128, leaky_relu=True, kernel_size=3, stride=1)
        self.layer4_3 = conv_block(self.name + '4_3', 128, 128, leaky_relu=True, kernel_size=3, stride=1)

        self.layer5_1 = conv_block(self.name, 128, 64, leaky_relu=True) 
        self.layer5_2 = conv_block(self.name + '5_2', 64, 64, leaky_relu=True, kernel_size=3, stride=1)

        self.layer6 = nn.Sequential()
        self.layer6.add_module(self.name + ' Conv2d' + str(6), nn.Conv2d(64, 1, 4, 2, 0, bias=False))
        self.layer6.add_module(self.name + ' Batcn_norm' + str(6), nn.BatchNorm2d(1))
        self.layer6.add_module(self.name + ' Sigmoid' + str(6), nn.Sigmoid())

    def forward(self, x):
        out1_1 = self.layer1_1(x)
        out2_1 = self.layer2_1(out1_1)
        out2_2 = self.layer2_2(out2_1)
        out3_1 = self.layer3_1(out2_2)
        out3_2 = self.layer3_2(out3_1)
        out3_3 = self.layer3_3(out3_2)
        out4_1 = self.layer4_1(out3_3)
        out4_2 = self.layer4_2(out4_1)
        out4_3 = self.layer4_3(out4_2)
        out5_1 = self.layer5_1(out4_3)
        out5_2 = self.layer5_2(out5_1)
        out6 = self.layer6(out5_2)
        return out6

class G(nn.Module):
    def __init__(self, name='G'):
        super(G, self).__init__()
        self.name = name

        self.build()

    def build(self):
        self.layer1_1 = conv_block(self.name + '1_1', 512, 256, leaky_relu=True, bias=False, transpose=True) # 8
        self.layer1_2 = conv_block(self.name + '1_2', 256, 256, 3, 1, leaky_relu=True, transpose=True)

        self.layer2_1 = conv_block(self.name + '2_1', 256, 128, leaky_relu=True, transpose=True)
        self.layer2_2 = conv_block(self.name + '2_2', 128, 128, 3, 1, leaky_relu=True, transpose=True)
        self.layer2_3 = conv_block(self.name + '2_3', 128, 128, 3, 1, leaky_relu=True, transpose=True)

        self.layer3_1 = conv_block(self.name + '3_1', 128, 64, leaky_relu=True, transpose=True) # 32
        self.layer3_2 = conv_block(self.name + '3_2', 64, 64, 3, 1, leaky_relu=True, transpose=True)

        self.layer4_1 = conv_block(self.name + '4_1', 64, 32, leaky_relu=True, transpose=True) # 64
        self.layer4_2 = conv_block(self.name + '4_2', 32, 32, 3, 1, leaky_relu=True, transpose=True)
        
        self.layer5 = nn.Sequential()
        self.layer5.add_module(self.name + ' Conv2d' + str(5), nn.ConvTranspose2d(32, 3, 4, 2, 1))
        self.layer5.add_module(self.name + ' Tanh' + str(5), nn.Tanh())
        
    def forward(self, x):
        out1_1 = self.layer1_1(x)
        out1_2 = self.layer1_2(out1_1)
        out2_1 = self.layer2_1(out1_2)
        out2_2 = self.layer2_2(out2_1)
        out2_3 = self.layer2_3(out2_2)
        out3_1 = self.layer3_1(out2_3)
        out3_2 = self.layer3_2(out3_1)
        out4_1 = self.layer4_1(out3_2)
        out4_2 = self.layer4_2(out4_1)
        out5_1 = self.layer5(out4_2)
        return out6


class E(nn.Module):
    def __init__(self, model_path, is_pre_train=False):
        super(E, self).__init__()

        self.model_path = model_path
        self.is_pre_train = is_pre_train

        self.build()

    def build(self):
        # load pretrained vgg16_bn 
        def load_vgg(model_path, pretrained=True, **kwargs):
            if pretrained:
                kwargs['init_weights'] = False
            model = vgg.VGG(vgg.make_layers(vgg.cfg['D'], batch_norm=True), **kwargs)
            if pretrained:
                state_dict = torch.load(model_path)
                # remove classifier layers 
                state_dict = {k:v for k, v in state_dict.items() if 'class' not in k}

                model.load_state_dict(state_dict)
            return model.features

        temp = load_vgg(self.model_path)
        self.model = temp

    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == '__main__':
    from data import PokemonDataset
    from torchvision import transforms

    
    transformations = transforms.Compose([transforms.ToTensor()])
    img_dir = "data/pokemon"
    poke_data = dset.ImageFolder(img_dir)
    
    trainer = DataLoader(poke_data, batch_size=10, shuffle=True)

    for i, data in enumerate(trainer):
        print(data.shape)

    for batch_idx, (data, target) in enumerate(trainer):
        e = E('model/vgg16bn.pth')
        asdf = e(Variable(data))
        g = G()
        image = g(asdf)
        d = D()
        print(d(image).shape)
        break

