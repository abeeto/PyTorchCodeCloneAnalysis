import torch
from torch import nn
from config import config

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
def vgg_base(channels_img, base=base):
    channels_in = channels_img
    layers_list = []

    for layer in base:
        if isinstance(layer, int):
            layers_list += [nn.Conv2d(channels_in, layer, kernel_size=3, padding=1),
                            nn.ReLU(True)]
            channels_in = layer
        else:
            if layer == "C":
                ceil_mode = True
            else:
                ceil_mode = False
            layers_list += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)]
    
    layers_list += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(base[-1], 1024, kernel_size=3, padding=6, dilation=6),
                    nn.ReLU(True),
                    nn.Conv2d(1024, 1024, kernel_size=1),
                    nn.ReLU(True)]
    return layers_list


def extra_layers(channels_in, batch_norm=False):
    layers_list = []
    layers_list += [
                    # Block 6: B, channels_in, 19, 19 -> B, 512, 10, 10
                    nn.Conv2d(channels_in, 256, kernel_size=1, stride=1),
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    
                    # Block 7: B, 512, 10, 10 -> B, 256, 5, 5
                    nn.Conv2d(512, 128, kernel_size=1, stride=1),
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    
                    # Block 8: B, 256, 5, 5 -> B, 256, 3, 3
                    nn.Conv2d(256, 128, kernel_size=1, stride=1),
                    nn.Conv2d(128, 256, kernel_size=3, stride=1),
                    
                    # Block 9: B, 256, 3, 3 -> B, 256, 1, 1
                    nn.Conv2d(256, 128, kernel_size=1, stride=1),
                    nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    return layers_list


def backbone_part(num_classes, config=config):
    mbox = [len(r)+2 for r in config["ratios"]] # +2是因为还有 2个 scale 为1的 prior_box
    vgg = vgg_base(3)
    extra = extra_layers(1024)

    loc_head = []
    conf_head = []
    for k, v in enumerate([21, -2]):
        loc_head += [nn.Conv2d(vgg[v].out_channels, mbox[k]*4, kernel_size=3, padding=1)]
        conf_head += [nn.Conv2d(vgg[v].out_channels, mbox[k]*num_classes, kernel_size=3, padding=1)]
    
    for k, v in enumerate(extra[1::2], 2):
        loc_head += [nn.Conv2d(v.out_channels, mbox[k]*4, kernel_size=3, padding=1)]
        conf_head += [nn.Conv2d(v.out_channels, mbox[k]*num_classes, kernel_size=3, padding=1)]
    return vgg, extra, loc_head, conf_head


class L2Norm(nn.Module):
    def __init__(self, num_channels, scale=20):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1)*scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        x = torch.div(x,norm)
        return self.weight.expand_as(x) * x


class SSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg, extra, loc_head, conf_head = backbone_part(num_classes)
        self.vgg = nn.ModuleList(vgg)
        self.L2Norm = L2Norm(self.vgg[21].out_channels, 20)
        self.extra = nn.ModuleList(extra)
        self.loc_head = nn.ModuleList(loc_head)
        self.conf_head = nn.ModuleList(conf_head)

    def forward(self, x):
        back_bone_out = []
        loc_out, conf_out = [], []

        # len(self.vgg) = 35
        for layer in self.vgg[0:23]:
            x = layer(x)
        x = self.L2Norm(x)
        back_bone_out += [x]

        for layer in self.vgg[23:]:
            x = layer(x)
        back_bone_out += [x]

        for idx, layer in enumerate(self.extra):
            x = nn.functional.relu(layer(x), True)
            if idx % 2 == 1:
                back_bone_out += [x]
        
        # B, H, W, C
        for x, loc, conf in zip(back_bone_out, self.loc_head, self.conf_head):
            loc_out += [loc(x).permute(0, 2, 3, 1).contiguous()]
            conf_out += [conf(x).permute(0, 2, 3, 1).contiguous()]
        
        # B, H*W*C
        # B = x.size(0)
        # loc_out = torch.cat([i.view(B, -1) for i in loc_out], 1)
        # conf_out = torch.cat([i.view(B, -1) for i in conf_out], 1)

        # loc_out = loc_out.view(B, -1, 4)
        # conf_out = conf_out.view(B, loc_out.size(1), -1)

        return loc_out, conf_out



if __name__ == "__main__":
    b, c, h, w = 2, 3, 300, 300
    num_classes = 5

    ssd = SSD(num_classes)
    x = torch.rand([b, c, h, w])
    loc_out, conf_out = ssd(x)
    feature_map_sizes = [l.shape[1:3] for l in loc_out]
    for i, j, s in zip(loc_out, conf_out, feature_map_sizes):
        print(i.shape, j.shape, s)
        