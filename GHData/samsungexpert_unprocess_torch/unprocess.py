import torch
from torchvision.transfomrs import ToTensor, ToPILImage
from PIL import Image


def random_ccm():
    xyz2cams= xyz2cams = [[[1.0234, -0.2969, -0.2266],
    [-0.5625, 1.6328, -0.0469],
    [-0.0703, 0.2188, 0.6406]],
    [[0.4913, -0.0541, -0.0202],
    [-0.613, 1.3513, 0.2906],
    [-0.1564, 0.2151, 0.7183]],
    [[0.838, -0.263, -0.0639],
    [-0.2887, 1.0725, 0.2496],
    [-0.0627, 0.1427, 0.5438]],
    [[0.6596, -0.2079, -0.0562],
    [-0.4782, 1.3016, 0.1933],
    [-0.097, 0.1581, 0.5181]]]

    num_ccm = len(xyz2cams)
    xyz2cams = torch.Tensor(xyz2cams)
    weights = torch.rand(num_ccm,1,1)
    weight_sum = torch.sum(weights, axis=0)

    xyz2cam = torch.sum(xyz2cams * weights, axis=0) / weight_sum


    rgb2xyz = torch.Tensor([[0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]])

    rgb2cam = xyz2cam * rgb2xyz

    #Normalizes each row

    rgb2cam = rgb2cam / torch.sum(rgb2cam, axis=01, keepdims=True)

    return rgb2cam

def random_gains():
    #RGB gain represens brithening

    rgb_gains = 1.0/ torch.normal(0.8, 0.1,(1,1)).squeeze().unsqueeze(dim=-1)

    red_gain = 1.9 +0.5 *torch.rand(1)
    blue_gain = 1.5 +0.4 * torch.rand(1)

    return rgb_gain,red_gain, blue_gain



