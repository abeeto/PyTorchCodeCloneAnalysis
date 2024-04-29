from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle

STEPS = 2000


def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

content = load_image('C:\\Users\\45569\\Pictures\\Saved Pictures\\c.jpg',
                     transform, max_size=500)
style = load_image('C:\\Users\\45569\\Pictures\\Saved Pictures\\b.jpg',
                   transform, shape=[content.size(2), content.size(3)])

print(content.shape)
# print(style.shape)   


def show_image(image):
    transforms.ToPILImage()(image.squeeze(0)).resize((500, 500)).show()


def denorm(image):
    d = transforms.Normalize([-2.12, -2.04, -1.80], [4.37, 4.46, 4.44])
    return d(image)

# show_image(content)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        pre = torch.load(r'D:\BaiduNetdiskDownload\models\vgg19-dcbb9e9d.pth')
        self.vgg = models.vgg19(pretrained=False)
        self.vgg.load_state_dict(pre)
        self.vgg = self.vgg.features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)

        return features


vgg = VGGNet().eval()
# features = vgg(content)
# for feat in features:
#     print(feat.shape)

TARGET_TMP_FILE = 'D:\\target'

try:
    with open(TARGET_TMP_FILE, 'rb') as f:
        print('Load the pre-trained target')
        target = pickle.load(f)
except Exception as e:
    print(e)
    target = content.clone().requires_grad_(True)
# show_image(denorm(target.clone().squeeze()))
# exit()
optimizer = torch.optim.Adam([target], lr=0.001, betas=(0.5, 0.999))

for step in range(STEPS):
    target_features = vgg(target)
    content_features = vgg(content)
    style_features = vgg(style)

    content_loss = style_loss = 0.
    for f1, f2, f3 in zip(target_features, content_features, style_features):
        content_loss += torch.mean((f1-f2)**2)
        _, c, h, w = f1.size()
        f1 = f1.view(c, h*w)
        f3 = f3.view(c, h*w)

        f1 = torch.mm(f1, f1.t())
        f3 = torch.mm(f3, f3.t())
        style_loss += torch.mean((f1-f3)**2) / (c*h*w)

    loss = content_loss + style_loss * 1000

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 25 == 0 and step != 0:
        with open(TARGET_TMP_FILE, 'wb') as f:
            pickle.dump(target, f)

    if step % 100 == 0 and step != 0:
        print('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'
              .format(step, STEPS, content_loss.item(), style_loss.item()))
        # show_image(denorm(target.clone().squeeze()))


# vgg = models.vgg19(pretrained=False)
# print(vgg.features)
# for i, j in enumerate(vgg.features):
#     print(i,j)
# for name, layer in vgg.features._modules.items():
#     print(name, layer)
