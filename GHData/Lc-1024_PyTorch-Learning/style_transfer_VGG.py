# 图片风格迁移 Style Transfer
# 用vgg19做特征提取，其中某些层生成的特征作为判定图片风格的标准
# 不断改变target的值，使之风格接近style
# 其中还用到了PLT和plt中的将image转换成Tensor和将Tensor转换成image

import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import argparse
import matplotlib.pyplot as plt


style_path = ".data\\photo\\yourName2.jpg"
content_path = ".data\\photo\\cat.jpg"

# 加载图片，并且处理成相应的格式和大小
def load_image(image_path, device, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    if transform:
        image = transform(image).unsqueeze(0)
    return image.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

content = load_image(content_path, device, transform, max_size=400)
style = load_image(style_path, device, transform, shape=[content.size(2), content.size(3)])
'''
print(content.shape)
print(style.shape)
'''
# 显示图片
unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(10) # pause a bit so that plots are updated
# plt.figure()
# imshow(content[0], title='Image')

def save_image(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save('style_transfer_result.jpg')

# 用vgg19做特征提取，其中某些层生成的特征作为判定图片风格的标准
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

vgg = VGGNet().to(device).eval()
target = content.clone().requires_grad_(True)
optimizer = torch.optim.Adam([target], lr=0.005, betas=[0.5, 0.999])

total_step = 2000
style_weight = 100.
for step in range(total_step):
    target_feature = vgg(target)
    content_feature = vgg(content)
    style_feature = vgg(style)

    style_loss = 0.
    content_loss = 0.
    for f1, f2, f3 in zip(target_feature, content_feature, style_feature):
        content_loss += torch.mean((f1-f2)**2)
        
        _, c, h, w = f1.size()
        f1 = f1.view(c, h*w)
        f3 = f3.view(c, h*w)
        # 计算gram matrix
        f1 = torch.mm(f1, f1.t())
        f3 = torch.mm(f3, f3.t())
        style_loss += torch.mean((f1-f3)**2) / (c*h*w)

    loss = content_loss + style_weight * style_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print("Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}"
             .format(step, total_step, content_loss.item(), style_loss.item()))  

# 将图片恢复正常，数据回归到0-1
denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
img = target.clone().squeeze()
img = denorm(img).clamp_(0, 1)
plt.figure()
imshow(img, title='Target Image')
save_image(img)