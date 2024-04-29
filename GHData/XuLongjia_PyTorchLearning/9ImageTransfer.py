from __future__ import division


from torchvision import models
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    if max_size:
        scale = max_size /max(image.size)
        size = np.array(image.size)*scale
        image = image.resize(size.astype(int),Image.ANTIALIAS)

    if shape:
        image = image.resize(shape,Image.LANCZOS)
    print("image size = ",image.size)

    if transform:
        image = transfrom(image).unsqueeze(0)
    print(image.shape)

    return image.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485,0.456,0.406],
        std = [0.229,0.224,0.225]
    )
])

content = load_image('content.jpeg',transform,max_size=400)
style = load_image('style.jpeg',transform,shape=[content.size(3),content.size(2)])

#print(content.shape)
#print(content.size(2),content.size(3))
unloader = transforms.ToPILImage()
plt.ion()
def imshow(tensor,title=None):
    image = tensor.cpu().clone()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
plt.figure()
imshow(style[0],title='StyleImage')
imshow(content[0],title='ContentImage')

#定义一个VGG用来特征提取
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet,self).__init__()
        self.select = ['0','5','10','19','28']
        self.vgg = models.vgg19(pretrained=True).features
    def forward(self,x):
        features = []
        for name,layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

target = content.clone().requires_grad_(True)
optimizer = torch.optim.Adam([target], lr=0.003, betas=[0.5,0.999])
vgg =VGGNet().to(device).eval()

total_step = 2000
style_weight = 100.0
for step in range(total_step):
    target_features = vgg(target)
    content_features = vgg(content)
    style_features = vgg(style)
    style_loss = content_loss = 0
    for f1,f2,f3 in zip(target_features,content_features,style_features):
        content_loss += torch.mean((f1-f2)**2)
        _,c,h,w =f1.size()
        f1 = f1.view(c,h*w)
        f3 = f3.view(c,h*w)
        f1 = torch.mm(f1,f1.t())
        f3 = torch.mm(f3,f3.t())
        style_loss += torch.mean((f1-f3)**2)/(c*h*w)
    loss = content_loss + style_weight*style_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 ==0:
        print('step [{}/{}],content loss : {},style loss : {}'
              .format(step,total_step,content_loss.item(),style_loss.item()))

denorm = transform.Normalize((-2.12,-2.04,-1.80),(4,37,4,46,4,44))
img = target.clone().squeeze()
img = denorm(img).clamp_(0,1)
plt.figure()
imshow(img,title='Target Image')
