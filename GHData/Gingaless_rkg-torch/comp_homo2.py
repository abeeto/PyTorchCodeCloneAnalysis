import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import stylegan1.c_dset as dset


def show_gray_image(img):
    plt.imshow(img.view(img.size()[2:]).numpy(),cmap='gray')
    plt.show()

def comp_ph3x3(img):
    ph_ker = torch.Tensor([1,2,4,8,16,32,64,128,256]).view(1,1,3,3)
    conv_ph = F.conv2d(img,ph_ker)
    betti0 =  torch.where(conv_ph==16,torch.ones(1),torch.zeros(1)).nonzero().size(0)
    betti1 = torch.where(conv_ph==495,torch.ones(1),torch.zeros(1)).nonzero().size(0)
    ft_ph = torch.where((conv_ph!=16) & ((conv_ph//16)%2==1),torch.ones(1),torch.zeros(1))
    return ft_ph, betti0, betti1

def comp_ph_pool(img):
    ph_ker = torch.Tensor([1,2,4,8,16,32,64,128,256]).view(1,1,3,3)
    ph_ker2 = F.conv_transpose2d(F.pad(ph_ker,pad=(1,1,1,1)),ph_ker,stride=1,padding=1,groups=1)
    conv_ph = F.conv2d(img,ph_ker2,padding=2,stride=1)
    ft_ph = torch.where((conv_ph!=8176) & ((conv_ph//511//16)%2==1), torch.ones(1),torch.zeros(1))
    return ft_ph

dl = dset.create_image_loader_from_path('p/', 256, 1)
img = next(iter(dl))[0]
gray_map = torch.Tensor([0.2989, 0.5870, 0.1140])
gray = torch.einsum('bchw,c->bhw',img,gray_map).view(img.size()[2:])
gray_ = gray.view(1,1,256,256)

l = 0.5

cub_com = torch.where(gray_>0.5,torch.Tensor([1.]),torch.Tensor([0.]))

ph_ker = torch.Tensor([1,2,4,8,16,32,64,128,256]).view(1,1,3,3)
ph_ker2 = F.conv_transpose2d(F.pad(ph_ker,pad=(1,1,1,1)),ph_ker,stride=1,padding=1,groups=1)
print([pow(2,i) for i in range(25)])
ph_ker22 = torch.Tensor([pow(2,i) for i in range(25)]).view(1,1,5,5)
print(ph_ker22)
conv_ph = F.conv2d(cub_com,ph_ker)
#print(torch.Tensor([10]).int().bitwise_and(ph_ker.view(-1).int()).ne(0))
ft_ph, betti0, betti1 = comp_ph3x3(cub_com)
print(betti0,betti1)
#show_gray_image(ft_ph)
show_gray_image(comp_ph_pool(cub_com))
'''
x=torch.randint(0,2,(10,1,5,5)).float()
print(x[0])
print(ph_ker2)
conv_ph1 = F.conv2d(x,ph_ker)
conv_ph1 = F.conv2d(conv_ph1,ph_ker)
conv_ph2 = F.conv2d(x,ph_ker2)
print((conv_ph1==conv_ph2).nonzero())
'''