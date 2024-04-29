
import torch
import torch.nn.functional as F
import stylegan1.c_dset as dset
import matplotlib.pyplot as plt


dl = dset.create_image_loader_from_path('p/', 256, 1)
img = next(iter(dl))[0]
gray_map = torch.Tensor([0.2989, 0.5870, 0.1140])
gray = torch.einsum('bchw,c->bhw',img,gray_map).view(img.size()[2:])
gray_ = gray.view(1,1,256,256)

l = 0.5

cub_com = torch.where(gray_>0.5,torch.Tensor([1.]),torch.Tensor([0.]))

'''
k_diag1 = torch.Tensor([[1,1,0],[1,0,-1],[0,-1,-1]]).view(1,1,3,3)
sobel_x = torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).view(1,1,3,3)
sobel_y = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).view(1,1,3,3)
conv_sobel1 = F.conv2d(cub_com,sobel_x,padding=1).abs()
conv_sobel2 = F.conv2d(cub_com,sobel_y,padding=1).abs()
conv_sobel3 = conv_sobel1 + conv_sobel2
ft1 = torch.where(conv_sobel3>4,torch.ones(1),torch.zeros(1))
conv_sobel3_2 = F.conv2d(ft1,sobel_x,padding=1).abs() + F.conv2d(ft1,sobel_y,padding=1).abs()
ft2 = torch.where(conv_sobel3_2>2,torch.ones(1),torch.zeros(1))
conv_sobel3_3 = F.conv2d(conv_sobel3_2,sobel_x,padding=1).abs() + F.conv2d(conv_sobel3_2,sobel_y,padding=1).abs()
'''

#plt.imshow(cub_com.view(cub_com.size()[2:]).numpy(),cmap='gray')
#plt.show()
'''
plt.imshow(ft1.view(conv_sobel3.size()[2:]).numpy(),cmap='gray')
plt.show()
plt.imshow(ft2.view(ft2.size()[2:]).numpy(),cmap='gray')
plt.show()
'''
'''
plt.imshow(conv_sobel3_2.view(conv_sobel3_2.size()[2:]).numpy(),cmap='gray')
plt.show()
plt.imshow(conv_sobel3_3.view(conv_sobel3_3.size()[2:]).numpy(),cmap='gray')
plt.show()
'''

def betti1_3x3(level_img):
    ft_ph = torch.where(conv_ph<40267)

def pool_homo2(level_img):
    betti0 = 0
    betti1 = 0
    ph_ker = torch.Tensor([[1,1,1],[1,67,1],[1,1,1]]).view(1,1,3,3)
    ph_ker1_2 = torch.Tensor([[1,1,1],[1,601,1],[1,1,1]]).view(1,1,3,3)
    ph_ker2 = F.conv_transpose2d(ph_ker1_2,ph_ker,stride=2,padding=1,groups=1)
    conv_ph = F.conv2d(level_img,ph_ker2,stride=2)
    ft_ph = torch.where(conv_ph>40867,conv_ph,torch.zeros(1))
    ft_ph = torch.where(ft_ph.int()%601>67,torch.ones(1),torch.zeros(1))
    return ft_ph

ph_ker = torch.Tensor([[1,1,1],[1,67,1],[1,1,1]]).view(1,1,3,3)
ph_ker1_2 = torch.Tensor([[1,1,1],[1,601,1],[1,1,1]]).view(1,1,3,3)
conv_ph = F.conv2d(cub_com,ph_ker,stride=1)
conv_ph = F.conv2d(cub_com,ph_ker,stride=1)
ft_ph1 = torch.where(conv_ph>2,torch.ones(1),torch.zeros(1))
conv_ph3 = F.conv2d(conv_ph,ph_ker1_2,stride=1)
ft_ph2 = pool_homo2(cub_com)
ft_ph2 = pool_homo2(ft_ph2)
ft_ph2 = pool_homo2(ft_ph2)
ft_ph3 = torch.where(conv_ph3>40867, conv_ph3,torch.zeros(1))
ft_ph3 = torch.where(ft_ph3.int()%601>67,torch.ones(1),torch.zeros(1))
#print((ft_ph2 - ft_ph3).nonzero().size())
#plt.imshow(ft_ph1.view(ft_ph1.size()[2:]).numpy(),cmap='gray')
#plt.show()
plt.imshow(ft_ph2.view(ft_ph2.size()[2:]).numpy(),cmap='gray')
plt.show()
plt.imshow(ft_ph3.view(ft_ph3.size()[2:]).numpy(),cmap='gray')
plt.show()
