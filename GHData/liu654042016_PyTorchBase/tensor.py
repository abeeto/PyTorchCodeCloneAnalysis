import torch

#查看维度
a = torch.randn(3,4)
a.size()

#对tensor 进行reshape: tensor.view
a= torch.randn(3, 4)
b = a.view(2, 6)

#交换tensor维度
a = torch.randn(2, 3, 4)
a = torch.permute(2, 0, 1)

#对tensor 维度进行压缩
a = torch.randn(1, 2, 1, 3, 4)

x = a.squeeze()#去掉维度为1的维度  torch.size([2, 3, 4])
y = a.squeeze(dim = 2)#去掉维度 为1的dim维度，torch.size([1,2,3,4])

#对tensor维度进行扩充 tensor.unsqueeze
#给指定位置加上维度为1的参数
a= torch.randn(2, 3, 4)
x = a.unsqueeze(dim = 1) #torch.Size([2, 1, 3, 4])