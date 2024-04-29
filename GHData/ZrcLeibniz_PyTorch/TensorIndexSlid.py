import torch

a = torch.rand(4, 3, 28, 28)
print(a[0].shape)
print(a[0, 0].shape)
print(a[0, 0, 2, 4])
print(a[:2].shape)
print(a[:2, :1, :, :].shape)

# 需求:对输入的图片采样--每隔一个点采用一次
b = a[:, :, 0:28:2, 0:28:2]
print(b.shape)

# 要拿出第0个维度的第零个和第2个
select = a.index_select(0, torch.tensor([0, 2]))
print(select.shape)

# 要拿出第1维度上的第1个和第2个
index_select = a.index_select(1, torch.tensor([1, 2]))
print(index_select.shape)

# 要对第2维度进行裁剪，只要8行
a_index_select = a.index_select(2, torch.arange(8))
print(a_index_select.shape)


